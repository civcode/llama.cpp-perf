#!/usr/bin/env python3
"""Bounded teacher-forced next-token loss assessment for aligned QSA logits.

Input: two manifests already emitted by LLAMA_QSA_LOGITS_DIR with exact
matching histories, and a separate JSONL reference labels file. The label
producer MUST establish which actual next token each target-logit row predicts
under an explicitly fixed held-out/teacher-forced schedule. Do not infer gold
tokens from sampled output or MTP acceptance.

Reference JSONL record:
  {"step":0,"phase":"verify_logits","seq":0,"position":32809,
   "label_token":1234,"origin":"heldout_reference"}

Only origin=heldout_reference or teacher_forced_reference is accepted. The
origin is declarative; it does not by itself verify dataset provenance.

  python3 docs/LLAMA-PERF/tools/compare_qsa_reference_nll.py \
    dense-logits/manifest.jsonl sparse-logits/manifest.jsonl reference.jsonl

Outputs signed sparse-minus-dense full-vocabulary negative log-likelihood
(NLL) deltas, reference-token rank/accuracy, and examples. There is no
automatic fidelity PASS criterion; reference-token NLL is only one metric.
All tensors and reference token IDs remain outside Git.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from compare_qsa_model_state import read_manifest, values


def event_key(item: dict) -> tuple[int, str, int, int]:
    return (item["step"], item["phase"], item["seq"], item["position"])


def read_labels(path: Path, max_rows: int) -> dict[tuple, dict]:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("reference label file exceeds 2 MiB")
    out = {}
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    raise ValueError("expected an object")
                key = event_key(rec)
                if (type(key[0]) is not int or key[0] < 0
                        or type(key[1]) is not str or not key[1]
                        or type(key[2]) is not int or key[2] < 0
                        or type(key[3]) is not int or key[3] < 0):
                    raise ValueError("invalid checkpoint identity")
                if type(rec["label_token"]) is not int or rec["label_token"] < 0:
                    raise ValueError("label_token must be a nonnegative integer")
                if rec["origin"] not in ("heldout_reference", "teacher_forced_reference"):
                    raise ValueError("reference label origin must be explicit")
                if key in out:
                    raise ValueError("duplicate label checkpoint")
                out[key] = rec
                if len(out) > max_rows:
                    raise ValueError("reference row count exceeds bound")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as err:
                raise ValueError(f"{path}:{lineno}: {err}") from err
    if not out:
        raise ValueError("reference label file has no rows")
    return out


def row_nll(logits, target: int) -> dict:
    if not logits or not (0 <= target < len(logits)):
        raise ValueError(f"reference token {target} outside logit vocabulary")
    if any(not math.isfinite(float(v)) for v in logits):
        raise ValueError("nonfinite logits")
    gold = float(logits[target])
    largest = max(float(x) for x in logits)
    log_z = largest + math.log(math.fsum(math.exp(float(x) - largest) for x in logits))
    nll = log_z - gold
    rank = 1 + sum(float(v) > gold or (float(v) == gold and i < target)
                   for i, v in enumerate(logits) if i != target)
    winner = max(range(len(logits)), key=lambda i: (logits[i], -i))
    return {"nll_nats": nll, "reference_token_rank": rank,
            "top1": winner, "reference_is_top1": winner == target,
            "reference_probability": math.exp(gold - log_z)}


def _median(xs: list[float]) -> float:
    vals = sorted(xs)
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def compare(dense_manifest: Path, sparse_manifest: Path, labels_file: Path,
            max_rows: int = 64, max_examples: int = 8) -> dict:
    if not (1 <= max_rows <= 256 and 0 <= max_examples <= 64):
        raise ValueError("max rows must be 1..256 and max examples 0..64")
    ev_a, ts_a = read_manifest(dense_manifest)
    ev_b, ts_b = read_manifest(sparse_manifest)
    if ev_a != ev_b:
        raise ValueError("dense and sparse token/candidate/acceptance histories differ")
    a_keys = {key for key in ts_a if key[-1] == "logits"}
    b_keys = {key for key in ts_b if key[-1] == "logits"}
    if not a_keys or a_keys != b_keys or len(a_keys) > max_rows:
        raise ValueError("logits snapshot keys differ, are absent, or exceed row bound")
    labels = read_labels(labels_file, max_rows)
    if {key[:4] for key in a_keys} != set(labels):
        raise ValueError("reference labels must match every logit checkpoint exactly")

    summary = {
        "dense_manifest": str(dense_manifest),
        "sparse_manifest": str(sparse_manifest),
        "reference_labels": str(labels_file),
        "rows": len(a_keys),
        "label_origins": sorted({r["origin"] for r in labels.values()}),
        "dense_mean_nll_nats": 0.0, "sparse_mean_nll_nats": 0.0,
        "mean_sparse_minus_dense_nll_nats": 0.0,
        "median_sparse_minus_dense_nll_nats": 0.0,
        "min_sparse_minus_dense_nll_nats": 0.0,
        "max_sparse_minus_dense_nll_nats": 0.0,
        "sparse_rows_worse_nll": 0, "sparse_rows_better_nll": 0,
        "dense_reference_top1_hits": 0, "sparse_reference_top1_hits": 0,
        "top1_choice_changed_rows": 0,
        "first_top1_change": None, "worst_sparse_nll_regression": None,
        "examples": [],
        "note": ("This is a fixed-reference next-token NLL diagnostic, "
                 "not a validated fidelity or downstream task-quality threshold."),
    }
    dense_total = sparse_total = 0.0
    deltas = []
    for key in sorted(a_keys):
        a, b = ts_a[key], ts_b[key]
        if (a["dtype"] != "f32" or b["dtype"] != "f32"
                or a["shape"] != b["shape"] or len(a["shape"]) != 1):
            raise ValueError(f"not aligned full-vocabulary F32 logits at {key}")
        gold = labels[key[:4]]["label_token"]
        dense = row_nll(values(dense_manifest, a), gold)
        sparse = row_nll(values(sparse_manifest, b), gold)
        delta = sparse["nll_nats"] - dense["nll_nats"]
        dense_total += dense["nll_nats"]
        sparse_total += sparse["nll_nats"]
        deltas.append(delta)
        summary["sparse_rows_worse_nll"] += int(delta > 0)
        summary["sparse_rows_better_nll"] += int(delta < 0)
        summary["dense_reference_top1_hits"] += int(dense["reference_is_top1"])
        summary["sparse_reference_top1_hits"] += int(sparse["reference_is_top1"])
        choice_changed = dense["top1"] != sparse["top1"]
        summary["top1_choice_changed_rows"] += int(choice_changed)
        row = {
            "step": key[0], "phase": key[1], "seq": key[2],
            "position": key[3], "reference_token": gold,
            "dense": dense, "sparse": sparse, "sparse_minus_dense_nll_nats": delta,
        }
        if choice_changed and summary["first_top1_change"] is None:
            summary["first_top1_change"] = row
        if (summary["worst_sparse_nll_regression"] is None or
                delta > summary["worst_sparse_nll_regression"]["sparse_minus_dense_nll_nats"]):
            summary["worst_sparse_nll_regression"] = row
        if (delta != 0 or choice_changed) and len(summary["examples"]) < max_examples:
            summary["examples"].append(row)
    n = len(deltas)
    summary.update({
        "dense_mean_nll_nats": dense_total / n,
        "sparse_mean_nll_nats": sparse_total / n,
        "mean_sparse_minus_dense_nll_nats": (sparse_total - dense_total) / n,
        "median_sparse_minus_dense_nll_nats": _median(deltas),
        "min_sparse_minus_dense_nll_nats": min(deltas),
        "max_sparse_minus_dense_nll_nats": max(deltas),
    })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_manifest", type=Path)
    parser.add_argument("sparse_manifest", type=Path)
    parser.add_argument("reference_labels_jsonl", type=Path)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()
    try:
        output = compare(args.dense_manifest, args.sparse_manifest,
                         args.reference_labels_jsonl,
                         args.max_rows, args.max_examples)
    except (OSError, KeyError, ValueError, TypeError) as err:
        parser.error(str(err))
    print(json.dumps(output, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
