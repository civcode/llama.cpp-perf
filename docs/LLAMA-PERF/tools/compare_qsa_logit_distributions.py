#!/usr/bin/env python3
"""Compare probability distributions from real, token-aligned QSA logits.

Uses the opt-in LLAMA_QSA_LOGITS_DIR manifest from both dense and sparse
processes, with exact event/token-history validation from
compare_qsa_model_state.py. Reports top-1/top-5 changes, shifted-logit
differences and KL/JS divergences; these are metrics, NOT correctness gates.
No extra Python dependencies and no model/GPU access are required.

    python3 docs/LLAMA-PERF/tools/compare_qsa_logit_distributions.py \
        dense-A-logits/manifest.jsonl dense-B-logits/manifest.jsonl
    python3 docs/LLAMA-PERF/tools/compare_qsa_logit_distributions.py \
        dense-A-logits/manifest.jsonl sparse-logits/manifest.jsonl
"""
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

from compare_qsa_model_state import read_manifest, values


def _logsumexp(xs) -> float:
    maximum = max(xs)
    return maximum + math.log(math.fsum(math.exp(x - maximum) for x in xs))


def row_metrics(dense, other) -> dict:
    if len(dense) != len(other) or not dense:
        raise ValueError("logit vocabulary dimensions differ")
    if any(not math.isfinite(x) for x in dense) or any(not math.isfinite(x) for x in other):
        raise ValueError("nonfinite logits")
    n = len(dense)
    top_a = heapq.nlargest(min(5, n), range(n), key=lambda i: (dense[i], -i))
    top_b = heapq.nlargest(min(5, n), range(n), key=lambda i: (other[i], -i))
    a_z = _logsumexp(dense)
    b_z = _logsumexp(other)

    kl_ab = 0.0
    kl_ba = 0.0
    js = 0.0
    centered_max_abs = 0.0
    shift = (math.fsum(dense) - math.fsum(other))/n
    for x, y in zip(dense, other):
        logp = x-a_z
        logq = y-b_z
        p = math.exp(logp)
        q = math.exp(logq)
        if p > 0:
            kl_ab += p*(logp-logq)
        if q > 0:
            kl_ba += q*(logq-logp)
        # Work in log space: (p + q)/2 can underflow even when p or q
        # remains a nonzero subnormal float.
        log_midpoint = max(logp, logq) + math.log1p(math.exp(-abs(logp-logq))) - math.log(2.0)
        if p > 0:
            js += 0.5*p*(logp-log_midpoint)
        if q > 0:
            js += 0.5*q*(logq-log_midpoint)
        centered_max_abs = max(centered_max_abs, abs((x-y)-shift))

    return {
        "vocab": n,
        "dense_top1": top_a[0], "other_top1": top_b[0],
        "top1_changed": top_a[0] != top_b[0],
        "top5_set_changed": set(top_a) != set(top_b),
        "dense_top1_prob": math.exp(dense[top_a[0]]-a_z),
        "other_prob_of_dense_top1": math.exp(other[top_a[0]]-b_z),
        "dense_prob_of_other_top1": math.exp(dense[top_b[0]]-a_z),
        "other_top1_prob": math.exp(other[top_b[0]]-b_z),
        "kl_dense_other_nats": max(0.0, kl_ab),
        "kl_other_dense_nats": max(0.0, kl_ba),
        "js_nats": max(0.0, js),
        "max_abs_logit_difference_after_mean_shift": centered_max_abs,
        "mean_dense_minus_other_logit_shift": shift,
    }


def compare(dense_manifest: Path, other_manifest: Path, max_rows: int = 64,
            max_examples: int = 8) -> dict:
    if not (1 <= max_rows <= 256 and 0 <= max_examples <= 64):
        raise ValueError("max rows must be 1..256, max examples must be 0..64")
    ae, at = read_manifest(dense_manifest)
    be, bt = read_manifest(other_manifest)
    if ae != be:
        raise ValueError("token/candidate, acceptance, or checkpoint histories differ")
    akeys = {key for key in at if key[-1] == "logits"}
    bkeys = {key for key in bt if key[-1] == "logits"}
    if akeys != bkeys or not akeys:
        raise ValueError("logits tensor keys differ or are absent")
    if len(akeys) > max_rows:
        raise ValueError(f"{len(akeys)} logit rows exceed --max-rows={max_rows}")
    result = {
        "dense_manifest": str(dense_manifest), "other_manifest": str(other_manifest),
        "logit_rows": len(akeys), "top1_changed_rows": 0,
        "top5_set_changed_rows": 0, "max_kl_dense_other_nats": 0.0,
        "max_kl_other_dense_nats": 0.0, "max_js_nats": 0.0,
        "max_centered_logit_abs_difference": 0.0,
        "first_top1_flip": None, "worst_kl_row": None,
        "examples": [],
    }
    for key in sorted(akeys):
        a, b = at[key], bt[key]
        if a["shape"] != b["shape"] or a["dtype"] != "f32" or b["dtype"] != "f32":
            raise ValueError(f"mismatched F32 logits shape at {key}")
        d = values(dense_manifest, a)
        o = values(other_manifest, b)
        stats = row_metrics(d, o)
        result["top1_changed_rows"] += int(stats["top1_changed"])
        result["top5_set_changed_rows"] += int(stats["top5_set_changed"])
        result["max_kl_dense_other_nats"] = max(
            result["max_kl_dense_other_nats"], stats["kl_dense_other_nats"])
        result["max_kl_other_dense_nats"] = max(
            result["max_kl_other_dense_nats"], stats["kl_other_dense_nats"])
        result["max_js_nats"] = max(result["max_js_nats"], stats["js_nats"])
        result["max_centered_logit_abs_difference"] = max(
            result["max_centered_logit_abs_difference"],
            stats["max_abs_logit_difference_after_mean_shift"])
        event = {"step": key[0], "phase": key[1], "seq": key[2],
                 "position": key[3], **stats}
        if stats["top1_changed"] and result["first_top1_flip"] is None:
            result["first_top1_flip"] = event
        if (result["worst_kl_row"] is None or
                stats["kl_dense_other_nats"] > result["worst_kl_row"]["kl_dense_other_nats"]):
            result["worst_kl_row"] = event
        if (stats["top1_changed"] or stats["top5_set_changed"]) and len(result["examples"]) < max_examples:
            result["examples"].append(event)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_manifest", type=Path)
    parser.add_argument("other_manifest", type=Path)
    parser.add_argument("--max-rows", type=int, default=64)
    parser.add_argument("--max-examples", type=int, default=8)
    args = parser.parse_args()
    try:
        result = compare(args.dense_manifest, args.other_manifest, args.max_rows, args.max_examples)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
