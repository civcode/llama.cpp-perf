#!/usr/bin/env python3
"""Analyze bounded HIP0 Qwen3.8 QSA layer-15 score/top-k traces.

Capture after GPU operations with GGML_QSA_MASK_AUDIT=1000,
GGML_QSA_TOPK_TRACE_DIR=<unique-external-dir>, GGML_QSA_TOPK_TRACE_LIMIT=64.
Run three separate, identical-token processes: dense A, dense B, sparse.
Capture is synchronous and disables GPU graphs: NEVER use for performance.

    python3 docs/LLAMA-PERF/tools/compare_qsa_topk_trace.py \
      dense-A.log dense-B.log > dense-control.json
    python3 docs/LLAMA-PERF/tools/compare_qsa_topk_trace.py \
      dense-A.log sparse.log > dense-sparse.json

The tool uses stdlib only and does not infer a global logical step from
a per-device/group ordinal. It compares selected pool *sets*, score margins,
and the pool->KV mapping for the fused TOP_K_MASK path. If the nonfused path
is used, TOP_K-only captures still quantify scores and pool IDs but cannot
map them to KV cells. Scores/masks here are evidence, not model acceptance
or GDN-state equivalence.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import sys


def _fields(line: str, marker: str) -> dict[str, str] | None:
    needle = marker + " "
    if needle not in line:
        return None
    return dict(part.split("=", 1) for part in line.split(needle, 1)[1].split()
                if "=" in part)


def _key(fields: dict[str, str]) -> tuple[str, int, str, int]:
    return (fields["backend"], int(fields["device"]), fields["kind"], int(fields["group"]))


def _shape(record: dict) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ne = tuple(int(x) for x in record["ne"].split(","))
    nb = tuple(int(x) for x in record["nb"].split(","))
    if len(ne) != 4 or len(nb) != 4 or any(n < 1 for n in ne):
        raise ValueError(f"invalid tensor shape: {ne} / {nb}")
    return ne, nb


def _data(record: dict, log: Path) -> array:
    ne, nb = _shape(record)
    kind = record["type"]
    if kind not in ("f32", "i32"):
        raise ValueError(f"unsupported tensor type {kind}")
    if nb[0] != 4 or any(nb[i] != nb[i-1]*ne[i-1] for i in range(1, 4)):
        raise ValueError(f"noncontiguous {record['name']} trace is not supported")
    path = Path(record["path"])
    if not path.is_absolute():
        path = log.parent / path
    raw = path.read_bytes()
    required = math.prod(ne)*4
    if required > 32*1024*1024 or len(raw) != required or len(raw) != int(record["bytes"]):
        raise ValueError(f"invalid raw byte count for {path}")
    values = array("f" if kind == "f32" else "i")
    if values.itemsize != 4:
        raise ValueError("host needs 32-bit float and signed integer")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if kind == "f32" and any(math.isnan(v) or v == math.inf for v in values):
        raise ValueError(f"invalid NaN/+Inf indexer score in {path}")
    return values


def read_log(log: Path) -> dict:
    groups = {}
    with log.open(encoding="utf-8", errors="replace") as lines:
        for line in lines:
            header = _fields(line, "QSA_TOPK_GROUP")
            if header:
                key = _key(header)
                if key in groups:
                    raise ValueError(f"duplicate QSA_TOPK_GROUP {key} in {log}")
                groups[key] = {"header": header, "parts": {}}
                continue
            part = _fields(line, "QSA_TOPK_PART")
            if part:
                key = _key(part)
                if key not in groups:
                    raise ValueError(f"QSA_TOPK_PART without group {key} in {log}")
                name = part["name"]
                if name in groups[key]["parts"]:
                    raise ValueError(f"duplicate {name} part in {key}")
                groups[key]["parts"][name] = part
    if not groups:
        raise ValueError(f"no QSA_TOPK_GROUP in {log}")
    # Use the pool-mapping variant if available. It comes after TOP_K.
    mask_keys = {key for key in groups if key[2] == "mask"}
    if mask_keys:
        groups = {key: value for key, value in groups.items() if key in mask_keys}
    else:
        groups = {key: value for key, value in groups.items() if key[2] == "topk"}

    results = {}
    for key, group in sorted(groups.items()):
        header, parts = group["header"], group["parts"]
        if (key[0] != "HIP" or key[1] != 0 or int(header["layer"]) != 15
                or key[2] not in ("topk", "mask")):
            raise ValueError(f"unexpected QSA top-k trace {key}")
        if not {"score", "topk"}.issubset(parts):
            raise ValueError(f"incomplete score/topk trace {key}")
        n_pool, k, nq = (int(header[n]) for n in ("n_pool", "n_top", "n_q"))
        if not (0 < k <= n_pool <= 262144 and 0 < nq <= 4096):
            raise ValueError(f"invalid topk group shape {key}")
        if key[2] == "mask" and "blocks" not in parts:
            raise ValueError(f"mask trace lacks pool-to-KV map {key}")
        scores = _data(parts["score"], log)
        selected = _data(parts["topk"], log)
        if _shape(parts["score"])[0] != (n_pool, nq, 1, 1):
            raise ValueError(f"score shape mismatch in {key}")
        if _shape(parts["topk"])[0] != (k, nq, 1, 1):
            raise ValueError(f"top-k shape mismatch in {key}")
        blocks = _data(parts["blocks"], log) if "blocks" in parts else None
        extra = _data(parts["extra"], log) if "extra" in parts else None
        bshape = _shape(parts["blocks"])[0] if blocks is not None else None
        eshape = _shape(parts["extra"])[0] if extra is not None else None
        if bshape is not None and (bshape[1] != n_pool or bshape[2:] != (1, 1)):
            raise ValueError(f"pool map has wrong shape in {key}")
        if eshape is not None and eshape[1:] != (nq, 1, 1):
            raise ValueError(f"tail cells have wrong shape in {key}")
        results[key] = {
            "key": key, "kind": key[2], "n_pool": n_pool, "n_top": k,
            "nq": nq, "kv": int(header["kv"]), "scores": scores,
            "selected": selected, "blocks": blocks, "bshape": bshape,
            "extra": extra, "eshape": eshape,
        }
        for q in range(nq):
            row = selected[q*k:(q+1)*k]
            if len(set(row)) != len(row) or any(p < 0 or p >= n_pool for p in row):
                raise ValueError(f"invalid top-k indices in {key} query {q}")
    return results


def _row(record: dict, query: int) -> dict:
    n_pool, k, nq = record["n_pool"], record["n_top"], record["nq"]
    if query < 0 or query >= nq:
        raise ValueError("invalid query index")
    scores = record["scores"][query*n_pool:(query+1)*n_pool]
    picks = record["selected"][query*k:(query+1)*k]
    # TOP_K_MASK ignores -Inf score picks and invalid cache cells.
    valid = {int(p) for p in picks if scores[p] != -math.inf}
    all_finite = {i for i, x in enumerate(scores) if math.isfinite(x)}
    excluded = all_finite - valid
    lowest = min((scores[i] for i in valid), default=None)
    highest_out = max((scores[i] for i in excluded), default=None)
    margin = float(lowest - highest_out) if lowest is not None and highest_out is not None else None
    mapped = None
    if record["blocks"] is not None:
        bsize = record["bshape"][0]
        kv = record["kv"]
        mapped = set()
        for pool in valid:
            for col in range(bsize):
                cell = int(record["blocks"][pool*bsize+col])
                if 0 <= cell < kv:
                    mapped.add(cell)
        if record["extra"] is not None:
            n_extra = record["eshape"][0]
            for cell in record["extra"][query*n_extra:(query+1)*n_extra]:
                if 0 <= cell < kv:
                    mapped.add(int(cell))
    return {"valid": valid, "scores": scores, "cutoff_score": lowest,
            "best_excluded_score": highest_out, "margin": margin,
            "kv_cells": mapped}


def compare_rows(a: dict, b: dict, query: int) -> dict:
    before = _row(a, query)
    after = _row(b, query)
    removed = sorted(before["valid"] - after["valid"])
    added = sorted(after["valid"] - before["valid"])
    scores_a, scores_b = before["scores"], after["scores"]
    diffs = [(i, abs(float(x)-float(y))) for i, (x, y) in enumerate(zip(scores_a, scores_b))
             if math.isfinite(x) and math.isfinite(y)]
    finite_mismatch = sum(math.isfinite(x) != math.isfinite(y) for x, y in zip(scores_a, scores_b))
    max_delta = max((v for _, v in diffs), default=0.0)
    first_changed_score = next((i for i, delta in diffs if delta != 0), None)
    mapped_removed = (sorted(before["kv_cells"]-after["kv_cells"])
                      if before["kv_cells"] is not None else None)
    mapped_added = (sorted(after["kv_cells"]-before["kv_cells"])
                    if after["kv_cells"] is not None else None)
    changed = bool(removed or added)
    return {
        "query": query,
        "changed_selection": changed,
        "pools_removed_count": len(removed), "pools_added_count": len(added),
        "pools_removed_first": removed[:16], "pools_added_first": added[:16],
        "dense_lowest_selected_score": before["cutoff_score"],
        "dense_best_excluded_score": before["best_excluded_score"],
        "dense_cutoff_margin": before["margin"],
        "other_lowest_selected_score": after["cutoff_score"],
        "other_best_excluded_score": after["best_excluded_score"],
        "other_cutoff_margin": after["margin"],
        "max_absolute_score_delta": max_delta,
        "finite_status_disagreements": finite_mismatch,
        "first_changed_score_pool": first_changed_score,
        "dense_score_for_changed_pools": {str(p): float(scores_a[p]) for p in (removed+added)[:16]},
        "other_score_for_changed_pools": {str(p): float(scores_b[p]) for p in (removed+added)[:16]},
        "kv_cells_removed_count": len(mapped_removed) if mapped_removed is not None else None,
        "kv_cells_added_count": len(mapped_added) if mapped_added is not None else None,
        "kv_cells_removed_first": mapped_removed[:24] if mapped_removed is not None else None,
        "kv_cells_added_first": mapped_added[:24] if mapped_added is not None else None,
    }


def compare(dense: Path, other: Path, max_events: int = 8) -> dict:
    left, right = read_log(dense), read_log(other)
    if left.keys() != right.keys():
        raise ValueError(f"top-k group keys differ: {list(left)} vs {list(right)}")
    summary = {
        "dense_log": str(dense), "other_log": str(other),
        "kind": next(iter(left))[2], "groups_compared": 0, "queries_compared": 0,
        "queries_with_changed_pool_selections": 0, "groups_with_changed_selections": 0,
        "events": [], "first_changed_selection": None,
    }
    for key in sorted(left):
        a, b = left[key], right[key]
        for field in ("n_pool", "n_top", "nq", "kv", "kind"):
            if a[field] != b[field]:
                raise ValueError(f"unaligned {field} for group {key}")
        summary["groups_compared"] += 1
        changed_group = False
        for query in range(a["nq"]):
            row = compare_rows(a, b, query)
            summary["queries_compared"] += 1
            if row["changed_selection"]:
                summary["queries_with_changed_pool_selections"] += 1
                changed_group = True
                event = {"backend": key[0], "device": key[1],
                         "kind": key[2], "group": key[3], **row}
                if summary["first_changed_selection"] is None:
                    summary["first_changed_selection"] = event
                if len(summary["events"]) < max_events:
                    summary["events"].append(event)
        if changed_group:
            summary["groups_with_changed_selections"] += 1
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_log", type=Path)
    parser.add_argument("other_log", type=Path)
    parser.add_argument("--max-events", type=int, default=8)
    parser.add_argument("--require-identical-selections", action="store_true")
    args = parser.parse_args()
    if args.max_events < 0 or args.max_events > 128:
        parser.error("--max-events must be 0..128")
    try:
        result = compare(args.dense_log, args.other_log, args.max_events)
    except (ValueError, OSError, KeyError, OverflowError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if args.require_identical_selections and result["queries_with_changed_pool_selections"] else 0


if __name__ == "__main__":
    sys.exit(main())
