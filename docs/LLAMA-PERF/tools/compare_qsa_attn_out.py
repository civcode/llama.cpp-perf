#!/usr/bin/env python3
"""Compare QSA F32 attention outputs from two graph-disabled audit runs.

Use with GGML_QSA_MASK_AUDIT, GGML_QSA_ATTN_OUT_DIR and
GGML_QSA_ATTN_OUT_LIMIT in *separate* dense and sparse server processes.
Requires matching token inputs and checks per-op QSA layer label, mask bound,
KV, stream and output shape. This is a diagnostic, not a benchmark or
a proof of recurrent-state/logits equivalence. Python 3 stdlib only.

python3 docs/LLAMA-PERF/tools/compare_qsa_attn_out.py dense.log sparse.log
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import json
import math
from pathlib import Path
import sys


def fields(line: str, prefix: str) -> dict[str, str] | None:
    needle = prefix + " "
    if needle not in line:
        return None
    return dict(token.split("=", 1)
                for token in line.split(needle, 1)[1].split() if "=" in token)


def audit_key(row: dict) -> tuple[str, int, int]:
    return row["backend"], int(row["device"]), int(row["id"])


def parse_log(path: Path) -> dict[tuple[str, int], list[dict]]:
    masks: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    records: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            mask = fields(line, "QSA_MASK_AUDIT")
            if mask and "id" in mask:
                masks[audit_key(mask)].append({
                    "kv": int(mask["kv"]), "bound": int(mask["bound"]),
                    "qheads": int(mask["qheads"]), "kvheads": int(mask["kvheads"]),
                    "stream": int(mask["stream"]), "union": int(mask["union"]),
                    "counts": [int(x) for x in mask["counts"].split(",")],
                })
            output = fields(line, "QSA_ATTN_OUT")
            if output and "path" in output:
                backend = output["backend"]
                device = int(output["device"])
                key = (backend, device)
                shape = tuple(int(x) for x in output["ne"].split(","))
                if len(shape) != 4 or any(x <= 0 for x in shape):
                    raise ValueError(f"invalid output shape: {shape}")
                file = Path(output["path"])
                file = file if file.is_absolute() else path.parent / file
                records[key].append({
                    "backend": backend, "device": device, "id": int(output["id"]),
                    "layer": output["tensor"], "shape": shape, "path": file,
                })
    if not records:
        raise ValueError(f"no QSA_ATTN_OUT records in {path}")
    for key, items in records.items():
        ids: set[int] = set()
        for i, row in enumerate(items):
            row["ordinal"] = i
            if row["id"] in ids:
                raise ValueError(f"duplicate QSA output id on {key}: {row['id']}")
            ids.add(row["id"])
            m = masks.get((row["backend"], row["device"], row["id"]))
            if not m:
                raise ValueError(f"no matching QSA_MASK_AUDIT for {key} id={row['id']}")
            row["masks"] = sorted(m, key=lambda x: x["stream"])
            if [x["stream"] for x in row["masks"]] != list(range(row["shape"][3])):
                raise ValueError(f"stream/shape mismatch on {key} id={row['id']}")
            if row["shape"][1] != row["masks"][0]["qheads"] or row["shape"][2] != 4:
                raise ValueError(f"Q/head/shape mismatch on {key} id={row['id']}")
    return records


def floats(row: dict) -> array:
    raw = row["path"].read_bytes()
    n = math.prod(row["shape"])
    if n > 1 << 20 or len(raw) != n * 4:
        raise ValueError(f"invalid float32 dump size for {row['path']}")
    values = array("f")
    if values.itemsize != 4:
        raise ValueError("requires IEEE-754 32-bit float")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    if any(not math.isfinite(x) for x in values):
        raise ValueError(f"nonfinite attention output in {row['path']}")
    return values


def compare_pair(dense: dict, sparse: dict, abs_tol: float, rel_tol: float) -> dict:
    for field in ("backend", "device", "layer", "shape"):
        if dense[field] != sparse[field]:
            raise ValueError(
                f"unaligned outputs on {dense['backend']}{dense['device']} "
                f"at ordinal={dense['ordinal']}: {field}: "
                f"{dense[field]} != {sparse[field]}"
            )
    # The selected indices and union counts may differ in *later* QSA layers.
    # Compare only identity/shape/bound here; preserve both unions in output.
    if len(dense["masks"]) != len(sparse["masks"]):
        raise ValueError("QSA mask stream count differs")
    for mask_a, mask_b in zip(dense["masks"], sparse["masks"]):
        for field in ("kv", "bound", "qheads", "kvheads", "stream"):
            if mask_a[field] != mask_b[field]:
                raise ValueError(f"unaligned mask {field} at ordinal {dense['ordinal']}")
    a = floats(dense)
    b = floats(sparse)
    dim, nheads, nq, nstreams = dense["shape"]
    different = 0
    first = None
    max_abs = 0.0
    max_rel = 0.0
    sum_err = 0.0
    sum_signal = 0.0
    by_head: dict[int, dict] = {}
    for index, (left, right) in enumerate(zip(a, b)):
        # Output layout: [D, Q-heads, Q positions, streams], dimension 0 fastest.
        head = (index // dim) % nheads
        err = abs(left - right)
        relative = err / max(abs(left), abs(right), 1e-8)
        sum_err += err * err
        sum_signal += left * left
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, relative)
        h = by_head.setdefault(head, {"max_abs": 0.0, "sum_sq_error": 0.0,
                                       "sum_sq_signal": 0.0, "different": 0})
        h["max_abs"] = max(h["max_abs"], err)
        h["sum_sq_error"] += err * err
        h["sum_sq_signal"] += left * left
        if err > abs_tol + rel_tol * max(abs(left), abs(right)):
            different += 1
            h["different"] += 1
            if first is None:
                first = {
                    "linear_index": index, "stream": index // (dim * nheads * nq),
                    "query": (index // (dim * nheads)) % nq, "head": head,
                    "feature": index % dim, "dense": left, "sparse": right,
                    "abs_difference": err,
                }
    head_errors = [{
        "head": head, "max_abs": s["max_abs"],
        "nmse_over_dense_energy": s["sum_sq_error"] / max(s["sum_sq_signal"], 1e-30),
        "changed_elements": s["different"],
    } for head, s in sorted(by_head.items())]
    return {
        "backend": dense["backend"], "device": dense["device"],
        "ordinal": dense["ordinal"], "dense_id": dense["id"], "sparse_id": sparse["id"],
        "layer": dense["layer"], "shape": dense["shape"],
        "dense_mask_union_per_stream": [m["union"] for m in dense["masks"]],
        "sparse_mask_union_per_stream": [m["union"] for m in sparse["masks"]],
        "total_elements": len(a), "changed_elements": different,
        "binary_identical": a.tobytes() == b.tobytes(),
        "max_abs": max_abs, "max_rel": max_rel,
        "nmse_over_dense_energy": sum_err / max(sum_signal, 1e-30),
        "first_difference": first, "worst_heads": sorted(
            head_errors, key=lambda x: x["nmse_over_dense_energy"], reverse=True)[:8],
    }


def compare_logs(dense: Path, sparse: Path, abs_tol: float = 0.0,
                 rel_tol: float = 0.0) -> dict:
    if not (math.isfinite(abs_tol) and math.isfinite(rel_tol) and
            abs_tol >= 0 and rel_tol >= 0):
        raise ValueError("tolerances must be nonnegative finite floats")
    a, b = parse_log(dense), parse_log(sparse)
    if a.keys() != b.keys():
        raise ValueError(f"device sets differ: {list(a)} vs {list(b)}")
    result = {"dense_log": str(dense), "other_log": str(sparse),
              "abs_tol": abs_tol, "rel_tol": rel_tol,
              "ops_compared": 0, "ops_changed": 0,
              "first_changed_per_device": [], "all_max_errors": []}
    for device in sorted(a):
        if len(a[device]) != len(b[device]):
            raise ValueError(f"number of QSA outputs differs on {device}")
        first = None
        for left, right in zip(a[device], b[device]):
            diff = compare_pair(left, right, abs_tol, rel_tol)
            result["ops_compared"] += 1
            if diff["changed_elements"]:
                result["ops_changed"] += 1
                if first is None:
                    first = diff
            result["all_max_errors"].append({
                "backend": diff["backend"], "device": diff["device"],
                "ordinal": diff["ordinal"], "layer": diff["layer"],
                "changed_elements": diff["changed_elements"],
                "max_abs": diff["max_abs"],
                "nmse_over_dense_energy": diff["nmse_over_dense_energy"],
            })
        if first is not None:
            result["first_changed_per_device"].append(first)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_log", type=Path)
    parser.add_argument("other_log", type=Path)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    parser.add_argument("--rel-tol", type=float, default=0.0)
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()
    try:
        result = compare_logs(args.dense_log, args.other_log, args.abs_tol, args.rel_tol)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 2 if args.require_identical and result["ops_changed"] else 0


if __name__ == "__main__":
    sys.exit(main())
