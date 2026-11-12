#!/usr/bin/env python3
"""Compare opt-in logical-token logits/GDN snapshots from dense and sparse.

This **does not capture** server data; it defines a strict machine-readable
contract for a bounded local observer. Each input is a JSONL manifest with:

  {"kind":"event","step":0,"phase":"after_prompt","seq":0,"position":33024,
   "target_ids":[12],"draft_ids":[],"accepted":[],"rollback":false}
  {"kind":"tensor","step":0,"phase":"after_prompt","seq":0,"position":33024,
   "layer":11,"name":"gdn_s","dtype":"f16","shape":[128],
   "path":"external-capture/gdn-s.bin","physical_cell":17,"rs_plane":0}
  {"kind":"tensor",...,"layer":-1,"name":"logits","dtype":"f32",...}

Tensors are raw little-endian arrays in C-flat order with exact shapes.
Supported dtype: f32, f16, bf16. Files and manifests stay OUTSIDE Git.
Physical cell/rollback snapshot plane are recorded for attribution, but they
are not logical matching keys: the observer must resolve the *logical* state
row before writing, including pending rollback snapshot-plane semantics.
Every tensor must match an event's step/phase/seq/position key.

    python3 docs/LLAMA-PERF/tools/compare_qsa_model_state.py denseA.jsonl denseB.jsonl
    python3 docs/LLAMA-PERF/tools/compare_qsa_model_state.py denseA.jsonl sparse.jsonl \
        --abs-tol 1e-4 --rel-tol 1e-3

Tolerances are reporting thresholds ONLY, not permission to claim model
correctness. The observer must separately prove target/draft token alignment.
"""
from __future__ import annotations

from array import array
import argparse
import json
import math
from pathlib import Path
import struct
import sys


def key(r: dict) -> tuple:
    return (r["step"], r["phase"], r["seq"], r["position"])


def tensor_key(r: dict) -> tuple:
    return (*key(r), r["layer"], r["name"])


def read_manifest(path: Path) -> tuple[dict, dict]:
    events = {}
    tensors = {}
    count = 0
    with path.open(encoding="utf-8") as source:
        for lineno, line in enumerate(source, 1):
            if not line.strip():
                continue
            count += 1
            if count > 1024:
                raise ValueError(f"manifest exceeds 1024 records: {path}")
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("not a JSON object")
                identity = key(record)
                if (not isinstance(record["step"], int) or
                        not isinstance(record["seq"], int) or
                        not isinstance(record["position"], int) or
                        not isinstance(record["phase"], str) or not record["phase"]):
                    raise ValueError("invalid event identity")
                kind = record["kind"]
                if kind == "event":
                    if identity in events:
                        raise ValueError(f"duplicate event {identity}")
                    for field in ("target_ids", "draft_ids", "accepted"):
                        if not isinstance(record[field], list):
                            raise ValueError(f"missing or invalid {field}")
                    if not isinstance(record["rollback"], bool):
                        raise ValueError("rollback must be boolean")
                    # Exclude only process-local observability data, never token
                    # identities, acceptance decisions or cache positions.
                    filtered = {k: v for k, v in record.items()
                                if k not in ("monotonic_ns", "log_path")}
                    events[identity] = filtered
                elif kind == "tensor":
                    tkey = tensor_key(record)
                    if tkey in tensors:
                        raise ValueError(f"duplicate tensor {tkey}")
                    if (not isinstance(record["layer"], int) or
                            not isinstance(record["name"], str) or not record["name"]):
                        raise ValueError("invalid tensor identity")
                    if record["name"] not in ("logits", "gdn_r", "gdn_s", "gdn_p"):
                        raise ValueError(f"unsupported tensor role {record['name']}")
                    shape = record["shape"]
                    if (not isinstance(shape, list) or not 1 <= len(shape) <= 4
                            or any(not isinstance(n, int) or n <= 0 for n in shape)):
                        raise ValueError("invalid tensor shape")
                    if record["dtype"] not in ("f32", "f16", "bf16"):
                        raise ValueError("unsupported snapshot dtype")
                    if not isinstance(record["path"], str) or not record["path"]:
                        raise ValueError("snapshot lacks path")
                    tensors[tkey] = record
                else:
                    raise ValueError(f"unsupported manifest kind {kind}")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not events or not tensors:
        raise ValueError(f"manifest lacks events or tensors: {path}")
    if {k[:4] for k in tensors} - events.keys():
        raise ValueError("tensor lacks matching logical token/checkpoint event")
    return events, tensors


def values(manifest: Path, record: dict) -> list[float] | array:
    filename = Path(record["path"])
    if not filename.is_absolute():
        filename = manifest.parent / filename
    raw = filename.read_bytes()
    count = math.prod(record["shape"])
    width = 4 if record["dtype"] == "f32" else 2
    if len(raw) != count*width or len(raw) > 64*1024*1024:
        raise ValueError(f"invalid bounded snapshot byte count: {filename}")
    if record["dtype"] == "f32":
        vals = array("f")
        if vals.itemsize != 4:
            raise ValueError("host lacks 32-bit float")
        vals.frombytes(raw)
        if sys.byteorder != "little":
            vals.byteswap()
    elif record["dtype"] == "f16":
        vals = [x[0] for x in struct.iter_unpack("<e", raw)]
    else:
        bits = array("H")
        bits.frombytes(raw)
        if sys.byteorder != "little":
            bits.byteswap()
        vals = [struct.unpack("<f", struct.pack("<I", int(v) << 16))[0] for v in bits]
    if any(not math.isfinite(v) for v in vals):
        raise ValueError(f"nonfinite logical snapshot values: {filename}")
    return vals


def diff_stats(a, b, abs_tol: float, rel_tol: float) -> dict:
    if len(a) != len(b):
        raise ValueError("snapshot element count differs")
    max_abs = max_rel = sum_sq_error = sum_sq_ref = 0.0
    changed = 0
    first = None
    for index, (x, y) in enumerate(zip(a, b)):
        err = abs(float(x)-float(y))
        relative = err/max(abs(float(x)), abs(float(y)), 1e-8)
        max_abs = max(max_abs, err)
        max_rel = max(max_rel, relative)
        sum_sq_error += err*err
        sum_sq_ref += float(x)*float(x)
        if err > abs_tol + rel_tol*max(abs(float(x)), abs(float(y))):
            changed += 1
            if first is None:
                first = {"index": index, "dense": float(x), "other": float(y),
                         "absolute_error": err}
    return {"elements": len(a), "changed_elements": changed,
            "max_abs": max_abs, "max_rel": max_rel,
            "nmse_over_dense_energy": sum_sq_error/max(sum_sq_ref, 1e-30),
            "first_changed": first}


def compare(dense: Path, other: Path, abs_tol: float = 0.0,
            rel_tol: float = 0.0, max_entries: int = 12) -> dict:
    if not (math.isfinite(abs_tol) and math.isfinite(rel_tol)
            and abs_tol >= 0 and rel_tol >= 0):
        raise ValueError("invalid comparison tolerances")
    ev_a, ts_a = read_manifest(dense)
    ev_b, ts_b = read_manifest(other)
    if ev_a != ev_b:
        missing = sorted(set(ev_a) ^ set(ev_b))
        mismatched = [k for k in ev_a.keys() & ev_b.keys() if ev_a[k] != ev_b[k]]
        raise ValueError(f"target/draft token, acceptance or checkpoint alignment mismatch: "
                         f"missing={missing[:2]}, differing={sorted(mismatched)[:2]}")
    if ts_a.keys() != ts_b.keys():
        raise ValueError(f"snapshot logical keys differ: "
                         f"dense_only={sorted(ts_a.keys()-ts_b.keys())[:2]} "
                         f"other_only={sorted(ts_b.keys()-ts_a.keys())[:2]}")
    result = {"dense_manifest": str(dense), "other_manifest": str(other),
              "logical_events": len(ev_a), "tensor_snapshots": len(ts_a),
              "abs_tol": abs_tol, "rel_tol": rel_tol,
              "changed_snapshots": 0, "first_changed": None,
              "changed_examples": [], "max_nmse": 0.0, "max_abs": 0.0}
    for tkey in sorted(ts_a):
        ta, tb = ts_a[tkey], ts_b[tkey]
        for field in ("dtype", "shape"):
            if ta[field] != tb[field]:
                raise ValueError(f"mismatched snapshot {field} at {tkey}")
        stats = diff_stats(values(dense, ta), values(other, tb), abs_tol, rel_tol)
        result["max_nmse"] = max(result["max_nmse"], stats["nmse_over_dense_energy"])
        result["max_abs"] = max(result["max_abs"], stats["max_abs"])
        if stats["changed_elements"]:
            result["changed_snapshots"] += 1
            event = {"step": tkey[0], "phase": tkey[1], "seq": tkey[2],
                     "position": tkey[3], "layer": tkey[4], "name": tkey[5],
                     "dtype": ta["dtype"], "shape": ta["shape"],
                     "dense_physical_cell": ta.get("physical_cell"),
                     "other_physical_cell": tb.get("physical_cell"),
                     "dense_rs_plane": ta.get("rs_plane"),
                     "other_rs_plane": tb.get("rs_plane"), **stats}
            if result["first_changed"] is None:
                result["first_changed"] = event
            if len(result["changed_examples"]) < max_entries:
                result["changed_examples"].append(event)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_manifest", type=Path)
    parser.add_argument("other_manifest", type=Path)
    parser.add_argument("--abs-tol", type=float, default=0.0)
    parser.add_argument("--rel-tol", type=float, default=0.0)
    parser.add_argument("--max-entries", type=int, default=12)
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.max_entries <= 128:
        parser.error("--max-entries must be 0..128")
    try:
        out = compare(args.dense_manifest, args.other_manifest, args.abs_tol,
                      args.rel_tol, args.max_entries)
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(out, indent=2, allow_nan=False))
    return 2 if args.require_identical and out["changed_snapshots"] else 0


if __name__ == "__main__":
    sys.exit(main())
