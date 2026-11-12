#!/usr/bin/env python3
"""CPU-only per-query/per-head FP64-reference diagnostic for captured QSA attention.

Checks three *independently captured*, otherwise-matched long-context QSA
attention operations: dense A, dense B and sparse. Requires bitwise-identical
Q/K/V input spans and F16 mask bytes across all three; requires bitwise-
identical dense A/B F32 attention outputs. Then compares dense/sparse with
the existing full-precision mathematical attention reference, exposing the
heads and query positions where either path is closer. No GPU probes, model
inference, arithmetic/source edits, performance claims, or release tolerance.

Usage:
 python3 docs/LLAMA-PERF/tools/analyze_qsa_headwise_precision.py \
  /external/dense-a/server.log /external/dense-b/server.log \
  /external/sparse/server.log --out /external/headwise-precision.json

The input captures must be the same QSA attention op (typically the first
HIP0 layer-11 Q4 group) with exactly the same logical token context. Raw
Q/K/V, mask and F32 output data remain outside Git.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from replay_qsa_captured_attention import (
    cpu_reference, read_capture, read_mask, read_output, read_tensor, resolved,
)

MAX_CAPTURE_BYTES = 512 * 1024 * 1024


def summarize(reference: np.ndarray, dense: np.ndarray, sparse: np.ndarray) -> dict:
    """Paired squared-error comparison; no numerical pass/fail threshold."""
    if (reference.ndim != 4 or reference.shape != dense.shape or
            dense.shape != sparse.shape or reference.shape[0] != 1 or
            reference.shape[1] != 4 or reference.shape[2] != 24 or
            reference.shape[3] != 256):
        raise ValueError("expected identical QSA output shape [1,4,24,256]")
    for name, value in (("reference", reference), ("dense", dense), ("sparse", sparse)):
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite floating point")

    ref = np.asarray(reference, dtype=np.float64)
    a = np.asarray(dense, dtype=np.float64)
    b = np.asarray(sparse, dtype=np.float64)
    err_a = a - ref
    err_b = b - ref
    # No underflow to float32 from small reference values.
    sq_a = err_a * err_a
    sq_b = err_b * err_b
    energy = ref * ref
    total_a = float(sq_a.sum(dtype=np.float64))
    total_b = float(sq_b.sum(dtype=np.float64))
    energy_all = float(energy.sum(dtype=np.float64))
    rows = []
    heads = []
    better = worse = tied = 0
    for query in range(4):
        for head in range(24):
            idx = (0, query, head)
            da = float(sq_a[idx].sum(dtype=np.float64))
            sb = float(sq_b[idx].sum(dtype=np.float64))
            en = float(energy[idx].sum(dtype=np.float64))
            if sb < da:
                outcome = "sparse"
                better += 1
            elif da < sb:
                outcome = "dense"
                worse += 1
            else:
                outcome = "tie"
                tied += 1
            rows.append({
                "query": query, "head": head,
                "dense_squared_error": da, "sparse_squared_error": sb,
                "reference_energy": en,
                "dense_nmse": da/max(en, 1e-30),
                "sparse_nmse": sb/max(en, 1e-30),
                "winner": outcome,
                "dense_max_abs": float(np.max(np.abs(err_a[idx]))),
                "sparse_max_abs": float(np.max(np.abs(err_b[idx]))),
            })
    for head in range(24):
        idx = (0, slice(None), head)
        da = float(sq_a[idx].sum(dtype=np.float64))
        sb = float(sq_b[idx].sum(dtype=np.float64))
        en = float(energy[idx].sum(dtype=np.float64))
        heads.append({
            "head": head, "dense_nmse": da/max(en, 1e-30),
            "sparse_nmse": sb/max(en, 1e-30),
            "dense_squared_error": da, "sparse_squared_error": sb,
            "reference_energy": en,
            "winner": "sparse" if sb < da else ("dense" if da < sb else "tie"),
        })
    return {
        "output_shape": [1,4,24,256],
        "total_dense_nmse": total_a/max(energy_all,1e-30),
        "total_sparse_nmse": total_b/max(energy_all,1e-30),
        "total_dense_squared_error": total_a,
        "total_sparse_squared_error": total_b,
        "total_reference_energy": energy_all,
        "heads_better_sparse": sum(h["winner"] == "sparse" for h in heads),
        "heads_better_dense": sum(h["winner"] == "dense" for h in heads),
        "heads_tied": sum(h["winner"] == "tie" for h in heads),
        "query_head_pairs_better_sparse": better,
        "query_head_pairs_better_dense": worse,
        "query_head_pairs_tied": tied,
        "headwise": heads,
        "query_headwise": rows,
        "interpretation": (
            "Error relative to the identical-input FP64 mathematical "
            "reference. Squared-error winners do not imply matching dense "
            "logits, identical selection or a release-quality threshold. "
            "The reference is not a hardware rounding specification."
        ),
        "numerical_equivalence": "NOT_ESTABLISHED",
        "production_promotion": "NO_GO",
    }


def _identical_capture_inputs(captures: list[dict]) -> tuple[dict, np.ndarray, dict[str, np.ndarray]]:
    baseline = captures[0]
    for other in captures[1:]:
        for key in ("scale",):
            if baseline[key] != other[key]:
                raise ValueError(f"input attention {key} differs")
        if baseline["meta"]["tensor"] != other["meta"]["tensor"]:
            raise ValueError("attention layer identity differs")
        for key in ("kv", "bound", "qheads", "kvheads"):
            if baseline["audit"][key] != other["audit"][key]:
                raise ValueError(f"attention metadata {key} differs")

    qkv: dict[str, np.ndarray] = {}
    total = 0
    for name in "qkv":
        primary = baseline["inputs"][name]
        for c in captures[1:]:
            other = c["inputs"][name]
            for field in ("type", "ne", "nb"):
                if primary[field] != other[field]:
                    raise ValueError(f"QKV metadata mismatch for {name}:{field}")
        paths = [resolved(c["inputs"][name]["path"], Path(c["log"])) for c in captures]
        blobs = [path.read_bytes() for path in paths]
        total += sum(len(x) for x in blobs)
        if total > MAX_CAPTURE_BYTES:
            raise ValueError("total captured raw tensor bytes exceed 512 MiB")
        if blobs[0] != blobs[1] or blobs[0] != blobs[2]:
            raise ValueError(f"QKV {name} raw span bytes differ across captures")
        qkv[name] = read_tensor(primary, Path(baseline["log"]))

    masks = [read_mask(c) for c in captures]
    if not (np.array_equal(masks[0].view(np.uint16), masks[1].view(np.uint16)) and
            np.array_equal(masks[0].view(np.uint16), masks[2].view(np.uint16))):
        raise ValueError("raw QSA F16 mask bytes differ across captures")
    for c in captures[1:]:
        if c["audit"]["union"] != baseline["audit"]["union"]:
            raise ValueError("audited sparse index mask union count differs")
    return baseline, masks[0], qkv


def analyze(dense_a_log: Path, dense_b_log: Path, sparse_log: Path) -> dict:
    logs = [dense_a_log.resolve(strict=True), dense_b_log.resolve(strict=True),
            sparse_log.resolve(strict=True)]
    if len(set(logs)) != 3:
        raise ValueError("three independent capture log paths are required")
    captures = [read_capture(path) for path in logs]
    first, mask, qkv = _identical_capture_inputs(captures)
    da, db, sp = [read_output(c) for c in captures]
    if not np.array_equal(da.view(np.uint32), db.view(np.uint32)):
        raise ValueError("dense A/B F32 outputs are not bitwise identical")
    if first["meta"]["tensor"] != "qsa_flash_attn_il11":
        raise ValueError("expected captured QSA layer 11 attention")
    reference = cpu_reference(qkv["q"], qkv["k"], qkv["v"], mask, first["scale"])
    result = summarize(reference, da, sp)
    result.update({
        "layer": first["meta"]["tensor"],
        "kv": int(first["audit"]["kv"]),
        "mask_union": int(first["audit"]["union"]),
        "raw_qkv_mask_identical_across_three": True,
        "dense_a_b_attention_bitwise_identical": True,
        "f64_mathematical_reference": True,
        "current_host_health": "NOT_VERIFIED_BY_OFFLINE_REPLAY",
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_a_log", type=Path)
    parser.add_argument("dense_b_log", type=Path)
    parser.add_argument("sparse_log", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = analyze(args.dense_a_log, args.dense_b_log, args.sparse_log)
        with args.out.open("x", encoding="utf-8") as out:
            json.dump(report, out, indent=2, allow_nan=False)
            out.write("\n")
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as err:
        parser.error(str(err))
    print(json.dumps({
        "layer": report["layer"],
        "heads_better_sparse": report["heads_better_sparse"],
        "heads_better_dense": report["heads_better_dense"],
        "query_head_pairs_better_sparse": report["query_head_pairs_better_sparse"],
        "total_dense_nmse": report["total_dense_nmse"],
        "total_sparse_nmse": report["total_sparse_nmse"],
        "production_promotion": "NO_GO",
        "external_report": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
