#!/usr/bin/env python3
"""Replay a captured real QSA attention op against an FP64 CPU reference.

Requires NumPy; model capture itself remains limited and opt-in. This tool reads
the first HIP0 layer-11 Q/K/V capture produced by GGML_QSA_INPUT_DIR, its raw
F16 mask and post-attention F32 output. It validates dense/sparse input identity
and computes scaled masked softmax(QK^T)V in float64 using the *actual* scale
logged as QSA_INPUT_META. It does not claim bit-exact GPU arithmetic, nor prove
logit/GDN-state equivalence.

    python3 docs/LLAMA-PERF/tools/replay_qsa_captured_attention.py \
      dense-server.log sparse-server.log > layer11-cpu-reference.json

Both runs must set GGML_QSA_MASK_AUDIT, GGML_QSA_MASK_AUDIT_DIR,
GGML_QSA_MASK_DUMP_LIMIT, GGML_QSA_INPUT_DIR, GGML_QSA_ATTN_OUT_DIR and
GGML_QSA_ATTN_OUT_LIMIT in separate external directories. Never benchmark
with captures enabled. No raw tensors are printed or committed.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

try:
    import numpy as np
except ImportError as exc:
    raise SystemExit("NumPy is required for the bounded FP64 reference: " + str(exc))


def fields(line: str, marker: str) -> dict[str, str] | None:
    needle = marker + " "
    if needle not in line:
        return None
    return dict(t.split("=", 1) for t in line.split(needle, 1)[1].split() if "=" in t)


def resolved(path_text: str, log_path: Path) -> Path:
    p = Path(path_text)
    return p if p.is_absolute() else log_path.parent / p


def read_capture(log: Path) -> dict:
    records: dict[str, list[dict]] = {
        "QSA_INPUT_META": [], "QSA_INPUT": [], "QSA_MASK_DUMP": [],
        "QSA_MASK_AUDIT": [], "QSA_ATTN_OUT": [],
    }
    with log.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            for marker in records:
                f = fields(line, marker)
                if f:
                    records[marker].append(f)
                    break
    metas = records["QSA_INPUT_META"]
    if len(metas) != 1:
        raise ValueError(f"{log}: require exactly one QSA_INPUT_META, got {len(metas)}")
    meta = metas[0]
    audit_id = int(meta["id"])
    if meta["backend"] != "HIP" or int(meta["device"]) != 0:
        raise ValueError("reference capture is restricted to HIP0")
    if meta["tensor"] != "qsa_flash_attn_il11":
        raise ValueError(f"wrong model layer: {meta['tensor']}")
    scale = float(meta["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid scale: {scale}")
    if any(float(meta[k]) != 0 for k in ("max_bias", "softcap")) or int(meta["sinks"]):
        raise ValueError("CPU reference does not implement ALiBi, logit softcap or sinks")
    inputs = [r for r in records["QSA_INPUT"] if int(r["id"]) == audit_id]
    if len(inputs) != 3 or {r["name"] for r in inputs} != set("qkv"):
        raise ValueError(f"{log}: missing Q, K or V input capture for id={audit_id}")
    inputs = {r["name"]: r for r in inputs}
    masks = [r for r in records["QSA_MASK_DUMP"]
             if r["backend"] == "HIP" and int(r["device"]) == 0
             and int(r["id"]) == audit_id and int(r["stream"]) == 0]
    audits = [r for r in records["QSA_MASK_AUDIT"]
              if r["backend"] == "HIP" and int(r["device"]) == 0
              and int(r["id"]) == audit_id and int(r["stream"]) == 0]
    outputs = [r for r in records["QSA_ATTN_OUT"]
               if r["backend"] == "HIP" and int(r["device"]) == 0
               and int(r["id"]) == audit_id]
    if len(masks) != 1 or len(audits) != 1 or len(outputs) != 1:
        raise ValueError(f"{log}: mask/audit/output not uniquely paired with input id={audit_id}")
    if outputs[0]["tensor"] != meta["tensor"] or audits[0]["tensor"] != meta["tensor"]:
        raise ValueError("model layer mismatch across input/mask/output records")
    return {"log": str(log), "id": audit_id, "scale": scale,
            "meta": meta, "inputs": inputs, "mask": masks[0],
            "audit": audits[0], "output": outputs[0]}


def spans(record: dict) -> tuple[tuple[int, ...], tuple[int, ...], np.dtype]:
    ne = tuple(int(x) for x in record["ne"].split(","))
    nb = tuple(int(x) for x in record["nb"].split(","))
    if len(ne) != 4 or len(nb) != 4 or any(x < 1 for x in ne) or any(x < 0 for x in nb):
        raise ValueError("bad GGML tensor dimensions/strides")
    if record["type"] == "f32":
        dtype = np.dtype("<f4")
    elif record["type"] == "f16":
        dtype = np.dtype("<f2")
    else:
        raise ValueError(f"unsupported tensor type: {record['type']}")
    if nb[0] != dtype.itemsize:
        raise ValueError("unexpected element stride")
    return ne, nb, dtype


def read_tensor(record: dict, log_path: Path) -> np.ndarray:
    ne, nb, dtype = spans(record)
    raw = resolved(record["path"], log_path).read_bytes()
    if len(raw) > 128 * 1024 * 1024 or len(raw) != int(record["bytes"]):
        raise ValueError("input capture span is missing or exceeds size bound")
    required = sum((ne[i] - 1)*nb[i] for i in range(4)) + dtype.itemsize
    if required > len(raw):
        raise ValueError(f"tensor view exceeds recorded span ({required} > {len(raw)})")
    # Reversing GGML's ne/nb creates the C-style [stream,head,token,feature] view.
    return np.ndarray(shape=ne[::-1], strides=nb[::-1], dtype=dtype, buffer=raw)


def read_mask(c: dict) -> np.ndarray:
    log = Path(c["log"])
    n_kv = int(c["audit"]["kv"])
    raw = resolved(c["mask"]["path"], log).read_bytes()
    if len(raw) != 4*n_kv*2 or n_kv > 262144:
        raise ValueError("bad raw mask shape or size")
    mask = np.frombuffer(raw, dtype="<f2").reshape(4, n_kv)
    if np.isnan(mask).any() or np.isposinf(mask).any():
        raise ValueError("invalid NaN or +inf in QSA mask")
    counts = np.isfinite(mask).sum(axis=1).tolist()
    union = int(np.isfinite(mask).any(axis=0).sum())
    audit = c["audit"]
    if counts != [int(v) for v in audit["counts"].split(",")]:
        raise ValueError("raw per-query counts do not match audit")
    if union != int(audit["union"]) or max(counts) > int(audit["bound"]):
        raise ValueError("raw union exceeds or differs from audit")
    return mask


def read_output(c: dict) -> np.ndarray:
    ne = tuple(int(x) for x in c["output"]["ne"].split(","))
    if len(ne) != 4 or any(x <= 0 for x in ne) or math.prod(ne) > 1 << 20:
        raise ValueError("invalid F32 QSA output shape")
    raw = resolved(c["output"]["path"], Path(c["log"])).read_bytes()
    if len(raw) != 4*math.prod(ne):
        raise ValueError("invalid F32 output byte length")
    result = np.frombuffer(raw, dtype="<f4").reshape(ne[::-1])
    if not np.isfinite(result).all():
        raise ValueError("nonfinite attention output")
    return result


def cpu_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                  mask: np.ndarray, scale: float) -> np.ndarray:
    """Compute scaled F64 attention. Inputs: Q [S,H,Q,D], K/V [S,HK,KV,D]."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or mask.ndim != 2:
        raise ValueError("invalid tensor rank")
    nstreams, heads, nqueries, dk = q.shape
    if nstreams != 1 or nqueries != 4 or k.shape[0] != nstreams or v.shape[0] != nstreams:
        raise ValueError("unsupported stream/Q configuration for this reference")
    kvheads, n_kv, dk_k = k.shape[1:]
    if dk != dk_k or v.shape[:3] != k.shape[:3] or mask.shape != (nqueries, n_kv):
        raise ValueError("incompatible Q/K/V/mask")
    if heads % kvheads:
        raise ValueError("nonintegral GQA ratio")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid attention scale")
    if not (np.isfinite(q).all() and np.isfinite(k).all() and np.isfinite(v).all()):
        raise ValueError("nonfinite Q/K/V")
    out = np.empty((nstreams, nqueries, heads, v.shape[3]), dtype=np.float64)
    ratio = heads // kvheads
    selected = [np.flatnonzero(np.isfinite(mask[qi])) for qi in range(nqueries)]
    if any(len(x) == 0 for x in selected):
        raise ValueError("empty attention row")
    for qi, rows in enumerate(selected):
        mask_vals = mask[qi, rows].astype(np.float64)
        for head in range(heads):
            kh = head // ratio
            query = q[0, head, qi, :].astype(np.float64)
            keys = k[0, kh, rows, :].astype(np.float64)
            vals = v[0, kh, rows, :].astype(np.float64)
            scores = (keys @ query) * scale + mask_vals
            weights = np.exp(scores - np.max(scores))
            weights /= np.sum(weights)
            out[0, qi, head, :] = weights @ vals
    return out


def error_summary(expected: np.ndarray, actual: np.ndarray) -> dict:
    if expected.shape != actual.shape or not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError("output shape or finiteness mismatch")
    diff = actual.astype(np.float64) - expected
    sq = diff * diff
    heads = expected.shape[2]
    sum_signal = float(np.sum(expected*expected))
    head_summaries = []
    for h in range(heads):
        signal = expected[:, :, h, :]
        err = diff[:, :, h, :]
        denom = max(float(np.sum(signal * signal)), 1e-30)
        head_summaries.append({
            "head": h, "max_abs": float(np.max(np.abs(err))),
            "nmse_over_reference_energy": float(np.sum(err*err) / denom),
        })
    loc = tuple(int(x) for x in np.unravel_index(int(np.argmax(np.abs(diff))), diff.shape))
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "nmse_over_reference_energy": float(np.sum(sq) / max(sum_signal, 1e-30)),
        "max_error_at_stream_query_head_feature": loc,
        "worst_heads": sorted(head_summaries, key=lambda x: x["nmse_over_reference_energy"], reverse=True)[:8],
    }


def compare(dense_log: Path, sparse_log: Path) -> dict:
    a, b = read_capture(dense_log), read_capture(sparse_log)
    if a["meta"]["tensor"] != b["meta"]["tensor"] or a["scale"] != b["scale"]:
        raise ValueError("attention layer or scale differs")
    if a["audit"]["kv"] != b["audit"]["kv"]:
        raise ValueError("KV length differs between captures")
    arrays = {}
    for name in "qkv":
        ia, ib = a["inputs"][name], b["inputs"][name]
        for key in ("type", "ne", "nb"):
            if ia[key] != ib[key]:
                raise ValueError(f"Q/K/V {name} metadata differs: {key}")
        pa, pb = resolved(ia["path"], dense_log), resolved(ib["path"], sparse_log)
        if pa.read_bytes() != pb.read_bytes():
            raise ValueError(f"{name} span bytes differ: cannot attribute output error to kernel")
        arrays[name] = read_tensor(ia, dense_log)
    mask_a, mask_b = read_mask(a), read_mask(b)
    if not np.array_equal(mask_a.view(np.uint16), mask_b.view(np.uint16)):
        raise ValueError("selected mask input bytes differ")
    dense, sparse = read_output(a), read_output(b)
    reference = cpu_reference(arrays["q"], arrays["k"], arrays["v"], mask_a, a["scale"])
    return {
        "layer": a["meta"]["tensor"], "backend": "HIP", "device": 0,
        "dense_id": a["id"], "sparse_id": b["id"],
        "kv": int(a["audit"]["kv"]), "scale": a["scale"],
        "mask_union": int(a["audit"]["union"]),
        "inputs_and_mask_identical": True,
        "output_shape": list(reference.shape),
        "dense_vs_fp64_reference": error_summary(reference, dense),
        "sparse_vs_fp64_reference": error_summary(reference, sparse),
        "sparse_vs_dense": error_summary(dense.astype(np.float64), sparse),
        "warning": "FP64 is a mathematical reference, not a guarantee of bitwise GPU equality or downstream model correctness",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_log", type=Path)
    parser.add_argument("sparse_log", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(compare(args.dense_log, args.sparse_log), indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
