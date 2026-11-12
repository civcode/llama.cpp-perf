#!/usr/bin/env python3
"""Summarize distinct QSA teacher-forced reference fixtures without a quality gate.

Input JSON manifest:
{
  "fixtures": [
    {"id":"doc-a", "origin":"independent_reference_text",
     "text_file":"/external/reference-a.txt",
     "dense_a":"/external/a", "dense_b":"/external/b",
     "sparse":"/external/s"}
  ]
}
Each output directory must contain manifest.jsonl, references.jsonl and raw
logits-*.f32 from llama-qsa-reference. This script verifies independently
hashed source texts and exact reference-label files across all three runs,
requires bitwise-identical dense A/B logits, then measures paired sparse NLL.

"independent_reference_text" means independently chosen from the previous
synthetic performance fixture; it is NOT a guarantee of training hold-out.
There are no numerical tolerance or statistical acceptance thresholds. Treat
documents (not their correlated token positions) as distinct observations.

python3 docs/LLAMA-PERF/tools/summarize_qsa_reference_fixtures.py \
  /external/iteration-0023/fixtures.json > /external/iteration-0023/summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from compare_qsa_model_state import compare as compare_state
from compare_qsa_reference_nll import compare as compare_nll


def _resolve(manifest: Path, name: str) -> Path:
    p = Path(name)
    return p if p.is_absolute() else manifest.parent / p


def _sha256(path: Path, max_bytes: int = 1024 * 1024) -> str:
    if not path.is_file() or path.stat().st_size == 0 or path.stat().st_size > max_bytes:
        raise ValueError(f"missing, empty or oversized reference input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(manifest: Path, max_rows: int = 64) -> dict:
    if not 4 <= max_rows <= 64:
        raise ValueError("max_rows must be between 4 and 64")
    if manifest.stat().st_size > 64 * 1024:
        raise ValueError("fixture manifest exceeds 64 KiB")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("fixtures"), list):
        raise ValueError("expected JSON object with fixtures list")
    fixtures = data["fixtures"]
    if not 1 <= len(fixtures) <= 12:
        raise ValueError("fixture count must be 1..12")

    seen_id = set()
    seen_text_sha = set()
    reports = []
    nll_weighted = 0.0
    nll_dense_weighted = 0.0
    nll_sparse_weighted = 0.0
    total_rows = total_worse = total_better = total_flip = 0
    doc_worse = doc_better = 0

    for i, fixture in enumerate(fixtures):
        if not isinstance(fixture, dict):
            raise ValueError(f"fixture {i} is not an object")
        for field in ("id", "origin", "text_file", "dense_a", "dense_b", "sparse"):
            if not isinstance(fixture.get(field), str) or not fixture[field]:
                raise ValueError(f"fixture {i} missing {field}")
        fid = fixture["id"]
        if fid in seen_id or len(fid) > 128:
            raise ValueError(f"duplicate or invalid fixture id {fid}")
        seen_id.add(fid)
        origin = fixture["origin"]
        if origin not in ("independent_reference_text", "synthetic_performance_fixture"):
            raise ValueError(f"unrecognized fixture provenance: {origin}")

        text_path = _resolve(manifest, fixture["text_file"])
        digest = _sha256(text_path)
        if digest in seen_text_sha:
            raise ValueError(f"duplicate source text digest for {fid}")
        seen_text_sha.add(digest)
        paths = {key: _resolve(manifest, fixture[key]) for key in ("dense_a", "dense_b", "sparse")}
        ref_files = [paths[k] / "references.jsonl" for k in ("dense_a", "dense_b", "sparse")]
        ref_sha = [_sha256(p, max_bytes=2*1024*1024) for p in ref_files]
        if ref_sha[0] != ref_sha[1] or ref_sha[0] != ref_sha[2]:
            raise ValueError(f"reference-token label files differ for {fid}")
        logs = {k: d / "manifest.jsonl" for k, d in paths.items()}

        # Bitwise equality of every full F32 dense-control row is stronger than
        # an equal average NLL. This fails closed on event/history misalignment.
        control = compare_state(logs["dense_a"], logs["dense_b"])
        if control["changed_snapshots"] != 0:
            raise ValueError(f"dense-repeat logits disagree for {fid}")
        if control["tensor_snapshots"] > max_rows:
            raise ValueError(f"control row count exceeds --max-rows for {fid}")

        result = compare_nll(logs["dense_a"], logs["sparse"],
                             ref_files[0], max_rows=max_rows, max_examples=0)
        if result["rows"] != control["tensor_snapshots"]:
            raise ValueError(f"incomplete or additional logits tensors for {fid}")
        rows = result["rows"]
        delta = result["mean_sparse_minus_dense_nll_nats"]
        if not all(math.isfinite(result[k]) for k in (
                "dense_mean_nll_nats", "sparse_mean_nll_nats",
                "mean_sparse_minus_dense_nll_nats")):
            raise ValueError(f"nonfinite NLL summary for {fid}")
        nll_weighted += rows * delta
        nll_dense_weighted += rows * result["dense_mean_nll_nats"]
        nll_sparse_weighted += rows * result["sparse_mean_nll_nats"]
        total_rows += rows
        total_worse += result["sparse_rows_worse_nll"]
        total_better += result["sparse_rows_better_nll"]
        total_flip += result["top1_choice_changed_rows"]
        doc_worse += int(delta > 0)
        doc_better += int(delta < 0)
        reports.append({
            "id": fid, "origin": origin, "source_sha256": digest,
            "rows": rows,
            "dense_repeat_bitwise_equal": True,
            "dense_mean_nll_nats": result["dense_mean_nll_nats"],
            "sparse_mean_nll_nats": result["sparse_mean_nll_nats"],
            "mean_sparse_minus_dense_nll_nats": delta,
            "median_token_delta_nll_nats": result["median_sparse_minus_dense_nll_nats"],
            "worst_token_regression_nats": result["max_sparse_minus_dense_nll_nats"],
            "best_token_improvement_nats": result["min_sparse_minus_dense_nll_nats"],
            "sparse_rows_worse": result["sparse_rows_worse_nll"],
            "sparse_rows_better": result["sparse_rows_better_nll"],
            "top1_choice_changed_rows": result["top1_choice_changed_rows"],
            "dense_reference_top1_hits": result["dense_reference_top1_hits"],
            "sparse_reference_top1_hits": result["sparse_reference_top1_hits"],
        })

    doc_deltas = sorted(r["mean_sparse_minus_dense_nll_nats"] for r in reports)
    midpoint = len(doc_deltas) // 2
    median = (doc_deltas[midpoint] if len(doc_deltas) % 2
              else (doc_deltas[midpoint - 1] + doc_deltas[midpoint]) / 2)
    return {
        "source_manifest": str(manifest),
        "independent_fixtures": len(reports),
        "independent_reference_text_fixtures": sum(
            x["origin"] == "independent_reference_text" for x in reports),
        "total_scored_rows": total_rows,
        "macro_mean_sparse_minus_dense_nll_nats": math.fsum(doc_deltas) / len(doc_deltas),
        "macro_median_sparse_minus_dense_nll_nats": median,
        "micro_mean_sparse_minus_dense_nll_nats": nll_weighted / total_rows,
        "micro_dense_reference_mean_nll_nats": nll_dense_weighted / total_rows,
        "micro_sparse_reference_mean_nll_nats": nll_sparse_weighted / total_rows,
        "documents_sparse_worse_mean_nll": doc_worse,
        "documents_sparse_better_mean_nll": doc_better,
        "individual_tokens_sparse_worse_nll": total_worse,
        "individual_tokens_sparse_better_nll": total_better,
        "individual_top1_choice_flips": total_flip,
        "per_document": reports,
        "note": ("Descriptive, independently sourced document-level exploration only. "
                 "Not a formal held-out study, significance test, quality acceptance "
                 "criterion or permission to promote sparse attention."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures_json", type=Path)
    parser.add_argument("--max-rows", type=int, default=64)
    args = parser.parse_args()
    try:
        report = summarize(args.fixtures_json, max_rows=args.max_rows)
    except (ValueError, OSError, TypeError, KeyError, json.JSONDecodeError) as err:
        parser.error(str(err))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
