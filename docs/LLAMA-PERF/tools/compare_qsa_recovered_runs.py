#!/usr/bin/env python3
"""CPU-only comparison for the frozen 0028 task set after PCIe Gen3 recovery.

The recovered host has CHANGED the NVIDIA PCIe link configuration, so fresh
dense A/B controls are compulsory. This tool checks full frozen manifest
alignment and freshness-by-distinct-path (not proof of execution provenance),
compares new dense repeats against the old controls, and optionally compares
a full *new* sparse run against the new matched dense controls.

It never launches inference, performs GPU recovery, examines devices, verifies
sparse HIP kernel launches or approves production promotion. The local agent
must independently verify run logs/configuration, per-request HIP sparse Q4
target-verification launches and host stability. It cannot accept any partial
sparse JSONL or concatenate the interrupted 0028 prefix.

Example:
 python3 docs/LLAMA-PERF/tools/compare_qsa_recovered_runs.py \
   /external/0028/tasks.json /external/0028/dense-a.jsonl \
   /external/0028/dense-b.jsonl /external/0030/dense-a.jsonl \
   /external/0030/dense-b.jsonl --out /external/0030/dense-review.json
Then, after a complete new sparse process:
 python3 docs/LLAMA-PERF/tools/compare_qsa_recovered_runs.py \
   /external/0028/tasks.json /external/0028/dense-a.jsonl \
   /external/0028/dense-b.jsonl /external/0030/dense-a.jsonl \
   /external/0030/dense-b.jsonl --sparse /external/0030/sparse.jsonl \
   --out /external/0030/full-review.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_qsa_quality_tasks import load_tasks
from score_qsa_quality_tasks import _read_results, grade

FROZEN_MANIFEST_SHA256 = "ed01ff9392f56c2e0c1e9af35c342e8cc565c71f4c8880df496469d9105d6fb5"
FROZEN_COUNT = 24
CONTEXT_FLOOR = 16408
CONTEXT_LIMIT = 65536


def analyze(manifest: Path, old_a: Path, old_b: Path,
            new_a: Path, new_b: Path, sparse: Path | None = None) -> dict:
    inputs = [manifest, old_a, old_b, new_a, new_b]
    if sparse is not None:
        inputs.append(sparse)
    canonical = [p.resolve(strict=True) for p in inputs]
    if len(set(canonical)) != len(canonical):
        raise ValueError("every artifact path must be distinct; never reuse an earlier run file")

    digest, tasks = load_tasks(manifest)
    if digest != FROZEN_MANIFEST_SHA256 or len(tasks) != FROZEN_COUNT:
        raise ValueError("frozen 0028 manifest SHA-256/count mismatch")
    runs = {
        "old-a": _read_results(old_a, "dense-a", digest, tasks),
        "old-b": _read_results(old_b, "dense-b", digest, tasks),
        "new-a": _read_results(new_a, "dense-a", digest, tasks),
        "new-b": _read_results(new_b, "dense-b", digest, tasks),
    }
    if sparse is not None:
        runs["sparse"] = _read_results(sparse, "sparse", digest, tasks)

    old_correct = new_correct = 0
    old_to_new_text_changes = 0
    sparse_correct = sparse_regressions = sparse_improvements = 0
    sparse_text_changes = 0
    details = []
    for task in tasks:
        key = task["id"]
        olda, oldb, newa, newb = (
            runs[name][key] for name in ("old-a", "old-b", "new-a", "new-b"))
        for pair_name, x, y in (
                ("old", olda, oldb), ("new", newa, newb)):
            if (x["text"] != y["text"] or
                    x["finish_reason"] != y["finish_reason"] or
                    type(x["prompt_tokens"]) is not int or
                    x["prompt_tokens"] != y["prompt_tokens"]):
                raise ValueError(f"{pair_name} dense A/B repeats are not identical for {key}")
            if not CONTEXT_FLOOR <= x["prompt_tokens"] or (
                    x["prompt_tokens"] + task["max_tokens"] > CONTEXT_LIMIT):
                raise ValueError(f"{pair_name} prompt outside eligibility/context for {key}")
        # A new PCIe Gen3 link setting is a changed host configuration: the
        # new A/B results are the only matched dense baseline for new sparse.
        if newa["prompt_tokens"] != olda["prompt_tokens"]:
            raise ValueError(f"old/new tokenizer prompt lengths differ for {key}")
        a = grade(task, olda["text"], olda["finish_reason"])[0]
        b = grade(task, newa["text"], newa["finish_reason"])[0]
        old_correct += int(a)
        new_correct += int(b)
        old_to_new_text_changes += int(
            olda["text"] != newa["text"] or
            olda["finish_reason"] != newa["finish_reason"])
        record = {
            "id": key,
            "domain": task.get("domain"),
            "prompt_tokens": newa["prompt_tokens"],
            "old_dense_correct": a,
            "new_dense_correct": b,
            "old_vs_new_dense_text_or_finish_changed": (
                olda["text"] != newa["text"] or
                olda["finish_reason"] != newa["finish_reason"]),
        }
        if sparse is not None:
            rs = runs["sparse"][key]
            if (rs["prompt_tokens"] != newa["prompt_tokens"] or
                    rs["finish_reason"] not in ("stop", "length")):
                raise ValueError(f"sparse prompt tokenization or finish metadata mismatch: {key}")
            c = grade(task, rs["text"], rs["finish_reason"])[0]
            record.update({
                "sparse_correct": c,
                "sparse_vs_new_dense_text_changed": rs["text"] != newa["text"],
                "sparse_regression_on_new_baseline": b and not c,
                "sparse_improvement_on_new_baseline": not b and c,
            })
            sparse_correct += int(c)
            sparse_regressions += int(b and not c)
            sparse_improvements += int(not b and c)
            sparse_text_changes += int(rs["text"] != newa["text"])
        details.append(record)
    # Verify report 0028's fixed historical result, but never reuse it as
    # the *matched* baseline when PCIe link generation has changed.
    if old_correct != 14:
        raise ValueError(f"historical 0028 dense correctness was 14/24, got {old_correct}")
    return {
        "manifest_sha256": digest,
        "question_count": len(tasks),
        "old_dense_correct": old_correct,
        "new_dense_correct": new_correct,
        "old_to_new_dense_output_changes": old_to_new_text_changes,
        "new_sparse_complete": sparse is not None,
        "new_sparse_correct": sparse_correct if sparse is not None else None,
        "new_sparse_text_changes": sparse_text_changes if sparse is not None else None,
        "new_sparse_regressions": sparse_regressions if sparse is not None else None,
        "new_sparse_improvements": sparse_improvements if sparse is not None else None,
        "matched_baseline": "new dense A/B from recovered PCIe Gen3 host",
        "local_execution_provenance": "MUST_BE_VERIFIED_FROM_EXTERNAL_RUN_LOGS",
        "actual_sparse_hip_q4_launches": "MUST_BE_VERIFIED_SEPARATELY",
        "current_nvidia_health": "NOT_CERTIFIED_BY_OFFLINE_ANALYZER",
        "quality_gate": "NOT_ESTABLISHED",
        "production_promotion": "NO_GO",
        "per_task": details,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("old_dense_a", type=Path)
    p.add_argument("old_dense_b", type=Path)
    p.add_argument("new_dense_a", type=Path)
    p.add_argument("new_dense_b", type=Path)
    p.add_argument("--sparse", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        result = analyze(
            args.manifest, args.old_dense_a, args.old_dense_b,
            args.new_dense_a, args.new_dense_b, args.sparse)
        with args.out.open("x", encoding="utf-8") as out:
            json.dump(result, out, indent=2, allow_nan=False)
            out.write("\n")
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        p.error(str(exc))
    print(json.dumps({
        "questions": result["question_count"],
        "new_dense_correct": result["new_dense_correct"],
        "new_sparse_complete": result["new_sparse_complete"],
        "new_sparse_correct": result["new_sparse_correct"],
        "new_sparse_regressions": result["new_sparse_regressions"],
        "quality_gate": "NOT_ESTABLISHED",
        "summary": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
