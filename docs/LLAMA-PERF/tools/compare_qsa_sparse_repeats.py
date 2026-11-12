#!/usr/bin/env python3
"""CPU-only check that a second frozen LongBench sparse run is a complete repeat.

Compare two complete, distinct sparse-result files on the EXACT immutable
24-question Iteration 0028 task manifest. This is not a device-health check,
GPU trace validator, benchmark-throughput estimator or numerical-equivalence
test. Do not merge, accept, or fill a partial run with historical responses.

The first input is the COMPLETE successful Iteration 0030 sparse JSONL. The
second is the NEW complete repeat from Iteration 0032. For attribution to
sparse vs dense, use the existing strict scorer with fresh/matched dense A/B
controls separately. Reuse controls only if the model, build, driver, PCIe,
device mapping, sampler, and chat settings are unchanged and documented.

Usage:
 python3 docs/LLAMA-PERF/tools/compare_qsa_sparse_repeats.py \
  /external/iteration-0028/tasks.json \
  /external/iteration-0030/sparse.jsonl \
  /external/iteration-0032/sparse.jsonl \
  --out /external/iteration-0032/repeat-comparison.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_qsa_quality_tasks import load_tasks
from score_qsa_quality_tasks import _read_results, grade

PINNED_MANIFEST_SHA256 = "ed01ff9392f56c2e0c1e9af35c342e8cc565c71f4c8880df496469d9105d6fb5"
FROZEN_COUNT = 24
EXPECTED_PREVIOUS_SPARSE_CORRECT = 14


def compare(manifest: Path, previous_sparse: Path, repeat_sparse: Path) -> dict:
    paths = [manifest.resolve(strict=True), previous_sparse.resolve(strict=True),
             repeat_sparse.resolve(strict=True)]
    if len(set(paths)) != 3:
        raise ValueError("manifest and both complete sparse files must be distinct paths")
    digest, tasks = load_tasks(manifest)
    if digest != PINNED_MANIFEST_SHA256 or len(tasks) != FROZEN_COUNT:
        raise ValueError("frozen 0028 manifest digest or task count differs")

    previous = _read_results(previous_sparse, "sparse", digest, tasks)
    repeated = _read_results(repeat_sparse, "sparse", digest, tasks)
    prev_correct = new_correct = 0
    text_changed = 0
    correct_to_wrong = wrong_to_correct = 0
    finish_changed = 0
    per_task = []
    for task in tasks:
        key = task["id"]
        old = previous[key]
        new = repeated[key]
        if type(old.get("prompt_tokens")) is not int or (
                old["prompt_tokens"] != new.get("prompt_tokens")):
            raise ValueError(f"prompt token counts do not match for {key}")
        if old["prompt_tokens"] < 16408 or (
                old["prompt_tokens"] + task["max_tokens"] > 65536):
            raise ValueError(f"prompt token count outside frozen context bounds for {key}")
        old_pass = grade(task, old["text"], old["finish_reason"])[0]
        new_pass = grade(task, new["text"], new["finish_reason"])[0]
        prev_correct += int(old_pass)
        new_correct += int(new_pass)
        different = old["text"] != new["text"]
        changed_finish = old["finish_reason"] != new["finish_reason"]
        text_changed += int(different)
        finish_changed += int(changed_finish)
        correct_to_wrong += int(old_pass and not new_pass)
        wrong_to_correct += int(not old_pass and new_pass)
        per_task.append({
            "id": key, "domain": task.get("domain"),
            "prompt_tokens": old["prompt_tokens"],
            "previous_correct": old_pass, "repeat_correct": new_pass,
            "answer_text_changed": different,
            "finish_reason_changed": changed_finish,
            "previous_to_repeat_loss": old_pass and not new_pass,
            "previous_to_repeat_gain": not old_pass and new_pass,
        })
    if prev_correct != EXPECTED_PREVIOUS_SPARSE_CORRECT:
        raise ValueError(
            f"previous complete 0030 sparse baseline expected 14/24 correct; got {prev_correct}/24")
    return {
        "manifest_sha256": digest,
        "questions": FROZEN_COUNT,
        "previous_sparse_correct": prev_correct,
        "repeat_sparse_correct": new_correct,
        "output_text_changed": text_changed,
        "finish_reason_changed": finish_changed,
        "previous_correct_to_repeat_incorrect": correct_to_wrong,
        "previous_incorrect_to_repeat_correct": wrong_to_correct,
        "prompt_tokens_identical": True,
        "all_sparse_responses_present": True,
        "host_stability": "NOT_VERIFIED_BY_OFFLINE_ANALYSIS",
        "actual_hip_sparse_q4_mtp_launches": "MUST_BE_TRACED_SEPARATELY",
        "matched_dense_controls": "MUST_BE_VERIFIED_SEPARATELY",
        "numerical_equivalence": "NO_GO",
        "production_promotion": "NO_GO",
        "per_task": per_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("previous_sparse", type=Path)
    parser.add_argument("repeat_sparse", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare(args.manifest, args.previous_sparse, args.repeat_sparse)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "questions": report["questions"],
        "previous_sparse_correct": report["previous_sparse_correct"],
        "repeat_sparse_correct": report["repeat_sparse_correct"],
        "changed_answers": report["output_text_changed"],
        "previous_to_repeat_losses": report["previous_correct_to_repeat_incorrect"],
        "production_promotion": "NO_GO",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
