#!/usr/bin/env python3
"""Validate two dense controls BEFORE spending GPU work on the sparse condition.

This tool identifies baseline-solvable tasks in a *predeclared* difficulty
ladder. It never changes/generates task prompts or gold labels and does not
select tasks using sparse outputs. It can stop an all-floor experiment before
a needless sparse run. This is an experimental feasibility check, not a
numerical quality threshold or product GO decision.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_qsa_quality_tasks import load_tasks
from score_qsa_quality_tasks import _read_results, grade


def calibrate(manifest: Path, dense_a: Path, dense_b: Path,
              context_floor: int = 16408) -> dict:
    if not 0 <= context_floor <= 65536:
        raise ValueError("context floor is outside 0..65536")
    digest, tasks = load_tasks(manifest)
    a = _read_results(dense_a, "dense-a", digest, tasks)
    b = _read_results(dense_b, "dense-b", digest, tasks)
    details = []
    stable = stable_solved = zero_usage = eligible = 0
    for task in tasks:
        id = task["id"]
        x, y = a[id], b[id]
        counts = x["prompt_tokens"], y["prompt_tokens"]
        aligned = all(type(n) is int for n in counts) and counts[0] == counts[1]
        repeated = (
            x["text"] == y["text"] and
            x["finish_reason"] == y["finish_reason"] and
            aligned
        )
        correct_a, _ = grade(task, x["text"], x["finish_reason"])
        correct_b, _ = grade(task, y["text"], y["finish_reason"])
        token_eligible = aligned and counts[0] >= context_floor
        zero_usage += int(not aligned)
        stable += int(repeated)
        eligible += int(token_eligible)
        is_solved = repeated and token_eligible and correct_a and correct_b
        stable_solved += int(is_solved)
        details.append({
            "id": id, "kind": task["kind"],
            "difficulty": task.get("difficulty", "unclassified"),
            "dense_repeat_identical": repeated,
            "dense_a_correct": correct_a,
            "dense_b_correct": correct_b,
            "context_eligible_by_reported_tokens": token_eligible,
            "dense_stably_solved": is_solved,
            "prompt_tokens": {"dense_a": counts[0], "dense_b": counts[1]},
        })
    return {
        "manifest": str(manifest), "manifest_sha256": digest,
        "total_tasks": len(tasks), "dense_repeat_stable": stable,
        "reported_long_context_eligible": eligible,
        "no_aligned_prompt_usage": zero_usage,
        "dense_stably_solved_long_context": stable_solved,
        "baseline_floor": stable_solved == 0,
        "per_task": details,
        "note": ("Baseline-only feasibility test; does not establish actual "
                 "sparse kernel dispatch or sparse performance/quality. "
                 "Predeclared full task suite must be preserved if sparse "
                 "runs follow; do not cherry-pick baseline-solved cases."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dense_a", type=Path)
    parser.add_argument("dense_b", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-floor", type=int, default=16408)
    parser.add_argument("--require-solvable", action="store_true")
    args = parser.parse_args()
    try:
        result = calibrate(args.manifest, args.dense_a, args.dense_b, args.context_floor)
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as err:
        parser.error(str(err))
    print(json.dumps({
        "total_tasks": result["total_tasks"],
        "dense_repeat_stable": result["dense_repeat_stable"],
        "long_context_by_usage": result["reported_long_context_eligible"],
        "dense_stably_solved": result["dense_stably_solved_long_context"],
        "baseline_floor": result["baseline_floor"],
        "output": str(args.out),
    }, indent=2))
    return 2 if args.require_solvable and result["baseline_floor"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
