#!/usr/bin/env python3
"""Compare deterministic QSA task outcomes with exact, non-model gold answers.

No generated code is ever executed. Strict literal and JSON validators score
fictional ledger retrieval, static code reasoning and JSON instruction following.
The report is descriptive, with no sparse quality acceptance threshold.
Dense A/B output instability is explicitly reported and blocks interpretation.

 python3 docs/LLAMA-PERF/tools/score_qsa_quality_tasks.py \
   /external/tasks.json /external/dense-a.jsonl /external/dense-b.jsonl \
   /external/sparse.jsonl --out /external/quality-summary.json \
   --require-stable-controls
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from run_qsa_quality_tasks import CONDITIONS, load_tasks, sha256


def _unique_dict(pairs: list[tuple]) -> dict:
    result = {}
    for key, val in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = val
    return result


def _reject_constant(value: str):
    raise ValueError(f"nonfinite JSON constant: {value}")


def _exact_json(a, b) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_exact_json(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_exact_json(x, y) for x, y in zip(a, b))
    return a == b


def grade(task: dict, output: str, finish: str = "stop") -> tuple[bool, str]:
    if finish == "length":
        return False, "generation_truncated"
    if finish != "stop":
        return False, "unexpected_finish_reason"
    if not isinstance(output, str):
        return False, "missing_text"
    criterion = task["expected"]
    if criterion["type"] == "literal":
        value = criterion["value"]
        if not isinstance(value, str):
            raise ValueError(f"invalid literal gold label for {task['id']}")
        return (output.strip() == value, "correct" if output.strip() == value else "incorrect_literal")
    if criterion["type"] == "choice":
        gold = criterion.get("value")
        if gold not in ("A", "B", "C", "D"):
            raise ValueError(f"invalid fixed-choice gold for {task['id']}")
        # Permit a limited, explicitly specified set of four-choice answer
        # forms, but never substring-search a free-form explanation.
        answer = output.strip()
        match = re.fullmatch(
            r"(?:The correct answer is\s*|Answer:\s*|Option\s*)?"
            r"\(?([A-D])\)?\.?\s*",
            answer, flags=re.IGNORECASE)
        if match is None:
            return False, "invalid_or_ambiguous_choice"
        chosen = match.group(1).upper()
        return chosen == gold, "correct" if chosen == gold else "incorrect_choice"
    if criterion["type"] == "json":
        gold = criterion["value"]
        if not isinstance(gold, (dict, list)):
            raise ValueError(f"invalid structured gold label for {task['id']}")
        try:
            parsed = json.loads(output, object_pairs_hook=_unique_dict,
                                parse_constant=_reject_constant)
        except (ValueError, TypeError, json.JSONDecodeError):
            return False, "invalid_json_or_duplicate_field"
        match = _exact_json(parsed, gold)
        return match, "correct" if match else "incorrect_json_value"
    raise ValueError(f"unsupported gold label for {task['id']}")


def _read_results(path: Path, condition: str, digest: str, tasks: list[dict]) -> dict:
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError(f"output file exceeds 2 MiB: {path}")
    tasks_by_id = {t["id"]: t for t in tasks}
    found = {}
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("expected output object")
                key = record["id"]
                if key not in tasks_by_id or key in found:
                    raise ValueError(f"missing/duplicate/unexpected task id {key}")
                task = tasks_by_id[key]
                if (record.get("condition") != condition or
                    record.get("kind") != task["kind"] or
                    record.get("task_manifest_sha256") != digest or
                    record.get("prompt_sha256") != sha256(task["prompt"].encode("utf-8")) or
                    record.get("requested_max_tokens") != task["max_tokens"] or
                    record.get("sampling") != {"temperature": 0.0, "top_p": 1.0, "seed": 42}):
                    raise ValueError(f"condition/manifest/sampling mismatch at {key}")
                if (not isinstance(record.get("text"), str) or
                    len(record["text"].encode("utf-8")) > 16384 or
                    record.get("finish_reason") not in ("stop", "length")):
                    raise ValueError(f"invalid model response at {key}")
                prompt_tokens = record.get("prompt_tokens")
                if prompt_tokens is not None and (
                        type(prompt_tokens) is not int or prompt_tokens < 0):
                    raise ValueError(f"invalid prompt token count at {key}")
                found[key] = record
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as err:
                raise ValueError(f"{path}:{lineno}: {err}") from err
    if set(found) != tasks_by_id.keys():
        raise ValueError(f"incomplete tasks in {path}: expected {sorted(tasks_by_id)}, got {sorted(found)}")
    return found


def score(tasks_file: Path, dense_a: Path, dense_b: Path, sparse: Path,
          context_floor: int = 16408) -> dict:
    if not 0 <= context_floor <= 65536:
        raise ValueError("context_floor must be 0..65536")
    digest, tasks = load_tasks(tasks_file)
    outputs = {
        "dense-a": _read_results(dense_a, "dense-a", digest, tasks),
        "dense-b": _read_results(dense_b, "dense-b", digest, tasks),
        "sparse": _read_results(sparse, "sparse", digest, tasks),
    }
    categories = defaultdict(lambda: {
        "tasks": 0, "dense_a_pass": 0, "dense_b_pass": 0, "sparse_pass": 0,
        "dense_repeat_text_disagreements": 0, "sparse_regressions": 0,
        "sparse_improvements": 0, "same_score": 0,
    })
    details = []
    difficulty_summary = defaultdict(lambda: {
        "tasks": 0, "dense_a_pass": 0, "dense_b_pass": 0, "sparse_pass": 0,
        "stable_dense_solved": 0, "sparse_regressions": 0,
        "sparse_improvements": 0, "control_unstable": 0,
    })
    benchmark_domains = defaultdict(lambda: {
        "tasks": 0, "dense_a_pass": 0, "dense_b_pass": 0, "sparse_pass": 0,
        "stable_dense_solved": 0, "sparse_regressions": 0,
        "sparse_improvements": 0, "control_unstable": 0,
    })
    controls_unstable = 0
    prompt_usage_mismatch = 0
    below_floor = 0
    unknown_usage = 0
    observed_output_changes = 0

    for task in tasks:
        key = task["id"]
        run = {name: result[key] for name, result in outputs.items()}
        scored = {name: grade(task, run[name]["text"], run[name]["finish_reason"])
                  for name in CONDITIONS}
        texts_stable = run["dense-a"]["text"] == run["dense-b"]["text"]
        finishes_stable = run["dense-a"]["finish_reason"] == run["dense-b"]["finish_reason"]
        counts = [run[name]["prompt_tokens"] for name in CONDITIONS]
        dense_counts = counts[:2]
        # Missing usage cannot establish that the tokenized inputs align.
        same_dense_counts = (all(n is not None for n in dense_counts) and
                             dense_counts[0] == dense_counts[1])
        stable = texts_stable and finishes_stable and same_dense_counts
        # Tokenization mismatches between dense and sparse are a separate
        # confounder, never a *dense repeat* instability.
        same_all_counts = all(n is not None for n in counts) and len(set(counts)) == 1
        prompt_usage_mismatch += int(not same_all_counts)
        category = categories[task["kind"]]
        tier = task.get("difficulty", "unclassified")
        if tier not in ("anchored", "lookup", "unclassified", "benchmark_easy", "benchmark_hard"):
            raise ValueError(f"invalid task difficulty {tier!r} at {key}")
        difficulty = difficulty_summary[tier]
        difficulty["tasks"] += 1
        difficulty["dense_a_pass"] += int(scored["dense-a"][0])
        difficulty["dense_b_pass"] += int(scored["dense-b"][0])
        difficulty["sparse_pass"] += int(scored["sparse"][0])
        domain = task.get("domain") if task["kind"] == "benchmark_mcq" else None
        if task["kind"] == "benchmark_mcq" and (
                not isinstance(domain, str) or not domain or len(domain) > 160):
            raise ValueError(f"missing benchmark domain at {key}")
        domain_group = benchmark_domains[domain] if domain is not None else None
        if domain_group is not None:
            domain_group["tasks"] += 1
            domain_group["dense_a_pass"] += int(scored["dense-a"][0])
            domain_group["dense_b_pass"] += int(scored["dense-b"][0])
            domain_group["sparse_pass"] += int(scored["sparse"][0])
        category["tasks"] += 1
        for name, label in (("dense-a", "dense_a_pass"), ("dense-b", "dense_b_pass"),
                            ("sparse", "sparse_pass")):
            category[label] += int(scored[name][0])
        category["dense_repeat_text_disagreements"] += int(not texts_stable)
        controls_unstable += int(not stable)
        observed_output_changes += int(run["dense-a"]["text"] != run["sparse"]["text"])
        if all(count is not None for count in counts):
            minimum_count = min(counts)
            eligible_length = minimum_count >= context_floor
            below_floor += int(not eligible_length)
        else:
            eligible_length = None
            unknown_usage += 1
        if not stable:
            difficulty["control_unstable"] += 1
            if domain_group is not None:
                domain_group["control_unstable"] += 1
        # Baseline solvability depends only on the two dense controls, not on
        # whether the sparse request happened to report matching token usage.
        if stable and dense_counts[0] >= context_floor and scored["dense-a"][0]:
            difficulty["stable_dense_solved"] += 1
            if domain_group is not None:
                domain_group["stable_dense_solved"] += 1
        classification = "control_unstable"
        if stable and not same_all_counts:
            classification = "prompt_usage_mismatch"
        elif stable and eligible_length is False:
            classification = "below_context_floor"
        elif stable:
            if scored["dense-a"][0] and not scored["sparse"][0]:
                classification = "sparse_regression"
                category["sparse_regressions"] += 1
                difficulty["sparse_regressions"] += 1
                if domain_group is not None:
                    domain_group["sparse_regressions"] += 1
            elif not scored["dense-a"][0] and scored["sparse"][0]:
                classification = "sparse_improvement"
                category["sparse_improvements"] += 1
                difficulty["sparse_improvements"] += 1
                if domain_group is not None:
                    domain_group["sparse_improvements"] += 1
            else:
                classification = "same_score"
                category["same_score"] += 1

        details.append({
            "id": key, "kind": task["kind"], "difficulty": tier,
            "benchmark_domain": domain,
            "classification": classification, "dense_repeat_stable": stable,
            "prompt_usage_aligned": same_all_counts,
            "dense_a": {"pass": scored["dense-a"][0], "reason": scored["dense-a"][1]},
            "dense_b": {"pass": scored["dense-b"][0], "reason": scored["dense-b"][1]},
            "sparse": {"pass": scored["sparse"][0], "reason": scored["sparse"][1]},
            "sparse_output_changed": run["dense-a"]["text"] != run["sparse"]["text"],
            "prompt_tokens": {name: run[name]["prompt_tokens"] for name in CONDITIONS},
            "length_eligible_by_usage": eligible_length,
            "finish_reason": {name: run[name]["finish_reason"] for name in CONDITIONS},
        })
    cases = len(tasks)
    return {
        "manifest": str(tasks_file),
        "manifest_sha256": digest,
        "quality_gate": "NOT_ESTABLISHED",
        "production_promotion": "NO_GO",
        "tasks": cases,
        "dense_repeat_unstable_tasks": controls_unstable,
        "prompt_usage_mismatch_tasks": prompt_usage_mismatch,
        "sparse_changed_output_tasks": observed_output_changes,
        "tasks_below_context_floor": below_floor,
        "tasks_without_complete_usage": unknown_usage,
        "context_floor_tokens": context_floor,
        "dense_a_pass": sum(v["dense_a_pass"] for v in categories.values()),
        "dense_b_pass": sum(v["dense_b_pass"] for v in categories.values()),
        "sparse_pass": sum(v["sparse_pass"] for v in categories.values()),
        "sparse_regressions_with_stable_control": sum(v["sparse_regressions"] for v in categories.values()),
        "sparse_improvements_with_stable_control": sum(v["sparse_improvements"] for v in categories.values()),
        "per_category": dict(categories),
        "per_difficulty": dict(difficulty_summary),
        "per_benchmark_domain": dict(benchmark_domains),
        "per_task": details,
        "note": (
            "A small synthetic, predeclared exact-answer task pilot; not a trained-model "
            "generalization estimate or certified held-out test. Token count is only a "
            "necessary selector precondition: validate actual Q4/GQA12 sparse HIP kernel "
            "activity independently in backend logs. No quality acceptance threshold."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("dense_a", type=Path)
    parser.add_argument("dense_b", type=Path)
    parser.add_argument("sparse", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--context-floor", type=int, default=16408)
    parser.add_argument("--require-stable-controls", action="store_true")
    args = parser.parse_args()
    try:
        report = score(args.manifest, args.dense_a, args.dense_b,
                       args.sparse, args.context_floor)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as err:
        parser.error(str(err))
    print(json.dumps({
        "tasks": report["tasks"],
        "dense_a_pass": report["dense_a_pass"],
        "dense_b_pass": report["dense_b_pass"],
        "sparse_pass": report["sparse_pass"],
        "dense_repeat_unstable_tasks": report["dense_repeat_unstable_tasks"],
        "prompt_usage_mismatch_tasks": report["prompt_usage_mismatch_tasks"],
        "sparse_regressions": report["sparse_regressions_with_stable_control"],
        "sparse_improvements": report["sparse_improvements_with_stable_control"],
        "below_context_floor": report["tasks_below_context_floor"],
        "missing_usage": report["tasks_without_complete_usage"],
        "summary": str(args.out),
    }, indent=2))
    return 2 if args.require_stable_controls and report["dense_repeat_unstable_tasks"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
