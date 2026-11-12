#!/usr/bin/env python3
"""Strict paired QSA task-quality audit, dense-only or dense/sparse.

Validate every counterfactual A/B pair changes exactly one target field
in one ledger row and its independently computable gold value. Audit
dense-repeat correctness and prompt eligibility *before* sparse execution,
then report counterfactual consistency for matched sparse answers. The
task and pair are units of evidence; results are descriptive with NO
numerical acceptance margin or production GO authority.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from run_qsa_quality_tasks import load_tasks
from score_qsa_quality_tasks import _read_results, grade


def _ledger_and_question(task: dict) -> tuple[list[str], str]:
    prompt = task["prompt"]
    if prompt.count("<LEDGER>\n") != 1 or prompt.count("\n</LEDGER>\nTASK:\n") != 1:
        raise ValueError(f"missing/unbalanced ledger or task delimiters: {task['id']}")
    body, question = prompt.split("\n</LEDGER>\nTASK:\n")
    lines = body.split("<LEDGER>\n", 1)[1].splitlines()
    if not lines or any(not s.startswith("REC K") for s in lines):
        raise ValueError(f"invalid ledger rows for {task['id']}")
    return lines, question


def validate_pairs(tasks: list[dict]) -> dict[str, list[dict]]:
    if len(tasks) != 12:
        raise ValueError("expected 12 tasks (6 two-variant counterfactual pairs)")
    pairs = defaultdict(dict)
    for task in tasks:
        pair = task.get("pair_id")
        variant = task.get("variant")
        band = task.get("target_band")
        if not isinstance(pair, str) or not pair or variant not in ("a", "b"):
            raise ValueError("missing/invalid pair id or variant")
        if pair in pairs and variant in pairs[pair]:
            raise ValueError(f"duplicate pair variant: {pair}/{variant}")
        if task.get("difficulty") != "lookup" or band not in ("early", "middle", "late"):
            raise ValueError(f"invalid difficulty/target band: {task['id']}")
        if task["kind"] not in ("retrieval", "instruction_following"):
            raise ValueError(f"unsupported counterfactual kind: {task['id']}")
        if task.get("changed_field") != ("tag" if task["kind"] == "retrieval" else "status"):
            raise ValueError(f"invalid changed field: {task['id']}")
        pairs[pair][variant] = task
    if len(pairs) != 6 or any(set(group) != {"a", "b"} for group in pairs.values()):
        raise ValueError("expected six complete A/B counterfactual pairs")
    result = {}
    kinds_bands = set()
    for pair_id, pair in sorted(pairs.items()):
        a, b = pair["a"], pair["b"]
        if a["kind"] != b["kind"] or a["target_id"] != b["target_id"] or (
                a["target_band"] != b["target_band"]) or (
                a["changed_field"] != b["changed_field"]):
            raise ValueError(f"counterfactual pair metadata mismatch: {pair_id}")
        kinds_bands.add((a["kind"], a["target_band"]))
        a_rows, a_question = _ledger_and_question(a)
        b_rows, b_question = _ledger_and_question(b)
        if a_question != b_question or len(a_rows) != len(b_rows):
            raise ValueError(f"counterfactual question/row count differs: {pair_id}")
        differences = [i for i, (ra, rb) in enumerate(zip(a_rows, b_rows)) if ra != rb]
        if len(differences) != 1:
            raise ValueError(f"counterfactual must change exactly one record: {pair_id}")
        idx = differences[0]
        expected_index = {"early": 3, "middle": len(a_rows)//2,
                          "late": len(a_rows)-4}[a["target_band"]]
        if idx != expected_index:
            raise ValueError(f"target band/actual source row differ: {pair_id}")
        parts_a = a_rows[idx].split(" | ")
        parts_b = b_rows[idx].split(" | ")
        if len(parts_a) != 6 or len(parts_a) != len(parts_b):
            raise ValueError(f"unrecognized target record: {pair_id}")
        if parts_a[0] != "REC " + a["target_id"] or parts_b[0] != parts_a[0]:
            raise ValueError(f"target record ID mismatch: {pair_id}")
        field_differences = [i for i, (fa, fb) in enumerate(zip(parts_a, parts_b))
                             if fa != fb]
        desired = 3 if a["changed_field"] == "tag" else 4
        if field_differences != [desired]:
            raise ValueError(f"unexpected changed field for {pair_id}")
        field = "TAG=" if desired == 3 else "STATUS="
        if not parts_a[desired].startswith(field) or not parts_b[desired].startswith(field):
            raise ValueError(f"invalid target field for {pair_id}")
        if a["kind"] == "retrieval":
            values = [parts_a[desired][len(field):], parts_b[desired][len(field):]]
            if (a["expected"] != {"type": "literal", "value": values[0]} or
                    b["expected"] != {"type": "literal", "value": values[1]}):
                raise ValueError(f"reference TAG mismatch for {pair_id}")
        else:
            values = [
                {"id": a["target_id"], "status": parts_a[desired][len(field):]},
                {"id": b["target_id"], "status": parts_b[desired][len(field):]},
            ]
            if (a["expected"] != {"type": "json", "value": values[0]} or
                    b["expected"] != {"type": "json", "value": values[1]}):
                raise ValueError(f"reference STATUS mismatch for {pair_id}")
        if values[0] == values[1]:
            raise ValueError(f"counterfactual did not change gold answer: {pair_id}")
        result[pair_id] = [a, b]
    expected_bands = {(kind, band) for kind in ("retrieval", "instruction_following")
                      for band in ("early", "middle", "late")}
    if kinds_bands != expected_bands:
        raise ValueError("counterfactual suite does not cover both kinds and all target bands")
    return result


def analyze(manifest: Path, dense_a: Path, dense_b: Path,
            sparse: Path | None = None, context_floor: int = 16408) -> dict:
    if not 0 <= context_floor <= 65536:
        raise ValueError("invalid context floor")
    digest, tasks = load_tasks(manifest)
    pairs = validate_pairs(tasks)
    data = {
        "dense-a": _read_results(dense_a, "dense-a", digest, tasks),
        "dense-b": _read_results(dense_b, "dense-b", digest, tasks),
    }
    if sparse is not None:
        data["sparse"] = _read_results(sparse, "sparse", digest, tasks)
    report = {
        "manifest_sha256": digest, "mode": "paired" if sparse else "dense_only",
        "quality_gate": "NOT_ESTABLISHED", "production_promotion": "NO_GO",
        "pairs": len(pairs), "tasks": len(tasks), "baseline_stably_solved_pairs": 0,
        "baseline_stably_solved_tasks": 0, "sparse_solved_pairs": None,
        "sparse_pair_regressions": None, "sparse_pair_improvements": None,
        "per_pair": [],
        "note": ("Predeclared synthetic counterfactual paired quality probe, "
                 "not an accuracy-equivalence or task representativeness proof. "
                 "Real per-task HIP sparse Q4 MTP launches must be checked separately."),
    }
    if sparse:
        report["sparse_solved_pairs"] = 0
        report["sparse_pair_regressions"] = 0
        report["sparse_pair_improvements"] = 0
    for pair_id, variants in pairs.items():
        per_task = []
        baseline_pair_ok = True
        sparse_pair_ok = True
        for t in variants:
            key = t["id"]
            ra = data["dense-a"][key]
            rb = data["dense-b"][key]
            dense_stable = (
                ra["text"] == rb["text"] and
                ra["finish_reason"] == rb["finish_reason"] and
                type(ra["prompt_tokens"]) is int and
                ra["prompt_tokens"] == rb["prompt_tokens"] and
                ra["prompt_tokens"] >= context_floor
            )
            correct_a = grade(t, ra["text"], ra["finish_reason"])[0]
            correct_b = grade(t, rb["text"], rb["finish_reason"])[0]
            baseline_ok = dense_stable and correct_a and correct_b
            report["baseline_stably_solved_tasks"] += int(baseline_ok)
            baseline_pair_ok &= baseline_ok
            case = {
                "id": key, "dense_stable": dense_stable,
                "dense_a_correct": correct_a, "dense_b_correct": correct_b,
                "baseline_stably_solved": baseline_ok,
                "prompt_tokens_dense_a": ra["prompt_tokens"],
            }
            if sparse:
                rs = data["sparse"][key]
                same_usage = ra["prompt_tokens"] == rs["prompt_tokens"]
                sparse_correct = grade(t, rs["text"], rs["finish_reason"])[0]
                comparable = dense_stable and same_usage
                case.update({
                    "prompt_tokens_sparse": rs["prompt_tokens"],
                    "sparse_correct": sparse_correct,
                    "prompt_usage_aligned": same_usage,
                    "sparse_vs_dense_output_differs": rs["text"] != ra["text"],
                    "comparable": comparable,
                })
                sparse_pair_ok &= comparable and sparse_correct
            per_task.append(case)
        report["baseline_stably_solved_pairs"] += int(baseline_pair_ok)
        pair_report = {
            "pair_id": pair_id, "kind": variants[0]["kind"],
            "target_band": variants[0]["target_band"],
            "baseline_stably_solved": baseline_pair_ok,
            "variants": per_task,
        }
        if sparse:
            pair_report["sparse_pair_solved"] = sparse_pair_ok
            pair_report["pair_regression"] = baseline_pair_ok and not sparse_pair_ok
            pair_report["pair_improvement"] = not baseline_pair_ok and sparse_pair_ok
            report["sparse_solved_pairs"] += int(sparse_pair_ok)
            report["sparse_pair_regressions"] += int(pair_report["pair_regression"])
            report["sparse_pair_improvements"] += int(pair_report["pair_improvement"])
        report["per_pair"].append(pair_report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest", type=Path)
    p.add_argument("dense_a", type=Path)
    p.add_argument("dense_b", type=Path)
    p.add_argument("--sparse", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--context-floor", type=int, default=16408)
    p.add_argument("--require-baseline-pair", action="store_true")
    args = p.parse_args()
    try:
        result = analyze(args.manifest, args.dense_a, args.dense_b,
                         args.sparse, args.context_floor)
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
        p.error(str(e))
    print(json.dumps({
        "mode": result["mode"], "total_pairs": result["pairs"],
        "baseline_stably_solved_pairs": result["baseline_stably_solved_pairs"],
        "sparse_solved_pairs": result["sparse_solved_pairs"],
        "sparse_pair_regressions": result["sparse_pair_regressions"],
        "sparse_pair_improvements": result["sparse_pair_improvements"],
    }, indent=2))
    return 2 if args.require_baseline_pair and result["baseline_stably_solved_pairs"] == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
