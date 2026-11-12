#!/usr/bin/env python3
"""Descriptive sparse-vs-dense quality evidence, with *illustrative* binomial bound.

Consumes the strict score_qsa_quality_tasks.py JSON summary. The one-sided
Clopper-Pearson upper bound applies ONLY if baseline-correct benchmark
questions are independent, identically distributed Bernoulli trials sampled
without bias. Our seeded, domain-balanced public LongBench convenience sample
DOES NOT justify those assumptions, so this is NOT an actionable confidence
interval or a quality gate; the bound is a sensitivity calculation.

Always require independent provenance assessment, actual HIP sparse Q4
launch evidence for each counted item, representative task distributions
and user-approved loss margin before any production-quality claim.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def _cdf_binomial_at_most(n: int, k: int, p: float) -> float:
    if not 0 <= k <= n or not 0 <= p <= 1:
        raise ValueError("invalid binomial parameters")
    if p == 0:
        return 1.0
    if p == 1:
        return float(k == n)
    terms = [
        math.lgamma(n + 1) - math.lgamma(j + 1) -
        math.lgamma(n - j + 1) + j*math.log(p) +
        (n-j)*math.log1p(-p)
        for j in range(k+1)
    ]
    top = max(terms)
    return math.exp(top) * math.fsum(math.exp(v-top) for v in terms)


def nominal_upper(n: int, regressions: int, confidence: float = 0.95) -> float | None:
    if not (type(n) is int and 0 <= n <= 503 and
            type(regressions) is int and 0 <= regressions <= n and
            0.5 < confidence < 1):
        raise ValueError("invalid regression count or confidence level")
    if n == 0:
        return None
    if regressions == n:
        return 1.0
    low, high = 0.0, 1.0
    alpha = 1 - confidence
    for _ in range(80):
        mid = (low + high) / 2
        if _cdf_binomial_at_most(n, regressions, mid) > alpha:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def summarize(source: Path) -> dict:
    if not source.is_file() or not 0 < source.stat().st_size <= 2*1024*1024:
        raise ValueError("missing/oversized quality summary JSON")
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("quality_gate") != "NOT_ESTABLISHED":
        raise ValueError("expected strict no-gate quality summary")
    tasks = data.get("per_task")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 503:
        raise ValueError("expected 1..503 per-task records")
    if data.get("tasks") != len(tasks):
        raise ValueError("task-count mismatch")
    if (not isinstance(data.get("manifest_sha256"), str) or
            len(data["manifest_sha256"]) != 64):
        raise ValueError("missing manifest digest")

    by_domain = defaultdict(lambda: {
        "tasks": 0, "comparable": 0, "stable_baseline_correct": 0,
        "sparse_preserved": 0, "sparse_regressions": 0,
        "sparse_improvements": 0, "control_unstable": 0,
        "prompt_usage_mismatch": 0, "insufficient_length": 0,
    })
    seen = set()
    counts = {"tasks": len(tasks), "comparable": 0,
              "stable_baseline_correct": 0, "sparse_preserved": 0,
              "sparse_regressions": 0, "sparse_improvements": 0,
              "control_unstable": 0, "prompt_usage_mismatch": 0,
              "insufficient_length": 0}
    for row in tasks:
        if not isinstance(row, dict):
            raise ValueError("non-object per-task row")
        tid = row.get("id")
        if not isinstance(tid, str) or not tid or tid in seen:
            raise ValueError(f"duplicate/missing task identifier: {tid}")
        seen.add(tid)
        domain = row.get("benchmark_domain") or "non_benchmark"
        if not isinstance(domain, str) or len(domain) > 160:
            raise ValueError(f"invalid benchmark domain: {tid}")
        d = by_domain[domain]
        d["tasks"] += 1
        flags = (
            row.get("dense_repeat_stable"),
            row.get("prompt_usage_aligned"),
            row.get("length_eligible_by_usage"),
        )
        if any(type(v) is not bool for v in flags):
            raise ValueError(f"required comparability flags missing: {tid}")
        outcome = {}
        for k in ("dense_a", "dense_b", "sparse"):
            rec = row.get(k)
            if not isinstance(rec, dict) or type(rec.get("pass")) is not bool:
                raise ValueError(f"missing correctness flag {k}: {tid}")
            outcome[k] = rec["pass"]
        stable, aligned, length_ok = flags
        if not stable:
            category = "control_unstable"
        elif not aligned:
            category = "prompt_usage_mismatch"
        elif not length_ok:
            category = "insufficient_length"
        else:
            category = "comparable"
        counts[category] += 1
        d[category] += 1
        if category != "comparable":
            continue
        if outcome["dense_a"] != outcome["dense_b"]:
            raise ValueError(f"dense correctness differs despite stable controls: {tid}")
        if outcome["dense_a"]:
            counts["stable_baseline_correct"] += 1
            d["stable_baseline_correct"] += 1
            if outcome["sparse"]:
                counts["sparse_preserved"] += 1
                d["sparse_preserved"] += 1
            else:
                counts["sparse_regressions"] += 1
                d["sparse_regressions"] += 1
        elif outcome["sparse"]:
            counts["sparse_improvements"] += 1
            d["sparse_improvements"] += 1
    n = counts["stable_baseline_correct"]
    k = counts["sparse_regressions"]
    bound = nominal_upper(n, k, 0.95)
    return {
        "source": str(source),
        "manifest_sha256": data["manifest_sha256"],
        **counts,
        "observed_regression_fraction_among_stable_baseline_correct": k / n if n else None,
        "illustrative_iid_exact_upper95_regression_fraction": bound,
        "per_domain": dict(by_domain),
        "quality_gate": "NOT_ESTABLISHED",
        "production_promotion": "NO_GO",
        "kernel_launch_evidence": "MUST_BE_VERIFIED_SEPARATELY",
        "statistical_warning": (
            "Binomial one-sided Clopper-Pearson upper95 assumes iid representative "
            "tasks. A domain-balanced, public, small benchmark sample violates or "
            "does not demonstrate these assumptions. This is an illustrative "
            "sensitivity measure, NOT a valid population quality assurance bound. "
            "Do not use it as a release margin, p-value or automatic GO criterion."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("quality_summary_json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        data = summarize(args.quality_summary_json)
        with args.out.open("x", encoding="utf-8") as out:
            json.dump(data, out, indent=2, allow_nan=False)
            out.write("\n")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as err:
        parser.error(str(err))
    print(json.dumps({
        "baseline_correct": data["stable_baseline_correct"],
        "observed_regressions": data["sparse_regressions"],
        "illustrative_iid_upper95": data["illustrative_iid_exact_upper95_regression_fraction"],
        "report": str(args.out),
        "quality_gate": "NOT_ESTABLISHED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
