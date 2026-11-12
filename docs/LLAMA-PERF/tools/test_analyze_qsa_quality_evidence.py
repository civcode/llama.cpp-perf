#!/usr/bin/env python3
"""Synthetic host tests for assumption-labelled QSA quality uncertainty summary."""
import json
import math
from pathlib import Path
import tempfile
import unittest

from analyze_qsa_quality_evidence import nominal_upper, summarize


def source_summary():
    rows = []
    for i in range(12):
        rows.append({
            "id": f"lb2-{i:03d}",
            "benchmark_domain": "domain-" + str(i % 3),
            "dense_repeat_stable": True,
            "prompt_usage_aligned": True,
            "length_eligible_by_usage": True,
            "dense_a": {"pass": i < 8},
            "dense_b": {"pass": i < 8},
            "sparse": {"pass": i < 8},
        })
    return {
        "quality_gate": "NOT_ESTABLISHED",
        "manifest_sha256": "a"*64,
        "tasks": 12,
        "per_task": rows,
    }


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / "score.json"
        self._write(source_summary())

    def _write(self, data):
        self.path.write_text(json.dumps(data), encoding="utf-8")

    def test_zero_regression_exact_nominal_formula(self):
        for n in (1, 8, 12, 48, 503):
            self.assertAlmostEqual(nominal_upper(n, 0),
                                   1-math.pow(.05, 1/n), places=11)

    def test_nominal_bound_is_not_available_without_baseline_successes(self):
        self.assertIsNone(nominal_upper(0, 0))
        self.assertEqual(nominal_upper(8, 8), 1.0)
        with self.assertRaises(ValueError):
            nominal_upper(8, 9)

    def test_one_regression_binomial_tail(self):
        n, k = 12, 1
        bound = nominal_upper(n, k)
        tail = sum(math.comb(n, j) * bound**j * (1-bound)**(n-j)
                   for j in range(k+1))
        self.assertAlmostEqual(tail, .05, places=10)

    def test_nominal_boundary_monotone(self):
        self.assertGreater(nominal_upper(8, 1), nominal_upper(8, 0))
        self.assertLess(nominal_upper(24, 0), nominal_upper(8, 0))

    def test_synthetic_0027_like_report(self):
        result = summarize(self.path)
        self.assertEqual(result["tasks"], 12)
        self.assertEqual(result["stable_baseline_correct"], 8)
        self.assertEqual(result["sparse_preserved"], 8)
        self.assertEqual(result["sparse_regressions"], 0)
        self.assertEqual(result["sparse_improvements"], 0)
        self.assertEqual(result["comparable"], 12)
        self.assertEqual(result["quality_gate"], "NOT_ESTABLISHED")
        self.assertEqual(result["kernel_launch_evidence"],
                         "MUST_BE_VERIFIED_SEPARATELY")
        self.assertEqual(sum(x["tasks"] for x in result["per_domain"].values()), 12)
        self.assertAlmostEqual(result["illustrative_iid_exact_upper95_regression_fraction"],
                               1-.05**(1/8))

    def test_real_sparse_regression_and_improvement(self):
        data = source_summary()
        data["per_task"][0]["sparse"]["pass"] = False
        data["per_task"][8]["sparse"]["pass"] = True
        self._write(data)
        report = summarize(self.path)
        self.assertEqual(report["sparse_regressions"], 1)
        self.assertEqual(report["sparse_improvements"], 1)
        self.assertEqual(report["stable_baseline_correct"], 8)
        self.assertGreater(report["illustrative_iid_exact_upper95_regression_fraction"],
                           1-.05**(1/8))

    def test_sparse_token_usage_mismatch_is_not_regression(self):
        data = source_summary()
        data["per_task"][0]["prompt_usage_aligned"] = False
        data["per_task"][0]["sparse"]["pass"] = False
        self._write(data)
        report = summarize(self.path)
        self.assertEqual(report["prompt_usage_mismatch"], 1)
        self.assertEqual(report["sparse_regressions"], 0)
        self.assertEqual(report["stable_baseline_correct"], 7)

    def test_baseline_unstable_is_not_attributable(self):
        data = source_summary()
        data["per_task"][0]["dense_repeat_stable"] = False
        data["per_task"][0]["dense_b"]["pass"] = False
        data["per_task"][0]["sparse"]["pass"] = False
        self._write(data)
        report = summarize(self.path)
        self.assertEqual(report["control_unstable"], 1)
        self.assertEqual(report["stable_baseline_correct"], 7)
        self.assertEqual(report["sparse_regressions"], 0)

    def test_stable_dense_conflicting_gold_fails_closed(self):
        data = source_summary()
        data["per_task"][0]["dense_b"]["pass"] = False
        self._write(data)
        with self.assertRaisesRegex(ValueError, "dense correctness differs"):
            summarize(self.path)

    def test_reject_duplicate_or_missing_ids(self):
        data = source_summary()
        data["per_task"][1]["id"] = data["per_task"][0]["id"]
        self._write(data)
        with self.assertRaisesRegex(ValueError, "duplicate/missing task"):
            summarize(self.path)

    def test_reject_manifest_or_count_mismatch(self):
        data = source_summary()
        data["manifest_sha256"] = "bad"
        self._write(data)
        with self.assertRaisesRegex(ValueError, "missing manifest digest"):
            summarize(self.path)
        data = source_summary()
        data["tasks"] = 13
        self._write(data)
        with self.assertRaisesRegex(ValueError, "task-count mismatch"):
            summarize(self.path)


if __name__ == "__main__":
    unittest.main()
