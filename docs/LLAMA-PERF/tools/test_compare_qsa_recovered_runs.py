#!/usr/bin/env python3
"""CPU-only tests for recovered-host matched dense and sparse evidence."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compare_qsa_recovered_runs import FROZEN_MANIFEST_SHA256, analyze


class RecoveredRunComparisonTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.paths = [self.root / x for x in
                      ("tasks.json", "old-a.jsonl", "old-b.jsonl",
                       "new-a.jsonl", "new-b.jsonl", "fresh-sparse.jsonl")]
        for path in self.paths:
            path.write_text("synthetic mock; the real loader is patched\n")
        self.tasks = [{
            "id": f"lb2-item-{i:03d}", "kind": "benchmark_mcq",
            "domain": "test-domain", "max_tokens": 32,
            "expected": {"type": "choice", "value": "A"},
        } for i in range(24)]
        self.outputs = {}
        for label in ("old-a", "old-b", "new-a", "new-b", "sparse"):
            self.outputs[label] = {
                t["id"]: {
                    "text": "The correct answer is (A)" if i < 14 else
                            "The correct answer is (B)",
                    "finish_reason": "stop",
                    "prompt_tokens": 20000 + i,
                }
                for i, t in enumerate(self.tasks)
            }

    def _run(self, sparse=True):
        def read(path, condition, digest, tasks):
            self.assertEqual(digest, FROZEN_MANIFEST_SHA256)
            self.assertEqual(len(tasks), 24)
            return self.outputs[
                "sparse" if condition == "sparse" else Path(path).stem]
        with (
            patch("compare_qsa_recovered_runs.load_tasks",
                  return_value=(FROZEN_MANIFEST_SHA256, self.tasks)),
            patch("compare_qsa_recovered_runs._read_results", side_effect=read),
        ):
            return analyze(*self.paths[:5],
                           self.paths[5] if sparse else None)

    def test_full_24_new_controls_and_sparse(self):
        report = self._run()
        self.assertEqual(report["old_dense_correct"], 14)
        self.assertEqual(report["new_dense_correct"], 14)
        self.assertEqual(report["new_sparse_correct"], 14)
        self.assertEqual(report["new_sparse_regressions"], 0)
        self.assertEqual(report["new_sparse_improvements"], 0)
        self.assertEqual(report["old_to_new_dense_output_changes"], 0)
        self.assertTrue(report["new_sparse_complete"])
        self.assertEqual(report["quality_gate"], "NOT_ESTABLISHED")

    def test_dense_only_mode_does_not_claim_sparse_result(self):
        report = self._run(sparse=False)
        self.assertFalse(report["new_sparse_complete"])
        self.assertIsNone(report["new_sparse_correct"])
        self.assertIsNone(report["new_sparse_regressions"])

    def test_new_baseline_regression_changes_are_separate(self):
        item = self.tasks[0]["id"]
        for group in ("new-a", "new-b"):
            self.outputs[group][item]["text"] = "B"
        report = self._run()
        self.assertEqual(report["old_dense_correct"], 14)
        self.assertEqual(report["new_dense_correct"], 13)
        self.assertEqual(report["old_to_new_dense_output_changes"], 1)
        self.assertEqual(report["new_sparse_improvements"], 1)
        self.assertEqual(report["new_sparse_regressions"], 0)

    def test_sparse_regression_compared_to_new_not_old(self):
        item = self.tasks[0]["id"]
        self.outputs["sparse"][item]["text"] = "B"
        report = self._run()
        self.assertEqual(report["new_sparse_correct"], 13)
        self.assertEqual(report["new_sparse_regressions"], 1)

    def test_new_dense_repeat_instability_fails_closed(self):
        self.outputs["new-b"][self.tasks[0]["id"]]["text"] = "B"
        with self.assertRaisesRegex(ValueError, "new dense A/B repeats"):
            self._run()

    def test_different_tokenizer_usage_fails_closed(self):
        self.outputs["new-a"][self.tasks[0]["id"]]["prompt_tokens"] += 1
        self.outputs["new-b"][self.tasks[0]["id"]]["prompt_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "old/new tokenizer prompt lengths"):
            self._run()

    def test_old_dense_score_is_pinned(self):
        item = self.tasks[20]["id"]
        for group in ("old-a", "old-b"):
            self.outputs[group][item]["text"] = "A"
        with self.assertRaisesRegex(ValueError, "historical 0028 dense correctness"):
            self._run()

    def test_sparse_tokenization_mismatch_fails_closed(self):
        self.outputs["sparse"][self.tasks[0]["id"]]["prompt_tokens"] += 1
        with self.assertRaisesRegex(ValueError, "sparse prompt tokenization"):
            self._run()

    def test_fresh_file_must_be_different_path(self):
        with (
            patch("compare_qsa_recovered_runs.load_tasks",
                  return_value=(FROZEN_MANIFEST_SHA256, self.tasks)),
        ):
            with self.assertRaisesRegex(ValueError, "every artifact path must be distinct"):
                analyze(self.paths[0], self.paths[1], self.paths[2],
                        self.paths[1], self.paths[4])

    def test_missing_fresh_sparse_file_is_not_partial_success(self):
        self.paths[5].unlink()
        with self.assertRaises(FileNotFoundError):
            self._run()

    def test_invalid_frozen_manifest_fails_closed(self):
        with patch("compare_qsa_recovered_runs.load_tasks",
                   return_value=("0"*64, self.tasks)):
            with self.assertRaisesRegex(ValueError, "frozen 0028 manifest"):
                analyze(*self.paths[:5])


if __name__ == "__main__":
    unittest.main()
