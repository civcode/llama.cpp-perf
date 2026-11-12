#!/usr/bin/env python3
"""CPU-only regression tests for complete sparse-run repeat comparator."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from compare_qsa_sparse_repeats import (
    PINNED_MANIFEST_SHA256, compare,
)


class SparseRepeatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.manifest = root / "tasks.json"
        self.old = root / "0030-sparse.jsonl"
        self.new = root / "0032-sparse.jsonl"
        for p in (self.manifest, self.old, self.new):
            p.write_text("synthetic fixture (content validated via stubs)\n")
        self.tasks = [
            {
                "id": f"lb2-{i:03d}", "kind": "benchmark_mcq",
                "expected": {"type": "choice", "value": "A"},
                "domain": "domain-1" if i % 2 else "domain-2",
                "max_tokens": 32,
            } for i in range(24)
        ]
        self.old_rows = {
            t["id"]: {
                "text": "The correct answer is (A)" if i < 14 else "The correct answer is (B)",
                "finish_reason": "stop",
                "prompt_tokens": 20000 + i,
            } for i, t in enumerate(self.tasks)
        }
        self.new_rows = {
            k: v.copy() for k, v in self.old_rows.items()
        }

    def run_compare(self):
        def read(path, condition, digest, tasks):
            self.assertEqual(condition, "sparse")
            self.assertEqual(digest, PINNED_MANIFEST_SHA256)
            self.assertEqual(len(tasks), 24)
            return self.old_rows if path == self.old else self.new_rows
        with (
            patch("compare_qsa_sparse_repeats.load_tasks",
                  return_value=(PINNED_MANIFEST_SHA256, self.tasks)),
            patch("compare_qsa_sparse_repeats._read_results", side_effect=read),
        ):
            return compare(self.manifest, self.old, self.new)

    def test_identical_complete_runs(self):
        report = self.run_compare()
        self.assertEqual(report["questions"], 24)
        self.assertEqual(report["previous_sparse_correct"], 14)
        self.assertEqual(report["repeat_sparse_correct"], 14)
        self.assertEqual(report["output_text_changed"], 0)
        self.assertEqual(report["finish_reason_changed"], 0)
        self.assertEqual(report["previous_correct_to_repeat_incorrect"], 0)
        self.assertEqual(report["previous_incorrect_to_repeat_correct"], 0)
        self.assertEqual(len(report["per_task"]), 24)
        self.assertTrue(report["all_sparse_responses_present"])
        self.assertEqual(report["production_promotion"], "NO_GO")

    def test_correct_to_incorrect_transition(self):
        self.new_rows[self.tasks[0]["id"]]["text"] = "B"
        report = self.run_compare()
        self.assertEqual(report["repeat_sparse_correct"], 13)
        self.assertEqual(report["previous_correct_to_repeat_incorrect"], 1)
        self.assertEqual(report["output_text_changed"], 1)

    def test_incorrect_to_correct_transition(self):
        self.new_rows[self.tasks[20]["id"]]["text"] = "A"
        report = self.run_compare()
        self.assertEqual(report["repeat_sparse_correct"], 15)
        self.assertEqual(report["previous_incorrect_to_repeat_correct"], 1)

    def test_both_correct_text_change_is_counted(self):
        self.new_rows[self.tasks[0]["id"]]["text"] = "Answer: A"
        report = self.run_compare()
        self.assertEqual(report["repeat_sparse_correct"], 14)
        self.assertEqual(report["output_text_changed"], 1)
        self.assertEqual(report["previous_correct_to_repeat_incorrect"], 0)

    def test_sparse_prompt_mismatch_rejected(self):
        self.new_rows[self.tasks[0]["id"]]["prompt_tokens"] = 19999
        with self.assertRaisesRegex(ValueError, "prompt token counts"):
            self.run_compare()

    def test_historical_sparse_wrong_correctness_rejected(self):
        self.old_rows[self.tasks[0]["id"]]["text"] = "B"
        with self.assertRaisesRegex(ValueError, "expected 14/24"):
            self.run_compare()

    def test_same_artifact_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct paths"):
            compare(self.manifest, self.old, self.old)

    def test_missing_new_file_rejected(self):
        self.new.unlink()
        with self.assertRaises(FileNotFoundError):
            self.run_compare()

    def test_wrong_manifest_rejected(self):
        with patch("compare_qsa_sparse_repeats.load_tasks",
                   return_value=("0"*64, self.tasks)):
            with self.assertRaisesRegex(ValueError, "manifest digest"):
                compare(self.manifest, self.old, self.new)

    def test_insufficient_context_rejected(self):
        self.new_rows[self.tasks[0]["id"]]["prompt_tokens"] = 16000
        self.old_rows[self.tasks[0]["id"]]["prompt_tokens"] = 16000
        with self.assertRaisesRegex(ValueError, "outside frozen context"):
            self.run_compare()


if __name__ == "__main__":
    unittest.main()
