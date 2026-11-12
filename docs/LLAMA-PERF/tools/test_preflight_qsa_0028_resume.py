#!/usr/bin/env python3
"""CPU-only tests for interrupted 0028 frozen-manifest and dense controls."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from preflight_qsa_0028_resume import (
    MANIFEST_SHA256, PRIOR_SHA256, SOURCE_SHA256,
    partial_sparse_prefix, preflight,
)
from run_qsa_quality_tasks import sha256


def tasks_fixture(n: int, start: int = 12) -> list[dict]:
    return [{
        "id": f"lb2-source-{i:03d}",
        "benchmark_id": f"source-{i:03d}",
        "source": "LongBench-v2",
        "kind": "benchmark_mcq",
        "difficulty": "benchmark_easy",
        "domain": "Synthetic",
        "prompt": f"Prompt {i}: choose an option",
        "max_tokens": 32,
        "expected": {"type": "choice", "value": "A"},
    } for i in range(start, start+n)]


def dense_fixture(tasks: list[dict], condition: str) -> dict:
    return {
        task["id"]: {
            "text": "The correct answer is (A)" if i < 14 else "B",
            "finish_reason": "stop",
            "prompt_tokens": 20000 + i,
        }
        for i, task in enumerate(tasks)
    }


def sparse_row(task: dict, manifest: str, i: int) -> dict:
    return {
        "id": task["id"], "kind": task["kind"],
        "condition": "sparse", "task_manifest_sha256": manifest,
        "prompt_sha256": sha256(task["prompt"].encode("utf-8")),
        "requested_max_tokens": task["max_tokens"],
        "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 42},
        "text": "The correct answer is (A)" if i < 14 else "B",
        "finish_reason": "stop", "prompt_tokens": 20000 + i,
    }


class FrozenResumeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.tasks = tasks_fixture(24)
        self.prior_tasks = tasks_fixture(12, start=0)
        self.prior = {
            "source_name": "LongBench-v2",
            "source_file_sha256": SOURCE_SHA256,
            "tasks": self.prior_tasks,
        }
        self.frozen = {
            "source_name": "LongBench-v2",
            "source_file_sha256": SOURCE_SHA256,
            "excluded_source_ids": sorted(t["benchmark_id"] for t in self.prior_tasks),
        }
        self.a = dense_fixture(self.tasks, "dense-a")
        self.b = dense_fixture(self.tasks, "dense-b")
        self.sparse_path = self.root / "historical-sparse.jsonl"
        self.sparse_path.write_text(
            "".join(json.dumps(sparse_row(t, MANIFEST_SHA256, i)) + "\n"
                    for i, t in enumerate(self.tasks[:3]))
        )

    def _preflight(self, **changes):
        old = {"source": self.root / "data.json",
               "prior": self.root / "old.json",
               "manifest": self.root / "current.json",
               "dense_a": self.root / "dense-a.jsonl",
               "dense_b": self.root / "dense-b.jsonl",
               "failed_sparse": self.sparse_path}
        old.update(changes)
        def pinned(obj, expected):
            return self.prior if expected == PRIOR_SHA256 else self.frozen
        def reads(path, condition, digest, tasks):
            self.assertEqual(digest, MANIFEST_SHA256)
            return self.a if condition == "dense-a" else self.b
        with (
            patch("preflight_qsa_0028_resume.hash_file", return_value=SOURCE_SHA256),
            patch("preflight_qsa_0028_resume._load_pinned_manifest", side_effect=pinned),
            patch("preflight_qsa_0028_resume.load_tasks", return_value=(MANIFEST_SHA256, self.tasks)),
            patch("preflight_qsa_0028_resume._read_results", side_effect=reads),
        ):
            return preflight(**old)

    def test_exact_frozen_controls_and_incomplete_prefix(self):
        result = self._preflight()
        self.assertEqual(result["frozen_questions"], 24)
        self.assertEqual(result["stable_dense_correct"], 14)
        self.assertTrue(result["dense_repeat_identical"])
        self.assertEqual(result["excluded_previous_source_ids"], 12)
        self.assertEqual(result["historical_sparse"]["historical_completed_prefix_rows"], 3)
        self.assertTrue(all(x["exact_text_match"]
                            for x in result["historical_sparse"]["observed_prefix"]))
        self.assertFalse(result["gpu_health_verified"])
        self.assertFalse(result["safe_to_resume_inference"])
        self.assertFalse(result["sparse_full_run_available"])

    def test_failed_sparse_prefix_not_full_run(self):
        with self.assertRaisesRegex(ValueError, "proper incomplete prefix"):
            full = self.root / "full.jsonl"
            full.write_text("".join(json.dumps(sparse_row(t, MANIFEST_SHA256, i))+"\n"
                                    for i, t in enumerate(self.tasks)))
            partial_sparse_prefix(full, self.tasks, MANIFEST_SHA256, self.a)

    def test_failed_sparse_prefix_out_of_order_rejected(self):
        lines = self.sparse_path.read_text().splitlines()
        self.sparse_path.write_text(lines[1]+"\n"+lines[0]+"\n")
        with self.assertRaisesRegex(ValueError, "order or identity changed"):
            partial_sparse_prefix(self.sparse_path, self.tasks, MANIFEST_SHA256, self.a)

    def test_failed_sparse_prefix_bad_sha_rejected(self):
        rows = [json.loads(s) for s in self.sparse_path.read_text().splitlines()]
        rows[0]["task_manifest_sha256"] = "wrong"
        self.sparse_path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            partial_sparse_prefix(self.sparse_path, self.tasks, MANIFEST_SHA256, self.a)

    def test_failed_sparse_prefix_malformed_json_rejected(self):
        self.sparse_path.write_text("{oops\n")
        with self.assertRaisesRegex(ValueError, "failed sparse JSONL row"):
            partial_sparse_prefix(self.sparse_path, self.tasks, MANIFEST_SHA256, self.a)

    def test_changed_dense_control_rejected(self):
        self.b[self.tasks[0]["id"]]["text"] = "B"
        with self.assertRaisesRegex(ValueError, "dense repeat evidence differs"):
            self._preflight()

    def test_too_short_context_rejected(self):
        self.b[self.tasks[0]["id"]]["prompt_tokens"] = 16000
        self.a[self.tasks[0]["id"]]["prompt_tokens"] = 16000
        with self.assertRaisesRegex(ValueError, "context not in sparse eligibility window"):
            self._preflight()

    def test_source_id_overlap_rejected(self):
        self.frozen["excluded_source_ids"] = self.frozen["excluded_source_ids"][:-1]
        with self.assertRaisesRegex(ValueError, "exclusions differ"):
            self._preflight()

    def test_wrong_dataset_digest_rejected(self):
        self.frozen["source_file_sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "publisher dataset SHA"):
            self._preflight()

    def test_bad_expected_dense_success_count_rejected(self):
        self.a[self.tasks[14]["id"]]["text"] = "A"
        self.b[self.tasks[14]["id"]]["text"] = "A"
        with self.assertRaisesRegex(ValueError, "solved=15"):
            self._preflight()

    def test_no_historical_sparse_file_needed(self):
        result = self._preflight(failed_sparse=None)
        self.assertNotIn("historical_sparse", result)
        self.assertFalse(result["safe_to_resume_inference"])


if __name__ == "__main__":
    unittest.main()
