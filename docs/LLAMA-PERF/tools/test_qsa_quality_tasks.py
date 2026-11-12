#!/usr/bin/env python3
"""Host-only regression tests for QSA task-quality generation and scoring."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from generate_qsa_quality_tasks import make_tasks
from run_qsa_quality_tasks import _localhost_api, load_tasks, sha256
from score_qsa_quality_tasks import grade, score


class TaskQualityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "tasks.json"
        self.manifest.write_text(
            json.dumps({"version": 1, "tasks": make_tasks(17, records=48)}),
            encoding="utf-8")
        self.digest, self.tasks = load_tasks(self.manifest)
        self.paths = {k: self.root / f"{k}.jsonl"
                      for k in ("dense-a", "dense-b", "sparse")}
        self._write_outputs()

    def _gold_text(self, task):
        expected = task["expected"]
        return expected["value"] if expected["type"] == "literal" else json.dumps(expected["value"])

    def _write_outputs(self, changes=None):
        changes = changes or {}
        for condition, path in self.paths.items():
            records = []
            for task in self.tasks:
                text = changes.get((condition, task["id"]), self._gold_text(task))
                records.append({
                    "id": task["id"], "kind": task["kind"], "condition": condition,
                    "task_manifest_sha256": self.digest,
                    "prompt_sha256": sha256(task["prompt"].encode("utf-8")),
                    "requested_max_tokens": task["max_tokens"],
                    "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 42},
                    "text": text, "finish_reason": "stop",
                    "prompt_tokens": 20000,
                    "completion_tokens": 10,
                    "response_model": "test",
                })
            path.write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")

    def _score(self):
        return score(self.manifest, self.paths["dense-a"], self.paths["dense-b"],
                     self.paths["sparse"])

    def test_generator_deterministic_and_diverse(self):
        self.assertEqual(make_tasks(17, 48), make_tasks(17, 48))
        self.assertNotEqual(make_tasks(17, 48), make_tasks(18, 48))
        self.assertEqual(len(self.tasks), 6)
        self.assertEqual(set(t["kind"] for t in self.tasks),
                         {"retrieval", "code_reasoning", "instruction_following"})
        self.assertEqual(len({t["id"] for t in self.tasks}), 6)

    def test_all_gold_responses_pass(self):
        result = self._score()
        self.assertEqual(result["tasks"], 6)
        self.assertEqual(result["dense_a_pass"], 6)
        self.assertEqual(result["dense_b_pass"], 6)
        self.assertEqual(result["sparse_pass"], 6)
        self.assertEqual(result["dense_repeat_unstable_tasks"], 0)
        self.assertEqual(result["sparse_regressions_with_stable_control"], 0)
        self.assertEqual(result["sparse_improvements_with_stable_control"], 0)

    def test_single_sparse_regression(self):
        first = self.tasks[0]
        self._write_outputs({("sparse", first["id"]): "WRONG"})
        result = self._score()
        self.assertEqual(result["sparse_pass"], 5)
        self.assertEqual(result["sparse_regressions_with_stable_control"], 1)
        self.assertEqual(result["per_task"][0]["classification"], "sparse_regression")

    def test_single_sparse_improvement(self):
        first = self.tasks[0]
        changes = {
            ("dense-a", first["id"]): "WRONG",
            ("dense-b", first["id"]): "WRONG",
        }
        self._write_outputs(changes)
        result = self._score()
        self.assertEqual(result["sparse_improvements_with_stable_control"], 1)
        self.assertEqual(result["dense_a_pass"], 5)
        self.assertEqual(result["sparse_pass"], 6)

    def test_dense_repeat_instability_not_counted_as_regression(self):
        first = self.tasks[0]
        self._write_outputs({("dense-b", first["id"]): "WRONG",
                             ("sparse", first["id"]): "WRONG"})
        result = self._score()
        self.assertEqual(result["dense_repeat_unstable_tasks"], 1)
        self.assertEqual(result["per_task"][0]["classification"], "control_unstable")
        self.assertEqual(result["sparse_regressions_with_stable_control"], 0)

    def test_truncated_response_is_incorrect(self):
        rows = [json.loads(s) for s in self.paths["sparse"].read_text().splitlines()]
        rows[0]["finish_reason"] = "length"
        self.paths["sparse"].write_text("".join(json.dumps(r)+"\n" for r in rows))
        result = self._score()
        self.assertEqual(result["per_task"][0]["sparse"]["reason"], "generation_truncated")

    def test_json_grader_rejects_duplicate_keys_and_nonfinite(self):
        task = {"id": "unit", "expected": {"type": "json", "value": {"a": 1}}}
        self.assertEqual(grade(task, '{"a":1}'), (True, "correct"))
        self.assertEqual(grade(task, '{"a":1,"a":1}')[0], False)
        self.assertEqual(grade(task, '{"a":NaN}')[0], False)
        self.assertEqual(grade(task, '{"a":true}')[0], False)
        self.assertFalse(grade(task, "a fenced response, not JSON")[0])

    def test_json_key_and_type_strictness(self):
        task = {"id": "unit", "expected": {
            "type": "json", "value": {"x": 2, "tags": ["A", "B"]}}}
        self.assertTrue(grade(task, '{"tags":["A","B"],"x":2}')[0])
        self.assertFalse(grade(task, '{"tags":["B","A"],"x":2}')[0])
        self.assertFalse(grade(task, '{"tags":["A","B"],"x":"2"}')[0])
        self.assertFalse(grade(task, '{"tags":["A","B"],"x":2,"extra":0}')[0])

    def test_mismatched_output_manifest_sha_rejected(self):
        rows = [json.loads(s) for s in self.paths["sparse"].read_text().splitlines()]
        rows[0]["task_manifest_sha256"] = "invalid"
        self.paths["sparse"].write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "manifest/sampling mismatch"):
            self._score()

    def test_missing_task_rejected(self):
        rows = self.paths["dense-b"].read_text().splitlines(keepends=True)
        self.paths["dense-b"].write_text("".join(rows[:-1]))
        with self.assertRaisesRegex(ValueError, "incomplete tasks"):
            self._score()

    def test_context_below_selector_floor_recorded(self):
        rows = [json.loads(s) for s in self.paths["sparse"].read_text().splitlines()]
        rows[0]["prompt_tokens"] = 1000
        self.paths["sparse"].write_text("".join(json.dumps(r)+"\n" for r in rows))
        report = self._score()
        self.assertEqual(report["tasks_below_context_floor"], 1)
        self.assertEqual(report["dense_repeat_unstable_tasks"], 0)
        self.assertEqual(report["prompt_usage_mismatch_tasks"], 1)
        self.assertEqual(report["per_task"][0]["classification"], "prompt_usage_mismatch")

    def test_dense_repeat_prompt_usage_mismatch_is_unstable(self):
        rows = [json.loads(s) for s in self.paths["dense-b"].read_text().splitlines()]
        rows[0]["prompt_tokens"] = 19999
        self.paths["dense-b"].write_text("".join(json.dumps(r)+"\n" for r in rows))
        report = self._score()
        self.assertEqual(report["dense_repeat_unstable_tasks"], 1)
        self.assertEqual(report["prompt_usage_mismatch_tasks"], 1)
        self.assertEqual(report["per_task"][0]["classification"], "control_unstable")

    def test_local_endpoint_validation(self):
        self.assertEqual(_localhost_api("http://127.0.0.1:8080"),
                         "http://127.0.0.1:8080/v1/chat/completions")
        for bad in ("https://127.0.0.1:8080", "http://example.org:8080",
                    "http://127.0.0.1:8080/other", "http://localhost"):
            with self.assertRaises(ValueError, msg=bad):
                _localhost_api(bad)


if __name__ == "__main__":
    unittest.main()
