#!/usr/bin/env python3
"""Synthetic host tests for paired long-context QSA task-quality probes."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from analyze_qsa_counterfactual_pairs import analyze, validate_pairs
from generate_qsa_counterfactual_tasks import make_counterfactual_tasks
from run_qsa_quality_tasks import load_tasks, sha256
from score_qsa_quality_tasks import grade


class CounterfactualPairTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.tasks = make_counterfactual_tasks(seed=29, records=80)
        self.manifest = self.root / "tasks.json"
        self.manifest.write_text(json.dumps({"version": 1, "tasks": self.tasks}))
        self.digest, _ = load_tasks(self.manifest)
        self.paths = {k: self.root / f"{k}.jsonl"
                      for k in ("dense-a", "dense-b", "sparse")}
        self._write()

    def _gold_text(self, task):
        result = task["expected"]["value"]
        return result if task["expected"]["type"] == "literal" else json.dumps(result)

    def _write(self, changes=None, usage=None):
        changes = changes or {}
        usage = usage or {}
        for condition, path in self.paths.items():
            items = []
            for task in self.tasks:
                items.append({
                    "id": task["id"], "kind": task["kind"],
                    "condition": condition, "task_manifest_sha256": self.digest,
                    "prompt_sha256": sha256(task["prompt"].encode("utf-8")),
                    "requested_max_tokens": task["max_tokens"],
                    "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 42},
                    "text": changes.get((condition, task["id"]), self._gold_text(task)),
                    "finish_reason": "stop",
                    "prompt_tokens": usage.get((condition, task["id"]), 22000),
                    "completion_tokens": 8,
                })
            path.write_text("".join(json.dumps(item)+"\n" for item in items))

    def _analysis(self, sparse=True):
        return analyze(self.manifest, self.paths["dense-a"], self.paths["dense-b"],
                       self.paths["sparse"] if sparse else None)

    def test_determinism_and_complete_pairs(self):
        self.assertEqual(self.tasks, make_counterfactual_tasks(29, 80))
        self.assertNotEqual(self.tasks, make_counterfactual_tasks(30, 80))
        grouped = validate_pairs(self.tasks)
        self.assertEqual(len(self.tasks), 12)
        self.assertEqual(len(grouped), 6)
        self.assertEqual(set(t["target_band"] for t in self.tasks),
                         {"early", "middle", "late"})
        self.assertEqual(set(t["kind"] for t in self.tasks),
                         {"retrieval", "instruction_following"})
        for a, b in grouped.values():
            self.assertNotEqual(a["expected"], b["expected"])
            self.assertTrue(grade(a, self._gold_text(a))[0])
            self.assertTrue(grade(b, self._gold_text(b))[0])

    def test_clean_baseline_and_sparse_all_pairs_preserved(self):
        r = self._analysis()
        self.assertEqual(r["baseline_stably_solved_pairs"], 6)
        self.assertEqual(r["baseline_stably_solved_tasks"], 12)
        self.assertEqual(r["sparse_solved_pairs"], 6)
        self.assertEqual(r["sparse_pair_regressions"], 0)
        self.assertEqual(r["sparse_pair_improvements"], 0)
        self.assertEqual(r["quality_gate"], "NOT_ESTABLISHED")

    def test_dense_only_calibrates_without_sparse_data(self):
        r = self._analysis(sparse=False)
        self.assertEqual(r["mode"], "dense_only")
        self.assertEqual(r["baseline_stably_solved_pairs"], 6)
        self.assertIsNone(r["sparse_solved_pairs"])

    def test_sparse_single_variant_failure_is_pair_regression(self):
        task = self.tasks[0]
        self._write({("sparse", task["id"]): "WRONG"})
        r = self._analysis()
        self.assertEqual(r["baseline_stably_solved_pairs"], 6)
        self.assertEqual(r["sparse_solved_pairs"], 5)
        self.assertEqual(r["sparse_pair_regressions"], 1)

    def test_baseline_pair_failure_then_sparse_success_is_improvement(self):
        first = self.tasks[0]
        changes = {(condition, first["id"]): "WRONG" for condition in ("dense-a", "dense-b")}
        self._write(changes)
        r = self._analysis()
        self.assertEqual(r["baseline_stably_solved_pairs"], 5)
        self.assertEqual(r["sparse_solved_pairs"], 6)
        self.assertEqual(r["sparse_pair_improvements"], 1)

    def test_dense_repeat_instability_blocks_pair_baseline(self):
        first = self.tasks[0]
        self._write({("dense-b", first["id"]): "WRONG"})
        r = self._analysis()
        self.assertEqual(r["baseline_stably_solved_pairs"], 5)
        self.assertFalse(r["per_pair"][0]["variants"][0]["dense_stable"])

    def test_sparse_prompt_token_disagreement_blocks_pair_success(self):
        first = self.tasks[0]
        self._write(usage={("sparse", first["id"]): 21000})
        r = self._analysis()
        self.assertEqual(r["baseline_stably_solved_pairs"], 6)
        self.assertEqual(r["sparse_solved_pairs"], 5)
        self.assertFalse(r["per_pair"][0]["variants"][0]["comparable"])
        self.assertEqual(r["sparse_pair_regressions"], 0)
        self.assertIsNone(r["per_pair"][0]["sparse_pair_solved"])
        self.assertFalse(r["per_pair"][0]["comparable_pair"])

    def test_modified_second_record_is_rejected(self):
        bad = json.loads(json.dumps(self.tasks))
        task = bad[1]
        task["prompt"] = task["prompt"].replace("AMOUNT=", "AMOUNT=9", 1)
        with self.assertRaisesRegex(ValueError, "exactly one record"):
            validate_pairs(bad)

    def test_wrong_reference_answer_is_rejected(self):
        bad = json.loads(json.dumps(self.tasks))
        bad[0]["expected"]["value"] = "WRONG"
        with self.assertRaisesRegex(ValueError, "reference TAG mismatch"):
            validate_pairs(bad)

    def test_missing_pair_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 12 tasks"):
            validate_pairs(self.tasks[:-1])

    def test_empty_or_invalid_configuration_rejected(self):
        for seed, records in ((-1, 520), (2**32, 520), (5, 20), (5, 1000)):
            with self.assertRaises(ValueError):
                make_counterfactual_tasks(seed, records)


if __name__ == "__main__":
    unittest.main()
