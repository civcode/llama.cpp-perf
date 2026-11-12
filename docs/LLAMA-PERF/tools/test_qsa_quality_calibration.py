#!/usr/bin/env python3
"""Host tests for predeclared QSA task difficulty and baseline-only calibration."""
from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import unittest

from calibrate_qsa_dense_tasks import calibrate
from generate_qsa_calibrated_tasks import make_calibrated_tasks
from run_qsa_quality_tasks import load_tasks, sha256
from score_qsa_quality_tasks import grade


class CalibratedTaskTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.tasks = make_calibrated_tasks(19, 40)
        self.manifest = self.root / "tasks.json"
        self.manifest.write_text(json.dumps({"version": 1, "tasks": self.tasks}), encoding="utf-8")
        self.digest, _ = load_tasks(self.manifest)
        self.a = self.root / "dense-a.jsonl"
        self.b = self.root / "dense-b.jsonl"
        self._write()

    def _gold_text(self, t):
        value = t["expected"]["value"]
        return value if t["expected"]["type"] == "literal" else json.dumps(value)

    def _write(self, changed=None, tokens=20000):
        changed = changed or {}
        for condition, path in (("dense-a", self.a), ("dense-b", self.b)):
            rows = []
            for t in self.tasks:
                rows.append({
                    "id": t["id"], "kind": t["kind"], "condition": condition,
                    "task_manifest_sha256": self.digest,
                    "prompt_sha256": sha256(t["prompt"].encode("utf-8")),
                    "requested_max_tokens": t["max_tokens"],
                    "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 42},
                    "text": changed.get((condition, t["id"]), self._gold_text(t)),
                    "finish_reason": "stop", "prompt_tokens": tokens,
                    "completion_tokens": 10,
                })
            path.write_text("".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")

    def test_predeclared_diverse_difficulty_and_determinism(self):
        self.assertEqual(self.tasks, make_calibrated_tasks(19, 40))
        self.assertNotEqual(self.tasks, make_calibrated_tasks(20, 40))
        self.assertEqual(len(self.tasks), 12)
        self.assertEqual(len({t["id"] for t in self.tasks}), 12)
        self.assertEqual({t["difficulty"] for t in self.tasks}, {"anchored", "lookup"})
        for kind in ("retrieval", "code_reasoning", "instruction_following"):
            for difficulty in ("anchored", "lookup"):
                self.assertEqual(sum(t["kind"] == kind and t["difficulty"] == difficulty
                                     for t in self.tasks), 2)

    def test_gold_keys_match_independently_parsed_source_records(self):
        for task in self.tasks:
            body, question = task["prompt"].split("\nTASK:\n", 1)
            source = {}
            regex = (r"REC (K\d{5}) \| AMOUNT=(\d+) \| OWNER=([A-Z0-9]+)"
                     r" \| TAG=([A-Z0-9]+) \| STATUS=(\w+) \| REGION=(\w+)")
            for match in re.finditer(regex, body):
                key, amount, owner, tag, status, region = match.groups()
                source[key] = {"amount": int(amount), "owner": owner, "tag": tag,
                               "status": status, "region": region}
            self.assertEqual(len(source), 40)
            key = re.search(r"K\d{5}", question).group(0)
            self.assertIn(key, source)
            row = source[key]
            kind = task["kind"]
            difficulty = task["difficulty"]
            if kind == "retrieval":
                expected = row["tag"]
            elif kind == "code_reasoning":
                if difficulty == "anchored":
                    keys = re.findall(r"AMOUNT\((K\d{5})\)", question)
                    self.assertEqual(len(keys), 2)
                    expected = str((source[keys[0]]["amount"] +
                                    2 * source[keys[1]]["amount"]) % 97)
                else:
                    expected = str((3 * row["amount"] + 7) % 97)
            elif difficulty == "anchored":
                expected = {"id": key, "region": row["region"],
                            "amount": row["amount"]}
            else:
                expected = {"id": key, "owner": row["owner"],
                            "status": row["status"]}
            self.assertEqual(task["expected"]["value"], expected)
            self.assertTrue(grade(task, self._gold_text(task))[0])
            if difficulty == "lookup":
                self.assertNotIn(row["tag"], question)
                self.assertNotIn(row["owner"], question)

    def test_all_dense_gold_outputs_solvable(self):
        result = calibrate(self.manifest, self.a, self.b)
        self.assertEqual(result["total_tasks"], 12)
        self.assertEqual(result["dense_repeat_stable"], 12)
        self.assertEqual(result["dense_stably_solved_long_context"], 12)
        self.assertFalse(result["baseline_floor"])

    def test_all_dense_fail_is_explicit_floor(self):
        changes = {(condition, task["id"]): "WRONG"
                   for task in self.tasks for condition in ("dense-a", "dense-b")}
        self._write(changes)
        result = calibrate(self.manifest, self.a, self.b)
        self.assertEqual(result["dense_repeat_stable"], 12)
        self.assertEqual(result["dense_stably_solved_long_context"], 0)
        self.assertTrue(result["baseline_floor"])

    def test_dense_disagreement_rejects_attribution(self):
        self._write({("dense-b", self.tasks[0]["id"]): "WRONG"})
        result = calibrate(self.manifest, self.a, self.b)
        self.assertEqual(result["dense_repeat_stable"], 11)
        self.assertEqual(result["dense_stably_solved_long_context"], 11)

    def test_short_context_not_counted_as_solvable(self):
        self._write(tokens=128)
        result = calibrate(self.manifest, self.a, self.b)
        self.assertEqual(result["dense_repeat_stable"], 12)
        self.assertEqual(result["reported_long_context_eligible"], 0)
        self.assertTrue(result["baseline_floor"])

    def test_invalid_seed_and_record_bounds_rejected(self):
        for seed, records in ((-1, 500), (2**32, 500), (19, 20), (19, 1000)):
            with self.assertRaises(ValueError):
                make_calibrated_tasks(seed, records)

    def test_manifest_mismatch_rejected(self):
        rows = [json.loads(s) for s in self.a.read_text().splitlines()]
        rows[0]["task_manifest_sha256"] = "bad"
        self.a.write_text("".join(json.dumps(r)+"\n" for r in rows))
        with self.assertRaisesRegex(ValueError, "manifest/sampling mismatch"):
            calibrate(self.manifest, self.a, self.b)


if __name__ == "__main__":
    unittest.main()
