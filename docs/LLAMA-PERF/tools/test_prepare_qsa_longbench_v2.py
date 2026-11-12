#!/usr/bin/env python3
"""Host-only LongBench-v2 adapter/MCQ scoring regression tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from prepare_qsa_longbench_v2 import make_manifest
from run_qsa_quality_tasks import load_tasks, sha256
from score_qsa_quality_tasks import grade, score


def source_row(i: int, domain="Single-Document QA", difficulty="easy",
               context_size=1200):
    return {
        "_id": f"question-{i:03d}",
        "domain": domain,
        "sub_domain": "reference-extraction",
        "difficulty": difficulty,
        "length": "short",
        "question": f"Which option correctly answers question {i}?",
        "choice_A": "The first possibility",
        "choice_B": "The second possibility",
        "choice_C": "The third possibility",
        "choice_D": "The fourth possibility",
        "answer": ("A", "B", "C", "D")[i % 4],
        "context": (f"Reference document number {i}. " +
                    "Source text for deterministic testing only. " * 100)[:context_size],
    }


class LongBenchV2AdapterTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "dataset.jsonl"
        rows = [source_row(i, "domain-" + str(i % 3)) for i in range(9)]
        self.source.write_text(
            "".join(json.dumps(r)+"\n" for r in rows), encoding="utf-8")

    def test_deterministic_and_domain_balanced(self):
        a = make_manifest(self.source, seed=123, limit=6,
                          min_context_bytes=100, max_prompt_bytes=4000)
        b = make_manifest(self.source, seed=123, limit=6,
                          min_context_bytes=100, max_prompt_bytes=4000)
        self.assertEqual(a, b)
        self.assertEqual(len(a["tasks"]), 6)
        self.assertEqual(set(a["selected_domains"]),
                         {"domain-0", "domain-1", "domain-2"})
        self.assertEqual({k: sum(t["domain"] == k for t in a["tasks"])
                          for k in a["selected_domains"]},
                         {"domain-0": 2, "domain-1": 2, "domain-2": 2})
        self.assertTrue(all(t["difficulty"] == "benchmark_easy" for t in a["tasks"]))
        self.assertTrue(all(t["source"] == "LongBench-v2" for t in a["tasks"]))

    def test_compatible_with_original_task_runner_and_gold(self):
        manifest = make_manifest(self.source, seed=1, limit=4,
                                 min_context_bytes=100, max_prompt_bytes=4000)
        path = self.root / "tasks.json"
        path.write_text(json.dumps(manifest))
        digest, tasks = load_tasks(path)
        self.assertEqual(digest, sha256(path.read_bytes()))
        self.assertEqual(len(tasks), 4)
        for t in tasks:
            self.assertEqual(t["kind"], "benchmark_mcq")
            self.assertEqual(t["max_tokens"], 32)
            self.assertTrue(grade(t, "The correct answer is (" +
                                  t["expected"]["value"] + ")")[0])

    def test_benchmark_scoring_and_domain_regressions(self):
        manifest = make_manifest(self.source, seed=1, limit=4,
                                 min_context_bytes=100, max_prompt_bytes=4000)
        path = self.root / "tasks.json"
        path.write_text(json.dumps(manifest))
        digest, tasks = load_tasks(path)
        runs = {}
        for condition in ("dense-a", "dense-b", "sparse"):
            records = []
            for i, t in enumerate(tasks):
                choice = t["expected"]["value"]
                if condition == "sparse" and i == 0:
                    choice = next(c for c in ("A", "B", "C", "D") if c != choice)
                records.append({
                    "id": t["id"], "kind": t["kind"], "condition": condition,
                    "task_manifest_sha256": digest,
                    "prompt_sha256": sha256(t["prompt"].encode("utf-8")),
                    "requested_max_tokens": t["max_tokens"],
                    "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 42},
                    "text": f"The correct answer is ({choice})",
                    "finish_reason": "stop", "prompt_tokens": 22000,
                    "completion_tokens": 8,
                })
            target = self.root / f"{condition}.jsonl"
            target.write_text("".join(json.dumps(r)+"\n" for r in records))
            runs[condition] = target
        result = score(path, runs["dense-a"], runs["dense-b"], runs["sparse"])
        self.assertEqual(result["dense_a_pass"], 4)
        self.assertEqual(result["sparse_pass"], 3)
        self.assertEqual(result["sparse_regressions_with_stable_control"], 1)
        self.assertEqual(result["per_difficulty"]["benchmark_easy"]["tasks"], 4)
        self.assertEqual(sum(x["sparse_regressions"] for x in
                             result["per_benchmark_domain"].values()), 1)

    def test_gold_never_inserted_as_answer_hint(self):
        manifest = make_manifest(self.source, seed=2, limit=3,
                                 min_context_bytes=100, max_prompt_bytes=4000)
        for t in manifest["tasks"]:
            self.assertIn("Output exactly one line:", t["prompt"])
            self.assertIn("Question:", t["prompt"])
            self.assertNotIn("Correct option:", t["prompt"])

    def test_input_json_array_supported(self):
        source = self.root / "dataset.json"
        source.write_text(json.dumps([source_row(i) for i in range(4)]))
        result = make_manifest(source, limit=3,
                               min_context_bytes=100, max_prompt_bytes=4000)
        self.assertEqual(len(result["tasks"]), 3)

    def test_reject_duplicate_record_identity(self):
        self.source.write_text(
            "".join(json.dumps(source_row(0))+"\n" for _ in range(2)))
        with self.assertRaisesRegex(ValueError, "duplicate benchmark record ID"):
            make_manifest(self.source, limit=1,
                          min_context_bytes=100, max_prompt_bytes=4000)

    def test_reject_incorrect_gold_and_missing_choices(self):
        bad = source_row(0)
        bad["answer"] = "E"
        self.source.write_text(json.dumps(bad)+"\n")
        with self.assertRaisesRegex(ValueError, "invalid MCQ answer"):
            make_manifest(self.source, limit=1,
                          min_context_bytes=100, max_prompt_bytes=4000)
        del bad["answer"]
        bad["choice_C"] = None
        bad["answer"] = "A"
        self.source.write_text(json.dumps(bad)+"\n")
        with self.assertRaisesRegex(ValueError, "missing/oversized choice"):
            make_manifest(self.source, limit=1,
                          min_context_bytes=100, max_prompt_bytes=4000)

    def test_skip_long_source_unchanged(self):
        original = source_row(0, context_size=1200)
        other = source_row(1, context_size=1200)
        other["context"] = "unneeded content " * 10000
        self.source.write_text(
            json.dumps(original)+"\n"+json.dumps(other)+"\n")
        result = make_manifest(self.source, limit=1,
                               min_context_bytes=100, max_prompt_bytes=4000)
        self.assertEqual(result["tasks"][0]["benchmark_id"], original["_id"])

    def test_do_not_silently_truncate_to_meet_size(self):
        with self.assertRaisesRegex(ValueError, "only .* eligible"):
            make_manifest(self.source, limit=2,
                          min_context_bytes=100, max_prompt_bytes=200)

    def test_scoring_accepts_unambiguous_choice_forms(self):
        for output in ("B", "(B)", "B.", "Answer: B",
                       "The correct answer is (B)", "option b"):
            self.assertEqual(grade({"id":"test", "expected":{
                "type": "choice", "value": "B"}}, output), (True, "correct"))
        for output in ("C", "The correct answer is (C)",
                       "B or C", "I think B, although C is possible", ""):
            self.assertFalse(grade({"id":"test", "expected":{
                "type": "choice", "value": "B"}}, output)[0])


if __name__ == "__main__":
    unittest.main()
