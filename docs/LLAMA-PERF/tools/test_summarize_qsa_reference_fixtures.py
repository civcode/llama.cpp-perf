#!/usr/bin/env python3
"""Synthetic host tests for independent-document QSA reference summaries."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from summarize_qsa_reference_fixtures import summarize


def create_fixture(root: Path, fid: str, *,
                   sparse_gold_logit=3.0, source_text=None,
                   dense_b_raw_signed_zero=False, wrong_sparse_label=False):
    folder = root / fid
    folder.mkdir()
    source = folder / "source.txt"
    source.write_text(source_text if source_text is not None else "Independent text for " + fid,
                      encoding="utf-8")
    for run in ("dense_a", "dense_b", "sparse"):
        dest = folder / run
        dest.mkdir()
        events = []
        labels = []
        for step in range(4):
            position = 4096 + step
            event = {
                "kind": "event", "step": step, "phase": "teacher_forced_logits",
                "seq": 0, "position": position, "target_ids": [5+step],
                "draft_ids": [], "accepted": [], "rollback": False,
                "replay": False, "prefix_token_hash64": str(step),
                "prefix_token_count": position+1,
                "query_group_first_pos": 4096,
            }
            tensor = {
                "kind": "tensor", "step": step, "phase": "teacher_forced_logits",
                "seq": 0, "position": position, "layer": -1,
                "name": "logits", "dtype": "f32", "shape": [3],
                "path": f"logits-{step}.f32",
            }
            events.extend((event, tensor))
            label_token = 1
            if wrong_sparse_label and run == "sparse" and step == 2:
                label_token = 0
            labels.append({
                "step": step, "phase": "teacher_forced_logits",
                "seq": 0, "position": position, "label_token": label_token,
                "origin": "teacher_forced_reference",
            })
            logits = [0.0, sparse_gold_logit if run == "sparse" else 2.0, -1.0]
            if dense_b_raw_signed_zero and run == "dense_b" and step == 0:
                logits[0] = -0.0
            (dest / f"logits-{step}.f32").write_bytes(
                b"".join(struct.pack("<f", value) for value in logits))
        (dest / "manifest.jsonl").write_text(
            "".join(json.dumps(v)+"\n" for v in events), encoding="utf-8")
        (dest / "references.jsonl").write_text(
            "".join(json.dumps(v)+"\n" for v in labels), encoding="utf-8")
    return {
        "id": fid, "origin": "independent_reference_text",
        "text_file": str(source),
        "dense_a": str(folder / "dense_a"),
        "dense_b": str(folder / "dense_b"),
        "sparse": str(folder / "sparse"),
    }


def write_manifest(root: Path, rows: list[dict]) -> Path:
    path = root / "fixtures.json"
    path.write_text(json.dumps({"fixtures": rows}), encoding="utf-8")
    return path


class ReferenceSummaryTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_three_sources_macro_micro_and_direction(self):
        fixtures = [
            create_fixture(self.root, "a", sparse_gold_logit=3.0),
            create_fixture(self.root, "b", sparse_gold_logit=2.0),
            create_fixture(self.root, "c", sparse_gold_logit=1.0),
        ]
        result = summarize(write_manifest(self.root, fixtures))
        self.assertEqual(result["independent_fixtures"], 3)
        self.assertEqual(result["total_scored_rows"], 12)
        self.assertEqual(result["documents_sparse_better_mean_nll"], 1)
        self.assertEqual(result["documents_sparse_worse_mean_nll"], 1)
        self.assertEqual(result["individual_top1_choice_flips"], 0)
        self.assertAlmostEqual(result["macro_mean_sparse_minus_dense_nll_nats"],
                               result["micro_mean_sparse_minus_dense_nll_nats"], places=12)
        self.assertTrue(all(x["dense_repeat_bitwise_equal"] for x in result["per_document"]))

    def test_signed_zero_dense_repeat_not_bitwise_equal(self):
        fixture = create_fixture(self.root, "a", dense_b_raw_signed_zero=True)
        with self.assertRaisesRegex(ValueError, "raw logits differ"):
            summarize(write_manifest(self.root, [fixture]))

    def test_source_text_dedup_by_hash_not_filename(self):
        a = create_fixture(self.root, "a", source_text="same content")
        b = create_fixture(self.root, "b", source_text="same content")
        with self.assertRaisesRegex(ValueError, "duplicate source text digest"):
            summarize(write_manifest(self.root, [a, b]))

    def test_reference_label_disagreement_fails_closed(self):
        a = create_fixture(self.root, "a", wrong_sparse_label=True)
        with self.assertRaisesRegex(ValueError, "reference-token label files differ"):
            summarize(write_manifest(self.root, [a]))

    def test_unknown_provenance_rejected(self):
        a = create_fixture(self.root, "a")
        a["origin"] = "certified_heldout"
        with self.assertRaisesRegex(ValueError, "unrecognized fixture provenance"):
            summarize(write_manifest(self.root, [a]))

    def test_duplicate_id_rejected(self):
        a = create_fixture(self.root, "a")
        with self.assertRaisesRegex(ValueError, "duplicate or invalid fixture id"):
            summarize(write_manifest(self.root, [a, a]))

    def test_q4_full_groups_required(self):
        a = create_fixture(self.root, "a")
        for run in ("dense_a", "dense_b", "sparse"):
            file = Path(a[run]) / "manifest.jsonl"
            records = [json.loads(line) for line in file.read_text().splitlines()]
            file.write_text("".join(json.dumps(x)+"\n" for x in records[:-2]))
            labels = Path(a[run]) / "references.jsonl"
            labels.write_text("".join(labels.read_text().splitlines(keepends=True)[:3]))
        with self.assertRaisesRegex(ValueError, "Q=4 groups"):
            summarize(write_manifest(self.root, [a]))

    def test_missing_reference_text_rejected(self):
        a = create_fixture(self.root, "a")
        Path(a["text_file"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing, empty or oversized"):
            summarize(write_manifest(self.root, [a]))


if __name__ == "__main__":
    unittest.main()
