#!/usr/bin/env python3
"""Host-only synthetic tests for fixed-reference QSA NLL comparisons."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from compare_qsa_reference_nll import compare, read_labels, row_nll


def write_fixture(root: Path, name: str, *, logits=(1.0, 2.0, 0.5),
                  target_ids=(1, 2), position=10, label=1, origin="heldout_reference",
                  truncate_labels=False, bad_dtype=False):
    folder = root / name
    folder.mkdir()
    (folder / "logits.f32").write_bytes(
        b"".join(struct.pack("<f", x) for x in logits))
    event = {"kind": "event", "step": 0, "phase": "verify_logits",
             "seq": 0, "position": position, "target_ids": list(target_ids),
             "draft_ids": [1], "accepted": [1], "rollback": False, "replay": False}
    tensor = {"kind": "tensor", "step": 0, "phase": "verify_logits",
              "seq": 0, "position": position, "layer": -1,
              "name": "logits", "dtype": "f16" if bad_dtype else "f32",
              "shape": [len(logits)], "path": "logits.f32"}
    manifest = folder / "manifest.jsonl"
    manifest.write_text(json.dumps(event)+"\n"+json.dumps(tensor)+"\n")
    reference = folder / "references.jsonl"
    if not truncate_labels:
        reference.write_text(json.dumps(
            {"step": 0, "phase": "verify_logits", "seq": 0,
             "position": position, "label_token": label, "origin": origin})+"\n")
    else:
        reference.write_text("")
    return manifest, reference


class ReferenceNllTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_identical_logits_zero_delta(self):
        a, labels = write_fixture(self.root, "a")
        b, _ = write_fixture(self.root, "b")
        r = compare(a, b, labels)
        self.assertEqual(r["rows"], 1)
        self.assertEqual(r["mean_sparse_minus_dense_nll_nats"], 0.0)
        self.assertEqual(r["dense_reference_top1_hits"], 1)
        self.assertEqual(r["sparse_reference_top1_hits"], 1)
        self.assertEqual(r["top1_choice_changed_rows"], 0)

    def test_sparse_worse_on_gold_and_top1_flip(self):
        a, labels = write_fixture(self.root, "a")
        b, _ = write_fixture(self.root, "b", logits=(3.0, 1.0, 0.5))
        r = compare(a, b, labels)
        self.assertEqual(r["sparse_rows_worse_nll"], 1)
        self.assertEqual(r["sparse_reference_top1_hits"], 0)
        self.assertEqual(r["top1_choice_changed_rows"], 1)
        self.assertGreater(r["mean_sparse_minus_dense_nll_nats"], 0)
        self.assertIsNotNone(r["first_top1_change"])

    def test_sparse_better_on_gold(self):
        a, labels = write_fixture(self.root, "a")
        b, _ = write_fixture(self.root, "b", logits=(0.0, 4.0, 0.0))
        r = compare(a, b, labels)
        self.assertEqual(r["sparse_rows_better_nll"], 1)
        self.assertLess(r["mean_sparse_minus_dense_nll_nats"], 0)

    def test_softmax_constant_shift_invariant(self):
        x = row_nll([0.0, 1.0, -2.0], 1)
        y = row_nll([10.0, 11.0, 8.0], 1)
        self.assertAlmostEqual(x["nll_nats"], y["nll_nats"])
        self.assertAlmostEqual(x["reference_probability"], y["reference_probability"])

    def test_target_label_not_in_vocab(self):
        a, labels = write_fixture(self.root, "a", label=7)
        b, _ = write_fixture(self.root, "b")
        with self.assertRaisesRegex(ValueError, "outside logit vocabulary"):
            compare(a, b, labels)

    def test_reject_different_token_histories(self):
        a, labels = write_fixture(self.root, "a")
        b, _ = write_fixture(self.root, "b", target_ids=(9, 2))
        with self.assertRaisesRegex(ValueError, "histories differ"):
            compare(a, b, labels)

    def test_reject_reference_label_checkpoint_mismatch(self):
        a, labels = write_fixture(self.root, "a")
        b, _ = write_fixture(self.root, "b")
        records = [json.loads(line) for line in labels.read_text().splitlines()]
        records[0]["position"] = 99
        labels.write_text("".join(json.dumps(x)+"\n" for x in records))
        with self.assertRaisesRegex(ValueError, "checkpoint exactly"):
            compare(a, b, labels)

    def test_reject_sampled_label_origin(self):
        a, labels = write_fixture(self.root, "a", origin="sampled")
        with self.assertRaisesRegex(ValueError, "origin must be explicit"):
            read_labels(labels, 64)

    def test_reject_nonfinite_logits(self):
        with self.assertRaisesRegex(ValueError, "nonfinite logits"):
            row_nll([1.0, float("nan")], 0)

    def test_tied_top1_has_deterministic_tiebreak(self):
        a = row_nll([3.0, 3.0, -1.0], 1)
        self.assertEqual(a["top1"], 0)
        self.assertEqual(a["reference_token_rank"], 2)

    def test_reject_incorrect_dtype(self):
        a, labels = write_fixture(self.root, "a", bad_dtype=True)
        b, _ = write_fixture(self.root, "b")
        with self.assertRaisesRegex(ValueError, "full-vocabulary F32"):
            compare(a, b, labels)


if __name__ == "__main__":
    unittest.main()
