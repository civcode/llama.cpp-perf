#!/usr/bin/env python3
"""Host-only tests for numerical QSA logit distribution diagnostics."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from compare_qsa_logit_distributions import row_metrics, compare


def make(root: Path, name: str, *, logits=None, token_ids=None):
    directory = root / name
    directory.mkdir()
    if logits is None:
        logits = [2.5, 1.5, 0.0, -1.0]
    if token_ids is None:
        token_ids = [4, 5, 6]
    (directory / "logits.f32").write_bytes(b"".join(struct.pack("<f", x) for x in logits))
    event = {"kind": "event", "step": 0, "phase": "verify_logits", "seq": 0,
             "position": 10, "target_ids": token_ids, "draft_ids": [2],
             "accepted": [2, 3], "rollback": False, "replay": False}
    tensor = {"kind": "tensor", "step": 0, "phase": "verify_logits", "seq": 0,
              "position": 10, "layer": -1, "name": "logits", "dtype": "f32",
              "shape": [len(logits)], "path": "logits.f32"}
    manifest = directory / "manifest.jsonl"
    manifest.write_text(json.dumps(event)+"\n"+json.dumps(tensor)+"\n")
    return manifest


class DistributionsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_identical_outputs_have_zero_divergence(self):
        a = make(self.root, "a")
        b = make(self.root, "b")
        result = compare(a, b)
        self.assertEqual(result["top1_changed_rows"], 0)
        self.assertEqual(result["max_kl_dense_other_nats"], 0.0)
        self.assertEqual(result["max_js_nats"], 0.0)
        self.assertEqual(result["max_centered_logit_abs_difference"], 0.0)

    def test_constant_logit_shift_does_not_change_probabilities(self):
        x = row_metrics([0.0, 1.0, -2.0], [5.0, 6.0, 3.0])
        self.assertFalse(x["top1_changed"])
        self.assertAlmostEqual(x["kl_dense_other_nats"], 0.0, places=13)
        self.assertAlmostEqual(x["js_nats"], 0.0, places=13)
        self.assertAlmostEqual(x["max_abs_logit_difference_after_mean_shift"], 0.0, places=13)

    def test_top1_flip_and_kl_positive(self):
        a = make(self.root, "a", logits=[0.0, 3.0, 2.0, -1.0])
        b = make(self.root, "b", logits=[0.0, 2.0, 3.0, -1.0])
        result = compare(a, b)
        self.assertEqual(result["top1_changed_rows"], 1)
        self.assertEqual(result["first_top1_flip"]["dense_top1"], 1)
        self.assertEqual(result["first_top1_flip"]["other_top1"], 2)
        self.assertGreater(result["max_kl_dense_other_nats"], 0)
        self.assertGreater(result["max_js_nats"], 0)

    def test_reject_token_histories_that_are_not_identical(self):
        a = make(self.root, "a")
        b = make(self.root, "b", token_ids=[7, 8])
        with self.assertRaisesRegex(ValueError, "histories differ"):
            compare(a, b)

    def test_reject_unbounded_row_count(self):
        a = make(self.root, "a")
        b = make(self.root, "b")
        with self.assertRaisesRegex(ValueError, "exceed"):
            compare(a, b, max_rows=0)  # Invalid bound itself fails closed.

    def test_reject_nonfinite_logits(self):
        a = make(self.root, "a")
        b = make(self.root, "b", logits=[2.5, float("nan"), 0.0, -1.0])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare(a, b)


if __name__ == "__main__":
    unittest.main()
