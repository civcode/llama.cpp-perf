#!/usr/bin/env python3
"""CPU-only synthetic regression tests for exact-input QSA FP64 headwise study."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from analyze_qsa_headwise_precision import (
    _identical_capture_inputs, analyze, summarize,
)


class HeadwisePrecisionTests(unittest.TestCase):
    def setUp(self):
        self.ref = np.ones((1, 4, 24, 256), dtype=np.float64)
        self.a = (self.ref + .1).astype(np.float32)
        self.b = (self.ref + .05).astype(np.float32)

    def test_sparse_more_accurate_all_heads(self):
        result = summarize(self.ref, self.a, self.b)
        self.assertEqual(result["heads_better_sparse"], 24)
        self.assertEqual(result["heads_better_dense"], 0)
        self.assertEqual(result["query_head_pairs_better_sparse"], 96)
        self.assertEqual(result["query_head_pairs_better_dense"], 0)
        self.assertLess(result["total_sparse_nmse"], result["total_dense_nmse"])
        self.assertEqual(len(result["headwise"]), 24)
        self.assertEqual(len(result["query_headwise"]), 96)
        self.assertEqual(result["production_promotion"], "NO_GO")

    def test_dense_more_accurate_one_head(self):
        self.b[:, :, 7, :] = 1.3
        result = summarize(self.ref, self.a, self.b)
        self.assertEqual(result["heads_better_dense"], 1)
        self.assertEqual(result["query_head_pairs_better_dense"], 4)
        self.assertEqual(result["headwise"][7]["winner"], "dense")
        self.assertEqual(result["heads_better_sparse"], 23)

    def test_per_query_winner_not_conflated_with_head_winner(self):
        self.b[:, 0, 8, :] = 1.5
        self.b[:, 1:, 8, :] = 1.0
        result = summarize(self.ref, self.a, self.b)
        self.assertEqual(result["query_headwise"][8]["winner"], "dense")
        self.assertEqual(result["query_headwise"][32+8]["winner"], "sparse")
        self.assertEqual(result["headwise"][8]["winner"], "dense")

    def test_identical_paths_tie(self):
        result = summarize(self.ref, self.a, self.a.copy())
        self.assertEqual(result["heads_tied"], 24)
        self.assertEqual(result["query_head_pairs_tied"], 96)
        self.assertEqual(result["total_dense_squared_error"], result["total_sparse_squared_error"])

    def test_wrong_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected identical QSA"):
            summarize(self.ref[:, :2], self.a[:, :2], self.b[:, :2])

    def test_nonfinite_rejected(self):
        self.b[0,0,0,0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite floating"):
            summarize(self.ref, self.a, self.b)

    def test_non_float_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite floating"):
            summarize(self.ref, self.a.astype(np.int32), self.b)

    def test_zero_reference_guard_is_finite(self):
        z = np.zeros_like(self.ref)
        r = summarize(z, z, z)
        self.assertTrue(np.isfinite(r["total_dense_nmse"]))
        self.assertEqual(r["total_sparse_nmse"], 0)
        self.assertEqual(r["heads_tied"], 24)


class CaptureContractTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.logs = [self.root / f"{name}.log" for name in ("dense-a","dense-b","sparse")]
        self.captures = []
        for i, log in enumerate(self.logs):
            log.write_text("synthetic model data outside Git\n")
            inputs = {}
            for name in "qkv":
                path = self.root / f"{i}-{name}.bin"
                path.write_bytes((name * 8).encode("ascii"))
                inputs[name] = {
                    "path": str(path), "type": "f16", "ne": "1,1,1,1",
                    "nb": "2,2,2,2",
                }
            self.captures.append({
                "log": str(log), "meta": {"tensor": "qsa_flash_attn_il11"},
                "scale": .0625, "audit": {
                    "kv": 32768, "bound": 2051, "qheads": 24, "kvheads": 2,
                    "union": 3200,
                }, "inputs": inputs,
            })

    def _mock_read_tensor(self, record, log):
        return np.zeros((1, 4, 24, 256))

    def _analyse(self, outputs=None, masks=None):
        if outputs is None:
            outputs = [np.ones((1,4,24,256), dtype=np.float32) for _ in range(3)]
        if masks is None:
            masks = [np.zeros((4,128), dtype=np.float16) for _ in range(3)]
        with (
            patch("analyze_qsa_headwise_precision.read_capture",
                  side_effect=self.captures),
            patch("analyze_qsa_headwise_precision.read_mask", side_effect=masks),
            patch("analyze_qsa_headwise_precision.read_tensor",
                  side_effect=self._mock_read_tensor),
            patch("analyze_qsa_headwise_precision.read_output",
                  side_effect=outputs),
            patch("analyze_qsa_headwise_precision.cpu_reference",
                  return_value=np.ones((1,4,24,256), dtype=np.float64)),
        ):
            return analyze(*self.logs)

    def test_triple_identical_capture_passes(self):
        result = self._analyse()
        self.assertTrue(result["raw_qkv_mask_identical_across_three"])
        self.assertTrue(result["dense_a_b_attention_bitwise_identical"])
        self.assertEqual(result["layer"], "qsa_flash_attn_il11")
        self.assertEqual(result["heads_tied"], 24)

    def test_qkv_raw_bytes_different_rejected(self):
        Path(self.captures[2]["inputs"]["k"]["path"]).write_bytes(b"different")
        with self.assertRaisesRegex(ValueError, "QKV k raw span bytes differ"):
            self._analyse()

    def test_mask_signed_zero_bit_mismatch_rejected(self):
        masks = [np.zeros((4,128),dtype=np.float16) for _ in range(3)]
        masks[2][0,0] = np.float16(-0.0)
        with self.assertRaisesRegex(ValueError, "F16 mask bytes differ"):
            self._analyse(masks=masks)

    def test_dense_signed_zero_bit_difference_rejected(self):
        outputs = [np.zeros((1,4,24,256), dtype=np.float32) for _ in range(3)]
        outputs[1][0,0,0,0] = np.float32(-0.0)
        with self.assertRaisesRegex(ValueError, "dense A/B F32 outputs"):
            self._analyse(outputs=outputs)

    def test_layer_identity_mismatch_rejected(self):
        self.captures[2]["meta"]["tensor"] = "qsa_flash_attn_il15"
        with self.assertRaisesRegex(ValueError, "layer identity differs"):
            self._analyse()

    def test_duplicate_log_path_rejected(self):
        with self.assertRaisesRegex(ValueError, "three independent capture"):
            analyze(self.logs[0], self.logs[0], self.logs[2])

    def test_raw_metadata_mismatch_rejected(self):
        self.captures[2]["inputs"]["v"]["ne"] = "2,1,1,1"
        with self.assertRaisesRegex(ValueError, "QKV metadata mismatch"):
            self._analyse()

    def test_mask_union_mismatch_rejected(self):
        self.captures[2]["audit"]["union"] = 3199
        with self.assertRaisesRegex(ValueError, "mask union count"):
            self._analyse()


if __name__ == "__main__":
    unittest.main()
