#!/usr/bin/env python3
"""Host-only regressions for QSA F32 post-kernel output comparison."""
from array import array
from pathlib import Path
import sys
import tempfile
import unittest

from compare_qsa_attn_out import compare_logs


def make_log(root: Path, name: str, values: list[float],
             union: int = 2, layer: str = "qsa_flash_attn_il5") -> Path:
    # [feature=2, head=4, query=4, stream=1]; dim-0 varies fastest.
    assert len(values) == 32
    a = array("f", values)
    if sys.byteorder != "little":
        a.byteswap()
    file = root / f"{name}.f32"
    file.write_bytes(a.tobytes())
    log = root / f"{name}.log"
    log.write_text(
        "QSA_MASK_AUDIT mono_ns=1 backend=HIP id=0 device=0 tensor=" +
        layer + " kv=8 q=4 qheads=4 kvheads=1 stream=0 bound=2 "
        f"counts=1,1,1,1 union={union} capacity=8 nan=0 posinf=0 neginf=28\n" +
        "QSA_ATTN_OUT backend=HIP device=0 id=0 tensor=" + layer +
        " ne=2,4,4,1 path=" + str(file) + "\n",
        encoding="utf-8",
    )
    return log


class CompareOutputsTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.values = [0.125 * (i + 1) for i in range(32)]

    def test_identical_outputs(self):
        a = make_log(self.root, "dense", self.values)
        b = make_log(self.root, "repeated", self.values)
        r = compare_logs(a, b)
        self.assertEqual(r["ops_compared"], 1)
        self.assertEqual(r["ops_changed"], 0)

    def test_signed_zero_is_binary_different(self):
        values = self.values.copy()
        values[0] = 0.0
        a = make_log(self.root, "dense", values)
        values[0] = -0.0
        b = make_log(self.root, "sparse", values)
        r = compare_logs(a, b)
        self.assertEqual(r["ops_changed"], 0)
        self.assertEqual(r["ops_binary_different"], 1)

    def test_single_changed_head_feature(self):
        a = make_log(self.root, "dense", self.values)
        changed = self.values.copy()
        changed[3] += 0.5  # feature=1, head=1, query=0
        b = make_log(self.root, "sparse", changed)
        r = compare_logs(a, b)
        self.assertEqual(r["ops_changed"], 1)
        item = r["first_changed_per_device"][0]
        self.assertEqual(item["layer"], "qsa_flash_attn_il5")
        self.assertEqual(item["first_difference"]["head"], 1)
        self.assertEqual(item["first_difference"]["feature"], 1)
        self.assertAlmostEqual(item["max_abs"], 0.5)
        self.assertGreater(item["worst_heads"][0]["nmse_over_dense_energy"], 0)

    def test_tolerance(self):
        a = make_log(self.root, "dense", self.values)
        changed = self.values.copy()
        changed[0] += 1e-5
        b = make_log(self.root, "sparse", changed)
        self.assertEqual(compare_logs(a, b)["ops_changed"], 1)
        self.assertEqual(compare_logs(a, b, abs_tol=1e-4)["ops_changed"], 0)

    def test_different_union_does_not_block_output_comparison(self):
        a = make_log(self.root, "dense", self.values, union=2)
        b = make_log(self.root, "sparse", self.values, union=3)
        self.assertEqual(compare_logs(a, b)["ops_changed"], 0)

    def test_reject_mismatched_layer_label(self):
        a = make_log(self.root, "dense", self.values)
        b = make_log(self.root, "sparse", self.values, layer="qsa_flash_attn_il6")
        with self.assertRaises(ValueError):
            compare_logs(a, b)

    def test_reject_nonfinite_output(self):
        a = make_log(self.root, "dense", self.values)
        changed = self.values.copy()
        changed[0] = float("nan")
        b = make_log(self.root, "sparse", changed)
        with self.assertRaises(ValueError):
            compare_logs(a, b)


if __name__ == "__main__":
    unittest.main()
