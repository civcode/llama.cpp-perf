#!/usr/bin/env python3
"""Host-only regression tests for compare_qsa_mask_dumps.py (stdlib only)."""
from array import array
from pathlib import Path
import sys
import tempfile
import unittest

from compare_qsa_mask_dumps import compare_logs


def write_case(root: Path, name: str, rows: list[dict[int, int]]) -> Path:
    """Create one raw [4,8] mask and one matching audit/dump transcript."""
    assert len(rows) == 4
    kv = 8
    values = array("H", [0xfc00] * (4 * kv))
    for q, selected in enumerate(rows):
        for pos, half in selected.items():
            values[q * kv + pos] = half
    if sys.byteorder != "little":
        values.byteswap()
    sample = root / f"{name}-mask.f16"
    sample.write_bytes(values.tobytes())
    counts = [len(row) for row in rows]
    union = len(set().union(*(set(row) for row in rows)))
    log = root / f"{name}.log"
    log.write_text(
        "QSA_MASK_DUMP backend=HIP device=0 id=0 stream=0 kv=8 path=" + str(sample) + "\n" +
        "QSA_MASK_AUDIT mono_ns=1 backend=HIP id=0 device=0 tensor=out kv=8 "
        "q=4 qheads=24 kvheads=2 stream=0 bound=2 counts=" +
        ",".join(map(str, counts)) + f" union={union} capacity=8 nan=0 posinf=0 neginf={32-sum(counts)}\n",
        encoding="utf-8",
    )
    return log


class CompareMasksTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base = [{0: 0x0000, 1: 0x3c00}, {2: 0}, {1: 0}, {2: 0}]

    def test_identical(self):
        a = write_case(self.root, "a", self.base)
        b = write_case(self.root, "b", self.base)
        result = compare_logs(a, b)
        self.assertEqual(result["groups_audited"], 1)
        self.assertEqual(result["groups_different"], 0)
        self.assertEqual(result["union_count_differences"], 0)

    def test_membership_difference_with_equal_query_counts(self):
        a = write_case(self.root, "a", self.base)
        changed = [dict(row) for row in self.base]
        changed[3] = {3: 0}
        b = write_case(self.root, "b", changed)
        result = compare_logs(a, b)
        self.assertEqual(result["groups_different"], 1)
        self.assertEqual(result["union_count_differences"], 1)
        first = result["first_difference_per_device"][0]
        self.assertEqual(first["dense_union"], 3)
        self.assertEqual(first["sparse_union"], 4)
        self.assertEqual(first["union_added_first"], [3])
        self.assertEqual(first["rows"][3]["removed_first"], [2])

    def test_score_difference_without_membership_difference(self):
        a = write_case(self.root, "a", self.base)
        changed = [dict(row) for row in self.base]
        changed[0][0] = 0x3c00
        b = write_case(self.root, "b", changed)
        result = compare_logs(a, b)
        self.assertEqual(result["groups_different"], 1)
        self.assertEqual(result["union_count_differences"], 0)
        first = result["first_difference_per_device"][0]
        self.assertEqual(first["selected_mask_value_changes"], 1)
        self.assertEqual(first["rows"][0]["common_selected_mask_value_examples"], [
            {"position": 0, "dense_bits": "0x0000", "sparse_bits": "0x3c00",
             "dense_value": 0.0, "sparse_value": 1.0}])

    def test_equal_union_different_membership(self):
        a = write_case(self.root, "a", self.base)
        changed = [dict(row) for row in self.base]
        changed[3] = {1: 0}
        b = write_case(self.root, "b", changed)
        result = compare_logs(a, b)
        self.assertEqual(result["groups_different"], 1)
        self.assertEqual(result["union_count_differences"], 0)
        self.assertEqual(result["first_difference_per_device"][0]["rows"][3]["added_first"], [1])

    def test_reject_nonnegative_infinity(self):
        a = write_case(self.root, "a", self.base)
        changed = [dict(row) for row in self.base]
        changed[0][0] = 0x7c00
        b = write_case(self.root, "b", changed)
        with self.assertRaises(ValueError):
            compare_logs(a, b)

    def test_reject_nan(self):
        a = write_case(self.root, "a", self.base)
        changed = [dict(row) for row in self.base]
        changed[0][0] = 0x7e00
        b = write_case(self.root, "b", changed)
        with self.assertRaises(ValueError):
            compare_logs(a, b)


if __name__ == "__main__":
    unittest.main()
