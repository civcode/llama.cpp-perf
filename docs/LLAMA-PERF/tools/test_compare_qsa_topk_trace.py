#!/usr/bin/env python3
"""Host-only synthetic regressions for layer-15 QSA score/top-k analysis."""
from array import array
from pathlib import Path
import sys
import tempfile
import unittest

from compare_qsa_topk_trace import compare, read_log


def write_run(root: Path, name: str, *, changed=False, fused=True,
              bad_score=False, mismatched_map=False) -> Path:
    folder = root / name
    folder.mkdir()
    score = [0.9, 0.8, 0.79, 0.1, 0.2, 0.6, 0.7, 0.4]
    picks = [0, 1, 2, 1]
    if changed:
        score[1], score[2] = 0.78, 0.81
        picks[:2] = [0, 2]
    if bad_score:
        score[0] = float("nan")
    blocks = [0, 1, 4, 5, 8, 9, 12, 13]
    if mismatched_map:
        blocks[2] = 7
    extra = [18, 19]
    kind = "mask" if fused else "topk"
    records = [
        ("score", "f32", (4, 2, 1, 1), score),
        ("topk", "i32", (2, 2, 1, 1), picks),
    ]
    if fused:
        records.extend([
            ("blocks", "i32", (2, 4, 1, 1), blocks),
            ("extra", "i32", (1, 2, 1, 1), extra),
        ])
    lines = [f"QSA_TOPK_GROUP backend=HIP device=0 layer=15 kind={kind} "
             f"group=0 kv={32 if fused else -1} n_pool=4 n_top=2 n_q=2"]
    for part, dtype, ne, data in records:
        values = array("f" if dtype == "f32" else "i", data)
        if sys.byteorder != "little":
            values.byteswap()
        file = folder / f"{part}.bin"
        file.write_bytes(values.tobytes())
        nb = [4]
        for n in ne[:-1]:
            nb.append(nb[-1]*n)
        lines.append(
            f"QSA_TOPK_PART backend=HIP device=0 layer=15 kind={kind} group=0 "
            f"name={part} type={dtype} ne={','.join(map(str, ne))} "
            f"nb={','.join(map(str, nb))} bytes={len(values)*4} path={file}"
        )
    log = folder / "server.log"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return log


class ScoreTopkCompareTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_identical_dense_replicates(self):
        dense_a = write_run(self.root, "a")
        dense_b = write_run(self.root, "b")
        result = compare(dense_a, dense_b)
        self.assertEqual(result["groups_compared"], 1)
        self.assertEqual(result["queries_compared"], 2)
        self.assertEqual(result["queries_with_changed_pool_selections"], 0)
        self.assertIsNone(result["first_changed_selection"])

    def test_changed_pool_selection_and_kv_mapping(self):
        dense = write_run(self.root, "dense")
        sparse = write_run(self.root, "sparse", changed=True)
        result = compare(dense, sparse)
        self.assertEqual(result["queries_with_changed_pool_selections"], 1)
        first = result["first_changed_selection"]
        self.assertEqual(first["query"], 0)
        self.assertEqual(first["pools_removed_first"], [1])
        self.assertEqual(first["pools_added_first"], [2])
        self.assertEqual(first["kv_cells_removed_first"], [4, 5])
        self.assertEqual(first["kv_cells_added_first"], [8, 9])
        self.assertAlmostEqual(first["dense_cutoff_margin"], 0.01, places=6)
        self.assertAlmostEqual(first["other_cutoff_margin"], 0.03, places=6)

    def test_nonfused_topk_works_without_pool_mapping(self):
        dense = write_run(self.root, "dense", fused=False)
        sparse = write_run(self.root, "sparse", changed=True, fused=False)
        result = compare(dense, sparse)
        first = result["first_changed_selection"]
        self.assertEqual(first["kind"], "topk")
        self.assertIsNone(first["kv_cells_removed_first"])

    def test_reject_nonfinite_score(self):
        dense = write_run(self.root, "dense")
        sparse = write_run(self.root, "sparse", bad_score=True)
        with self.assertRaisesRegex(ValueError, "NaN"):
            compare(dense, sparse)

    def test_pool_mapping_is_recorded_but_no_false_selection_difference(self):
        dense = write_run(self.root, "dense")
        sparse = write_run(self.root, "sparse", mismatched_map=True)
        result = compare(dense, sparse)
        self.assertEqual(result["queries_with_changed_pool_selections"], 0)
        self.assertEqual(result["queries_with_changed_kv_selections"], 1)
        self.assertEqual(result["groups_with_changed_selections"], 1)
        first = result["first_changed_selection"]
        self.assertEqual(first["kv_cells_removed_first"], [4])
        self.assertEqual(first["kv_cells_added_first"], [7])

    def test_bad_or_missing_part_detected(self):
        path = write_run(self.root, "a")
        txt = path.read_text()
        path.write_text(txt.replace("name=topk", "name=score"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            read_log(path)


if __name__ == "__main__":
    unittest.main()
