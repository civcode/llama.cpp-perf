"""Synthetic clock/domain and ROCpd kernel-overlap test (no GPU required)."""
import sqlite3
import tempfile
from pathlib import Path
import unittest

from rocpd_step_alignment import (
    analyze, busy_union_ns, clipped_for_window, prefixed_maximum_end,
)


class TestROCPDStepAlignment(unittest.TestCase):
    def test_overlaps_and_unions(self):
        self.assertEqual(busy_union_ns([(1, 5), (3, 10), (12, 13)]), 10)
        self.assertEqual(busy_union_ns([]), 0)
        kernels = [(0, 5, "0", "a"), (4, 9, "0", "b"), (11, 20, "1", "c")]
        got = clipped_for_window(kernels, [r[0] for r in kernels],
                                 prefixed_maximum_end(kernels), 6, 13)
        self.assertEqual(got, [(6, 9, "0", "b"), (11, 13, "1", "c")])

    def test_synthetic_aligned_trace(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            tsv = root / "trace.tsv"
            dbpath = root / "sample.db"
            with tsv.open("w", encoding="utf-8") as f:
                f.write(
                    "mono_ns\tos_tid\tphase\tslot_id\tverification_steps_before"
                    "\tgeneration_tokens_before\tbatch_tokens\tdraft_tokens\traw_ns\n"
                )
                for i in range(6):
                    start = 2_000_000_000 + i * 20_000_000
                    end = start + 10_000_000
                    for timestamp, phase in ((start, "target_sync_begin"),
                                             (end, "target_sync_end")):
                        f.write(f"{timestamp}\t123\t{phase}\t0\t{i}\t{i*2}"
                                f"\t4\t3\t{timestamp+1_000_000_000}\n")
            with sqlite3.connect(dbpath) as db:
                db.executescript("""
                    CREATE TABLE kernels (
                        agent_abs_index INTEGER, start INTEGER, end INTEGER, name TEXT);
                    CREATE TABLE regions_and_samples (
                        name TEXT, category TEXT, start INTEGER, end INTEGER);
                """)
                for i in range(6):
                    start = 2_000_000_000 + i * 20_000_000
                    db.execute("INSERT INTO kernels VALUES(?,?,?,?)",
                               (1, start+1_000_000, start+5_000_000,
                                "flash_attn_tile"))
                    db.execute("INSERT INTO kernels VALUES(?,?,?,?)",
                               (2, start+2_000_000, start+7_000_000,
                                "mul_mat_vec_q"))
                    db.execute("INSERT INTO regions_and_samples VALUES(?,?,?,?)",
                               ("hipStreamSynchronize", "HIP_RUNTIME_API_EXT",
                                start, start+1_000_000))
                db.commit()
            output = root / "windows.csv"
            report = analyze(tsv, dbpath, "auto", output)
            self.assertEqual(report["clock_used"], "mono")
            self.assertEqual(report["host_sync_windows"], 6)
            self.assertEqual(report["unpaired_host_events"], 0)
            self.assertAlmostEqual(report["median_host_sync_ms"], 10)
            self.assertAlmostEqual(report["per_gpu_amd_kernel_overlap"]["1"]
                                   ["median_amd_busy_overlap_ms"], 4)
            self.assertAlmostEqual(report["per_gpu_amd_kernel_overlap"]["2"]
                                   ["median_amd_busy_overlap_ms"], 5)
            self.assertAlmostEqual(report["per_gpu_amd_kernel_overlap"]["1"]
                                   ["median_flash_overlap_ms"], 4)
            self.assertEqual(len(report["steps"]), 6)
            self.assertEqual(report["top_hip_host_apis"][0]["count"], 6)
            self.assertTrue(output.is_file())
            with self.assertRaisesRegex(ValueError, "do not overlap enough"):
                analyze(tsv, dbpath, "raw", None)


if __name__ == "__main__":
    unittest.main()
