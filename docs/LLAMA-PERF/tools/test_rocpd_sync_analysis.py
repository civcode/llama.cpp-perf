#!/usr/bin/env python3
"""Targeted SQLite fixture tests; no ROCm installation or real trace required."""

import sqlite3
import unittest

from rocpd_sync_analysis import analyze, union_duration


class AnalysisTests(unittest.TestCase):
    def test_interval_union(self):
        self.assertEqual(union_duration([(1, 4), (3, 7), (9, 11)]), 8)
        self.assertEqual(union_duration([]), 0)

    def test_analysis_on_minimal_rocpd_views(self):
        db = sqlite3.connect(":memory:")
        db.executescript("""
            CREATE TABLE kernels (id INTEGER PRIMARY KEY, start INTEGER, end INTEGER,
                duration INTEGER, agent_abs_index INTEGER, name TEXT);
            CREATE TABLE regions_and_samples (id INTEGER PRIMARY KEY, tid INTEGER,
                start INTEGER, end INTEGER, category TEXT, name TEXT);
            CREATE TABLE rocpd_memory_copy (src_agent_id INTEGER, dst_agent_id INTEGER,
                size INTEGER, start INTEGER, end INTEGER);
            CREATE TABLE rocpd_info_agent (id INTEGER, name TEXT, type TEXT, absolute_index INTEGER);
            INSERT INTO rocpd_info_agent VALUES (0, 'Ryzen', 'CPU', 0), (1, 'gfx1201', 'GPU', 1);
            INSERT INTO kernels VALUES (1, 0, 10000000, 10000000, 1, 'kernelA');
            INSERT INTO kernels VALUES (2, 6200000000, 6500000000, 300000000, 1, 'kernelB');
            INSERT INTO kernels VALUES (3, 7000000000, 10000000000, 3000000000, 1, 'kernelC');
            INSERT INTO regions_and_samples VALUES (1, 42, 6100000000, 6150000000, 'HIP_RUNTIME_API_EXT', 'hipGraphLaunch');
            INSERT INTO regions_and_samples VALUES (2, 42, 6200000000, 6800000000, 'HIP_RUNTIME_API_EXT', 'hipStreamSynchronize');
            INSERT INTO regions_and_samples VALUES (3, 42, 6900000000, 6950000000, 'HIP_RUNTIME_API_EXT', 'hipMemcpyAsync');
            INSERT INTO regions_and_samples VALUES (4, 42, 7100000000, 7400000000, 'HIP_RUNTIME_API_EXT', 'hipStreamSynchronize');
            INSERT INTO regions_and_samples VALUES (5, 99, 7200000000, 7500000000, 'HIP_RUNTIME_API_EXT', 'hipStreamSynchronize');
            INSERT INTO rocpd_memory_copy VALUES (1, 0, 993280, 7300000000, 7400000000);
        """)
        result = analyze(db, 6.0)
        self.assertEqual(result["sync"]["calls"], 3)
        self.assertEqual(result["sync"]["categories"], {"HIP_RUNTIME_API_EXT": 3})
        self.assertAlmostEqual(result["sync"]["sum_ms"], 1200)
        self.assertAlmostEqual(result["sync"]["union_ms"], 1000)
        self.assertEqual(result["sync"]["threads"][0]["tid"], 42)
        self.assertEqual(result["memory_copies"][0]["calls"], 1)
        self.assertTrue(result["neighbor_patterns"][0]["patterns"])
        db.close()

    def test_empty_window_fails(self):
        db = sqlite3.connect(":memory:")
        with self.assertRaises(ValueError):
            analyze(db)
        db.close()


if __name__ == "__main__":
    unittest.main()
