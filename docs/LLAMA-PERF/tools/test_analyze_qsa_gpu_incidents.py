#!/usr/bin/env python3
"""CPU-only tests for historical Xid/PCIe event correlation parser."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from analyze_qsa_gpu_incidents import (
    analyze_lines, analyze_sources, classify, time_of,
)


class IncidentAnalyzerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_classify_realistic_xid79_and_xid154(self):
        self.assertEqual(classify(
            "NVRM: Xid (PCI:0000:08:00): 79, GPU has fallen off the bus."),
            ("nvidia_xid", "79"))
        self.assertEqual(classify(
            "NVRM: Xid (PCI:0000:08:00): 154, GPU recovery action changed."),
            ("nvidia_xid", "154"))
        self.assertEqual(classify("Xid 79: GPU fallen off"), ("nvidia_xid", "79"))

    def test_classify_pcie_and_cuda_not_as_sparse_kernel_failure(self):
        self.assertEqual(classify(
            "pcieport 0000:00:01.0: AER: Corrected error received"),
            ("pcie_aer", None))
        self.assertEqual(classify(
            "pcieport 0000:00:02.0: DPC: containment event"),
            ("pcie_dpc", None))
        self.assertEqual(classify(
            "PCIe link down on device"),
            ("pcie_link_problem", None))
        self.assertEqual(classify(
            "cudaStreamSynchronize: CUDA unspecified launch failure"),
            ("cuda_error", None))
        self.assertIsNone(classify("nvidia GPU enumerated and healthy"))
        self.assertIsNone(classify("flash_attn_ext_f16 launch success"))

    def test_isotime_parsing_with_timezone_and_invalid_input(self):
        t = time_of("2026-11-12T19:20:30+0100 host kernel: NVRM: Xid 79")
        self.assertIsNotNone(t)
        self.assertEqual(t.utcoffset().total_seconds(), 3600)
        self.assertIsNone(time_of("Nov 12 19:20:30 host kernel: Xid 79"))

    def test_sequence_and_timed_precursors(self):
        lines = [
            "2026-11-12T19:20:20+0100 host kernel: pcieport: AER: Corrected error\n",
            "2026-11-12T19:20:25+0100 host process: CUDA unspecified launch failure\n",
            "2026-11-12T19:20:30+0100 host kernel: NVRM: Xid (PCI:0000:01:00): 79\n",
            "2026-11-12T19:20:31+0100 host kernel: NVRM: Xid (PCI:0000:01:00): 154\n",
        ]
        result = analyze_lines(lines, "failure-one")
        self.assertEqual(result["lines_scanned"], 4)
        self.assertEqual(result["xid_code_counts"], {"79": 1, "154": 1})
        self.assertEqual(result["event_type_counts"]["pcie_aer"], 1)
        self.assertEqual(result["event_type_counts"]["cuda_error"], 1)
        incident = result["xid79_incidents"][0]
        self.assertTrue(incident["xid154_after_before_next_xid79"])
        self.assertEqual(
            incident["preceding_relevant_events"][0]["seconds_before_xid79"], 10)
        self.assertEqual(
            incident["preceding_relevant_events"][1]["seconds_before_xid79"], 5)

    def test_no_logs_does_not_certify_no_error(self):
        result = analyze_lines([
            "2026-11-12T20:20:00+0100 kernel: normal startup\n",
            "2026-11-12T20:21:00+0100 kernel: normal idle\n",
        ], "healthy-boot")
        self.assertEqual(result["matched_diagnostic_lines"], 0)
        self.assertIn("cannot certify", result["caveat"])

    def test_two_incidents_and_no_conflation_across_sources(self):
        p1 = self.root / "boot1.log"
        p2 = self.root / "boot2.log"
        good = self.root / "boot3.log"
        p1.write_text(
            "2026-11-12T18:00:01+0100 kernel: NVRM: Xid (PCI:0000:01:00): 79\n"
            "2026-11-12T18:00:02+0100 kernel: NVRM: Xid (PCI:0000:01:00): 154\n")
        p2.write_text(
            "2026-11-12T19:00:01+0100 kernel: NVRM: Xid (PCI:0000:01:00): 79\n"
            "2026-11-12T19:00:02+0100 kernel: NVRM: Xid (PCI:0000:01:00): 154\n")
        good.write_text("2026-11-12T20:20:30+0100 kernel: normal boot\n")
        report = analyze_sources([("first", p1), ("second", p2), ("gen3", good)])
        self.assertEqual(report["total_xid79_in_supplied_logs"], 2)
        self.assertEqual(report["total_xid154_in_supplied_logs"], 2)
        self.assertEqual(len(report["sources"]), 3)
        self.assertEqual(report["sources"][2]["xid79_incidents"], [])

    def test_missing_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing, empty or oversized"):
            analyze_sources([("absent", self.root / "does-not-exist.log")])

    def test_duplicate_labels_and_no_source_rejected(self):
        path = self.root / "log.txt"
        path.write_text("safe log")
        with self.assertRaisesRegex(ValueError, "duplicate source"):
            analyze_sources([("same", path), ("same", path)])
        with self.assertRaisesRegex(ValueError, "1..6"):
            analyze_sources([])

    def test_missing_timestamp_reduces_claims(self):
        result = analyze_lines([
            "pcieport 0000:00:01.0: AER: Corrected error received\n",
            "NVRM: Xid (PCI:0000:08:00): 79\n",
            "NVRM: Xid (PCI:0000:08:00): 154\n",
        ], "syslog")
        self.assertEqual(result["timestamps_available"], 0)
        before = result["xid79_incidents"][0]["preceding_relevant_events"]
        self.assertIsNone(before[0]["seconds_before_xid79"])

    def test_excessively_long_log_line_ignored(self):
        result = analyze_lines(["Xid 79" + "x"*30000], "untrusted-log")
        self.assertEqual(result["matched_diagnostic_lines"], 0)


if __name__ == "__main__":
    unittest.main()
