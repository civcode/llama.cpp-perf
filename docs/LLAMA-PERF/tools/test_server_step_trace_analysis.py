import tempfile
import unittest
from pathlib import Path

from server_step_trace_analysis import (
    analyze, match_intervals, parse_events, quantile, write_windows,
)

HEADER = (
    "mono_ns\tos_tid\tphase\tslot_id\tverification_steps_before"
    "\tgeneration_tokens_before\tbatch_tokens\tdraft_tokens\n"
)


def line(t, phase, step=4, gen=14, draft=3):
    return f"{t}\t1234\t{phase}\t0\t{step}\t{gen}\t4\t{draft}\n"


class TestServerStepTrace(unittest.TestCase):
    def test_phases_and_counter_increment(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "trace.tsv"
            path.write_text(
                HEADER +
                line(1000, "draft_begin", draft=0) +
                line(2000, "draft_end", draft=0) +
                line(3000, "target_process_begin") +
                line(63000, "target_process_end") +
                line(64000, "target_sync_begin") +
                line(5064000, "target_sync_end") +
                line(5065000, "post_decode_begin") +
                line(5565000, "verification_sample_begin") +
                line(5665000, "verification_sample_end") +
                line(5765000, "verification_commit", step=5) +
                line(6265000, "post_decode_end", step=5, draft=0),
                encoding="utf-8",
            )
            events = parse_events(path)
            intervals, missing = match_intervals(events)
            self.assertEqual(len(missing), 0)
            self.assertEqual(len(intervals), 5)
            by_phase = {w["phase"]: w for w in intervals}
            self.assertEqual(by_phase["target_sync"]["duration_ms"], 5.0)
            self.assertEqual(by_phase["post_decode"]["step_before"], 4)
            self.assertEqual(by_phase["post_decode"]["duration_ms"], 1.2)
            self.assertEqual(by_phase["verification_sample"]["duration_ms"], 0.1)
            summary = analyze(intervals, only_speculative=True)
            self.assertEqual(summary["target_sync"]["median_ms"], 5.0)
            self.assertIn("draft", summary)
            out = Path(temp) / "windows.tsv"
            write_windows(out, intervals)
            self.assertIn("target_sync", out.read_text(encoding="utf-8"))

    def test_incomplete_and_quantile(self):
        records = [{"phase":"target_sync_end", "slot_id":0, "mono_ns":1000}]
        intervals, missing = match_intervals(records)
        self.assertEqual(intervals, [])
        self.assertEqual(len(missing), 1)
        self.assertEqual(quantile([1, 2, 3], 0.5), 2)
        self.assertEqual(quantile([1, 2], 0.9), 1.9)
        self.assertIsNone(quantile([], 0.5))

    def test_missing_columns(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.tsv"
            path.write_text("phase\nx\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "headers"):
                parse_events(path)


if __name__ == "__main__":
    unittest.main()
