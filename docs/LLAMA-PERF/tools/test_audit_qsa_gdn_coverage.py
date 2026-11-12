#!/usr/bin/env python3
"""Host-only synthetic tests for actual GDN row and rollback coverage audit."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from audit_qsa_gdn_coverage import audit


def write_trace(root: Path, name: str, *,
                physical_cell=0, source_cell=0, plane=0, cache_size=4,
                mode="none", bad_row=False, logical_pos=10,
                omit_s=False, staged_source=-1):
    directory = root / name
    directory.mkdir()
    step = 0
    event = {
        "kind": "event", "step": step, "phase": "post_rollback" if mode != "none" else "pre_verification",
        "seq": 0, "position": 10, "target_ids": [1, 2],
        "draft_ids": [3], "accepted": [1], "rollback": mode != "none",
        "rollback_mode": mode, "replay": False,
    }
    records = [event]
    for name in (["gdn_r"] if omit_s else ["gdn_r", "gdn_s"]):
        file = directory / f"{name}.bin"
        file.write_bytes(struct.pack("<ff", 1.0, 2.0))
        record = {
            "kind": "tensor", "step": step, "phase": event["phase"],
            "seq": 0, "position": 10, "layer": 12, "name": name,
            "dtype": "f32", "shape": [2], "path": file.name,
            "physical_cell": physical_cell, "source_cell": source_cell,
            "staged_source_cell": staged_source, "cache_size": cache_size,
            "logical_source_row": plane*cache_size +
                (source_cell if source_cell >= 0 else physical_cell) +
                (1 if bad_row else 0),
            "rs_plane": plane, "logical_cell_pos": logical_pos,
        }
        records.append(record)
    manifest = directory / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r)+"\n" for r in records))
    return manifest


class GdnCoverageTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_normal_state_not_falsely_called_source_indirection(self):
        r = audit(write_trace(self.root, "normal"))
        self.assertEqual(r["nontrivial_source_indirection_rows"], 0)
        self.assertEqual(r["nonzero_rollback_plane_rows"], 0)

    def test_synthetic_source_cell_indirection_and_pending_snapshot(self):
        r = audit(write_trace(self.root, "source", physical_cell=1,
                              source_cell=3, plane=2, mode="snapshot"))
        self.assertEqual(r["nontrivial_source_indirection_rows"], 2)
        self.assertEqual(r["nonzero_rollback_plane_rows"], 2)
        self.assertEqual(r["post_rollback_snapshot_events"], 1)
        self.assertEqual(r["source_indirection_examples"][0]["logical_source_row"], 11)

    def test_staged_source_is_not_a_serialized_source_row(self):
        r = audit(write_trace(self.root, "staged",
                              physical_cell=1, source_cell=1, staged_source=3))
        self.assertEqual(r["nontrivial_source_indirection_rows"], 0)
        self.assertEqual(r["nontrivial_staged_source_rows"], 2)
        self.assertEqual(r["staged_source_examples"][0]["source_cell"], 1)
        self.assertEqual(r["staged_source_examples"][0]["staged_source_cell"], 3)

    def test_checkpoint_restore_tag_is_distinguished(self):
        r = audit(write_trace(self.root, "restore",
                              physical_cell=1, source_cell=-1, mode="checkpoint_restore"))
        self.assertEqual(r["post_rollback_checkpoint_restore_events"], 1)
        self.assertEqual(r["nontrivial_source_indirection_rows"], 0)

    def test_row_formula_mismatch_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "incorrect logical source row"):
            audit(write_trace(self.root, "bad", bad_row=True))

    def test_rollback_mode_must_match_decision(self):
        manifest = write_trace(self.root, "bad_mode", mode="snapshot")
        records = [json.loads(line) for line in manifest.read_text().splitlines()]
        records[0]["rollback_mode"] = "none"
        manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
        with self.assertRaisesRegex(ValueError, "rollback mode inconsistent"):
            audit(manifest)

    def test_missing_recurrent_s_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "incomplete R/S"):
            audit(write_trace(self.root, "no_s", omit_s=True))


if __name__ == "__main__":
    unittest.main()
