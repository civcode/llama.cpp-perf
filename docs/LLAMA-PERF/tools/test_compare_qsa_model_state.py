#!/usr/bin/env python3
"""Host-only tests for strict logical-token logits/GDN state comparison."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from compare_qsa_model_state import compare, read_manifest


def encode(dtype, vals):
    if dtype == "f32":
        return b"".join(struct.pack("<f", float(v)) for v in vals)
    if dtype == "f16":
        return b"".join(struct.pack("<e", float(v)) for v in vals)
    if dtype == "bf16":
        return b"".join(
            struct.pack("<H", struct.unpack("<I", struct.pack("<f", float(v)))[0] >> 16)
            for v in vals
        )
    raise ValueError(dtype)


def sample_manifest(root: Path, label: str, *, tweak=None, token_override=None,
                    cell=2, nonfinite=False, duplicate=False):
    folder = root / label
    folder.mkdir()
    event = {
        "kind": "event", "step": 0, "phase": "after_prompt",
        "seq": 0, "position": 33024, "target_ids": [71, 72],
        "draft_ids": [], "accepted": [], "rollback": False,
    }
    if token_override:
        event["target_ids"] = token_override
    rows = [event]
    for i, (name, dtype, vals) in enumerate([
        ("logits", "f32", [0.3, 0.7, -1.5]),
        ("gdn_r", "f16", [1.0, 2.0, 0.5]),
        ("gdn_s", "bf16", [0.25, -0.5]),
    ]):
        if tweak and name == tweak[0]:
            vals = vals.copy()
            vals[tweak[1]] += tweak[2]
        if nonfinite and name == "logits":
            vals[0] = float("nan")
        file = folder / f"{name}.bin"
        file.write_bytes(encode(dtype, vals))
        rows.append({
            "kind": "tensor", "step": 0, "phase": "after_prompt",
            "seq": 0, "position": 33024,
            "layer": -1 if name == "logits" else 11,
            "name": name, "dtype": dtype, "shape": [len(vals)],
            "path": file.name, "physical_cell": cell, "rs_plane": 0,
        })
    if duplicate:
        rows.append(rows[1])
    manifest = folder / "trace.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return manifest


class CompareModelStateTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_identical_logical_data_even_with_different_physical_cells(self):
        a = sample_manifest(self.root, "a", cell=2)
        b = sample_manifest(self.root, "b", cell=5)
        r = compare(a, b)
        self.assertEqual(r["logical_events"], 1)
        self.assertEqual(r["tensor_snapshots"], 3)
        self.assertEqual(r["changed_snapshots"], 0)

    def test_detects_pre_sampling_logits_difference(self):
        a = sample_manifest(self.root, "a")
        b = sample_manifest(self.root, "b", tweak=("logits", 1, 0.25))
        r = compare(a, b)
        self.assertEqual(r["changed_snapshots"], 1)
        first = r["first_changed"]
        self.assertEqual(first["name"], "logits")
        self.assertEqual(first["first_changed"]["index"], 1)
        self.assertAlmostEqual(first["max_abs"], 0.25, places=6)

    def test_detects_bfloat_gdn_state_difference(self):
        a = sample_manifest(self.root, "a")
        b = sample_manifest(self.root, "b", tweak=("gdn_s", 0, 0.25))
        r = compare(a, b)
        self.assertEqual(r["changed_snapshots"], 1)
        self.assertEqual(r["first_changed"]["name"], "gdn_s")

    def test_tolerance_changes_only_comparison_threshold(self):
        a = sample_manifest(self.root, "a")
        b = sample_manifest(self.root, "b", tweak=("logits", 1, 0.0001))
        self.assertEqual(compare(a, b)["changed_snapshots"], 1)
        self.assertEqual(compare(a, b, abs_tol=0.001)["changed_snapshots"], 0)

    def test_token_alignment_mismatch_fails_closed(self):
        a = sample_manifest(self.root, "a")
        b = sample_manifest(self.root, "b", token_override=[71, 99])
        with self.assertRaisesRegex(ValueError, "alignment mismatch"):
            compare(a, b)

    def test_nonfinite_logits_rejected(self):
        a = sample_manifest(self.root, "a")
        b = sample_manifest(self.root, "b", nonfinite=True)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            compare(a, b)

    def test_duplicate_tensor_rejected(self):
        a = sample_manifest(self.root, "a", duplicate=True)
        with self.assertRaisesRegex(ValueError, "duplicate tensor"):
            read_manifest(a)

    def test_recurrent_logical_cell_position_must_align(self):
        a = sample_manifest(self.root, "a")
        b = sample_manifest(self.root, "b")
        for path, logical_pos in ((a, 123), (b, 124)):
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for record in records:
                if record["kind"] == "tensor":
                    record["logical_cell_pos"] = logical_pos
            path.write_text("".join(json.dumps(row) + "\n" for row in records))
        with self.assertRaisesRegex(ValueError, "logical recurrent cell position"):
            compare(a, b)

    def test_pre_verify_and_post_rollback_phases_are_distinct(self):
        a = sample_manifest(self.root, "a")
        b = sample_manifest(self.root, "b")
        for manifest in (a, b):
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            event = dict(records[0])
            event.update(step=1, phase="post_rollback", accepted=[42], rollback=True)
            tensor = dict(records[1])
            tensor.update(step=1, phase="post_rollback", path="rollback.bin",
                          name="gdn_r", layer=11, logical_cell_pos=33024,
                          rs_plane=1)
            (manifest.parent / "rollback.bin").write_bytes((manifest.parent / "logits.bin").read_bytes())
            records.extend([event, tensor])
            manifest.write_text("".join(json.dumps(row) + "\n" for row in records))
        result = compare(a, b)
        self.assertEqual(result["logical_events"], 2)
        self.assertEqual(result["changed_snapshots"], 0)


if __name__ == "__main__":
    unittest.main()
