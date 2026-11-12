#!/usr/bin/env python3
"""Synthetic host-only tests for the recurrent serializer byte comparer."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from compare_qsa_gdn_serialized_bytes import compare


def build(root: Path, label: str, *, alter=False, shift=False,
          missing=False, duplicate=False, bad_dtype=False):
    directory = root / label
    directory.mkdir()
    records = [{
        "kind": "event", "step": 0, "phase": "post_rollback",
        "seq": 0, "position": 10, "target_ids": [1, 2],
        "draft_ids": [3], "accepted": [1],
        "rollback": True, "rollback_mode": "snapshot", "replay": False,
    }]
    refs = []
    for role in (["gdn_r"] if missing else ["gdn_r", "gdn_s"]):
        vector = [0.25, 0.75]
        observer_file = directory / (role + ".bin")
        reference_file = directory / (role + ".ref.bin")
        raw = b"".join(struct.pack("<f", x) for x in vector)
        observer_file.write_bytes(raw)
        reference_file.write_bytes(raw[:-1] + b"\x00" if alter and role == "gdn_r" else raw)
        rec = {
            "kind": "tensor", "step": 0, "phase": "post_rollback",
            "seq": 0, "position": 10, "layer": 12, "name": role,
            "dtype": "f32", "shape": [2], "path": observer_file.name,
            "physical_cell": 1, "source_cell": 3, "cache_size": 4,
            "logical_source_row": 11, "rs_plane": 2, "logical_cell_pos": 9,
        }
        records.append(rec)
        refs.append({
            "kind": "serializer_reference",
            "step": 0, "phase": "post_rollback",
            "seq": 0, "position": 10, "layer": 12, "name": role,
            "type": "f16" if bad_dtype and role == "gdn_r" else "f32",
            "offset": 88 + (8 if shift and role == "gdn_r" else 0),
            "bytes": 8, "path": reference_file.name,
        })
    if duplicate:
        refs.append(refs[0])
    manifest = directory / "manifest.jsonl"
    reference = directory / "serializer-reference.jsonl"
    manifest.write_text("".join(json.dumps(x)+"\n" for x in records), encoding="utf-8")
    reference.write_text("".join(json.dumps(x)+"\n" for x in refs), encoding="utf-8")
    return manifest, reference


class SerializerBytesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def test_identical_full_tensor_bytes(self):
        a, b = build(self.root, "match")
        result = compare(a, b)
        self.assertEqual(result["rows_identical"], 2)
        self.assertEqual(result["rows_different"], 0)
        self.assertEqual(result["reference_bytes"], 16)
        self.assertEqual(result["observed_layers"], [12])

    def test_detects_byte_difference(self):
        a, b = build(self.root, "different", alter=True)
        result = compare(a, b)
        self.assertEqual(result["rows_different"], 1)
        self.assertEqual(result["first_difference"]["name"], "gdn_r")
        self.assertEqual(result["first_difference"]["first_byte_offset"], 7)

    def test_reject_wrong_logical_row_offset(self):
        a, b = build(self.root, "shifted", shift=True)
        with self.assertRaisesRegex(ValueError, "offset disagrees"):
            compare(a, b)

    def test_reject_serialized_dtype_mismatch(self):
        a, b = build(self.root, "dtype", bad_dtype=True)
        with self.assertRaisesRegex(ValueError, "dtype"):
            compare(a, b)

    def test_reject_duplicate_reference(self):
        a, b = build(self.root, "duplicate", duplicate=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            compare(a, b)

    def test_reject_missing_observer_row(self):
        a, b = build(self.root, "missing", missing=True)
        with b.open("a") as f:
            f.write(json.dumps({
                "kind": "serializer_reference", "step": 0,
                "phase": "post_rollback", "seq": 0,
                "position": 10, "layer": 13, "name": "gdn_r",
                "type": "f32", "offset": 0,
                "bytes": 8, "path": "gdn_r.ref.bin",
            })+"\n")
        with self.assertRaisesRegex(ValueError, "no aligned observer row"):
            compare(a, b)


if __name__ == "__main__":
    unittest.main()
