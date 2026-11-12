#!/usr/bin/env python3
"""CPU-only synthetic tests for bounded GDN drift localization."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest

from analyze_qsa_gdn_drift_layers import analyze, csv_indices


class GdnDriftLocalizerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.manifests = []
        for condition in ("dense-a", "dense-b", "sparse"):
            directory = self.root / condition
            directory.mkdir()
            self.manifests.append(directory / "manifest.jsonl")
        self.events = [
            {"kind": "event", "step": step, "phase": "pre_verification",
             "seq": 0, "position": 100+step, "target_ids": [1,2],
             "draft_ids": [3], "accepted": [1], "rollback": False}
            for step in (0, 1)
        ]
        self.rows = [
            (0, 12, "gdn_r"), (0, 12, "gdn_s"), (0, 32, "gdn_r"),
            (1, 12, "gdn_r"), (1, 12, "gdn_p"),
        ]
        self.mutate = {}

    def _prepare(self):
        for run, path in zip(("dense-a", "dense-b", "sparse"), self.manifests):
            records = [dict(event) for event in self.events]
            for step, layer, name in self.rows:
                row_id = (run, step, layer, name)
                vals = self.mutate.get(row_id, (1.0, 2.0))
                filename = f"row-{step}-{layer}-{name}.bin"
                (path.parent / filename).write_bytes(
                    b"".join(struct.pack("<f", v) for v in vals))
                records.append({
                    "kind": "tensor", "step": step, "phase": "pre_verification",
                    "seq": 0, "position": 100+step,
                    "layer": layer, "name": name, "dtype": "f32",
                    "shape": [2], "logical_cell_pos": 100+step,
                    "physical_cell": 0, "rs_plane": 0,
                    "path": filename,
                })
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in records))

    def _run(self, **kwargs):
        self._prepare()
        return analyze(*self.manifests, **kwargs)

    def test_exact_dense_and_no_sparse_differences(self):
        report = self._run(steps=(0,1))
        self.assertEqual(report["selected_recurrent_rows"], 5)
        self.assertEqual(report["dense_controls_exact_bytes"], 5)
        self.assertEqual(report["sparse_different_rows"], 0)
        self.assertIsNone(report["first_observed_difference"])
        self.assertEqual(report["production_promotion"], "NO_GO")

    def test_earliest_step_and_layer_for_changed_recurrent_row(self):
        self.mutate[("sparse", 0, 12, "gdn_r")] = (1.25, 2.0)
        self.mutate[("sparse", 1, 12, "gdn_r")] = (2.0, 2.0)
        report = self._run(steps=(0,1))
        self.assertEqual(report["sparse_different_rows"], 2)
        self.assertEqual(report["first_observed_difference"]["step"], 0)
        self.assertEqual(report["first_observed_difference"]["layer"], 12)
        self.assertEqual(report["first_observed_difference"]["role"], "gdn_r")
        self.assertEqual(report["by_layer"]["12"]["different_rows"], 2)
        self.assertEqual(report["by_layer"]["32"]["different_rows"], 0)

    def test_step_and_layer_filter(self):
        self.mutate[("sparse", 0, 12, "gdn_r")] = (1.25, 2.0)
        report = self._run(steps=(1,), layers=(12,))
        self.assertEqual(report["selected_recurrent_rows"], 2)
        self.assertEqual(report["sparse_different_rows"], 0)

    def test_dense_signed_zero_bit_difference_rejected(self):
        self.mutate[("dense-a", 0, 12, "gdn_r")] = (0.0, 2.0)
        self.mutate[("dense-b", 0, 12, "gdn_r")] = (-0.0, 2.0)
        with self.assertRaisesRegex(ValueError, "dense A/B snapshots differ bytewise"):
            self._run()

    def test_dense_ordinary_difference_rejected(self):
        self.mutate[("dense-b", 0, 12, "gdn_r")] = (1.01, 2.0)
        with self.assertRaisesRegex(ValueError, "dense A/B snapshots differ bytewise"):
            self._run()

    def test_mismatched_token_histories_rejected(self):
        self._prepare()
        lines = self.manifests[2].read_text().splitlines()
        row = json.loads(lines[0])
        row["target_ids"] = [4]
        lines[0] = json.dumps(row)
        self.manifests[2].write_text("\n".join(lines)+"\n")
        with self.assertRaisesRegex(ValueError, "event history differs"):
            analyze(*self.manifests)

    def test_mismatched_logical_recurrent_position_rejected(self):
        self._prepare()
        lines = self.manifests[2].read_text().splitlines()
        for i in range(2, len(lines)):
            row = json.loads(lines[i])
            if row.get("kind") == "tensor" and row.get("name") == "gdn_r":
                row["logical_cell_pos"] = 99
                lines[i] = json.dumps(row)
                break
        self.manifests[2].write_text("\n".join(lines)+"\n")
        with self.assertRaisesRegex(ValueError, "unaligned logical_cell_pos"):
            analyze(*self.manifests)

    def test_missing_tensor_rejected(self):
        self._prepare()
        lines = self.manifests[2].read_text().splitlines()
        self.manifests[2].write_text("\n".join(lines[:-1])+"\n")
        with self.assertRaisesRegex(ValueError, "tensor key sets"):
            analyze(*self.manifests, steps=(0,))

    def test_missing_requested_step_rejected(self):
        with self.assertRaisesRegex(ValueError, "checkpoint missing"):
            self._run(steps=(5,))

    def test_invalid_step_csv_rejected(self):
        for text in ("", "0,,1", "0,0", "-1", "x", "0, 1", "10001"):
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    csv_indices(text, name="steps", maximum=16)
        self.assertEqual(csv_indices("9,0,6", name="steps", maximum=16), (0,6,9))

    def test_too_many_selected_rows_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceed max_rows"):
            self._run(steps=(0,1), max_rows=2)

    def test_distinct_capture_paths_required(self):
        self._prepare()
        with self.assertRaisesRegex(ValueError, "manifests must be distinct"):
            analyze(self.manifests[0], self.manifests[0], self.manifests[2])

    def test_missing_f32_row_rejected(self):
        self._prepare()
        (self.manifests[2].parent / "row-0-12-gdn_r.bin").unlink()
        with self.assertRaises(FileNotFoundError):
            analyze(*self.manifests)

    def test_role_filter_excludes_logits(self):
        self.rows.append((0, -1, "logits"))
        report = self._run()
        self.assertEqual(report["selected_recurrent_rows"], 3)
        self.assertNotIn("-1", report["by_layer"])


if __name__ == "__main__":
    unittest.main()
