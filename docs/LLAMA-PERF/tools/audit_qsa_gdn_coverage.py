#!/usr/bin/env python3
"""Audit opt-in logical GDN observer coverage without reading multi-GB raw rows.

Validates all recurrent tensor records and their source-cell/rollback-plane
mapping against the logged per-sequence cache size. This validates metadata
consistency, not backend byte equality or numerical model equivalence.

 python3 docs/LLAMA-PERF/tools/audit_qsa_gdn_coverage.py \
    /path/to/iteration-0020/dense-A-gdn/manifest.jsonl
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import json
from pathlib import Path
from compare_qsa_model_state import read_manifest


def _integer(record: dict, name: str) -> int:
    value = record.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"missing or invalid integer metadata {name}")
    return value


def audit(path: Path) -> dict:
    events, tensors = read_manifest(path)
    grouped = defaultdict(list)
    count_source_ref = 0
    count_nontrivial = 0
    count_planes = 0
    modes = defaultdict(int)
    source_examples = []
    rolled_examples = []

    for tkey, t in tensors.items():
        if t["name"] not in ("gdn_r", "gdn_s", "gdn_p"):
            raise ValueError(f"unexpected non-GDN tensor: {tkey}")
        cell = _integer(t, "physical_cell")
        src = _integer(t, "source_cell")
        cache_size = _integer(t, "cache_size")
        row = _integer(t, "logical_source_row")
        plane = _integer(t, "rs_plane")
        logical_pos = _integer(t, "logical_cell_pos")
        if (cache_size < 1 or cache_size > 65536 or cell < 0 or cell >= cache_size
                or src < -1 or src >= cache_size or plane < 0 or plane > 256
                or logical_pos < 0):
            raise ValueError(f"invalid GDN cache metadata: {tkey}")
        expected = plane*cache_size + (src if src >= 0 else cell)
        if row != expected:
            raise ValueError(f"incorrect logical source row for {tkey}: {row} != {expected}")
        if tkey[4] < 0:
            raise ValueError(f"invalid GDN layer: {tkey}")
        if len(t["shape"]) != 1:
            raise ValueError(f"unexpected GDN row shape: {tkey}")

        if src >= 0:
            count_source_ref += 1
        if src >= 0 and src != cell:
            count_nontrivial += 1
            if len(source_examples) < 4:
                source_examples.append({
                    "step": tkey[0], "phase": tkey[1], "layer": tkey[4],
                    "role": tkey[5], "physical_cell": cell, "source_cell": src,
                    "rs_plane": plane, "logical_source_row": row,
                })
        if plane > 0:
            count_planes += 1
            if len(rolled_examples) < 4:
                rolled_examples.append({
                    "step": tkey[0], "phase": tkey[1],
                    "layer": tkey[4], "role": tkey[5],
                    "rs_plane": plane, "logical_source_row": row,
                })
        grouped[tkey[:4]].append((tkey, (cell, src, cache_size, row, plane, logical_pos)))

    if set(grouped) != set(events):
        raise ValueError("GDN tensor groups do not match event checkpoints")
    for event_key, records in grouped.items():
        # All layers in a checkpoint must resolve the same sequence state.
        common = records[0][1]
        for tensor_key, metadata in records[1:]:
            if common != metadata:
                raise ValueError(f"inconsistent cell/plane mapping within checkpoint {tensor_key}")
        layers = defaultdict(set)
        for tensor_key, _ in records:
            layers[tensor_key[4]].add(tensor_key[5])
        for layer, roles in layers.items():
            if not {"gdn_r", "gdn_s"}.issubset(roles):
                raise ValueError(f"incomplete R/S layer {layer} in {event_key}")
        event = events[event_key]
        mode = event.get("rollback_mode", "unrecorded")
        if mode not in ("none", "snapshot", "checkpoint_restore", "unrecorded"):
            raise ValueError(f"unknown rollback mode in {event_key}: {mode}")
        if mode != "unrecorded" and mode != "none" and not event["rollback"]:
            raise ValueError(f"rollback mode without rollback in {event_key}")
        modes[(event_key[1], mode)] += 1

    return {
        "manifest": str(path),
        "events": len(events),
        "tensor_rows": len(tensors),
        "source_cell_referenced_rows": count_source_ref,
        "nontrivial_source_indirection_rows": count_nontrivial,
        "nonzero_rollback_plane_rows": count_planes,
        "has_nontrivial_source_indirection": count_nontrivial > 0,
        "has_pending_rollback_plane": count_planes > 0,
        "post_rollback_snapshot_events": modes.get(("post_rollback", "snapshot"), 0),
        "post_rollback_checkpoint_restore_events": modes.get(("post_rollback", "checkpoint_restore"), 0),
        "phase_and_mode_events": [
            {"phase": phase, "rollback_mode": mode, "count": count}
            for (phase, mode), count in sorted(modes.items())
        ],
        "source_indirection_examples": source_examples,
        "rollback_plane_examples": rolled_examples,
        "note": ("Coverage of source indirection and checkpoint restore must be "
                 "demonstrated in live or controlled tests; metadata validation alone "
                 "cannot prove correct tensor contents."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-source-indirection", action="store_true")
    parser.add_argument("--require-checkpoint-restore", action="store_true")
    args = parser.parse_args()
    try:
        result = audit(args.manifest)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    if args.require_source_indirection and not result["has_nontrivial_source_indirection"]:
        return 2
    if args.require_checkpoint_restore and not result["post_rollback_checkpoint_restore_events"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
