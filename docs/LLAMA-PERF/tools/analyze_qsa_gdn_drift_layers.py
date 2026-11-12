#!/usr/bin/env python3
"""Bounded, CPU-only localization of QSA recurrent-state drift.

Uses the already established logical-event and F32/F16/BF16 tensor capture
schema from compare_qsa_model_state.py. Input is a TRIPLE of independently
captured natural-MTP runs: dense A, dense B and sparse. Raw artifacts must
stay outside Git. No inference, CUDA/HIP operations, GPU probing, code/math
changes, numeric acceptance thresholds, or causal attribution occur here.

Unlike the earlier full-state comparator this reports the earliest *observed*
change by event step, layer and recurrent role, then aggregates by
step/phase/layer/role, while hard-requiring byte-identical dense A/B.
Be careful: a "pre_verification" checkpoint in old probes is AFTER the target
decode, not a before-decode measurement. Selection changes and arithmetic
origins may precede the first snapshot. CPU comparison is diagnostic only.

Example (use actual existing external capture manifests, not task JSONL):
 python3 docs/LLAMA-PERF/tools/analyze_qsa_gdn_drift_layers.py \
   /external/dense-a/manifest.jsonl /external/dense-b/manifest.jsonl \
   /external/sparse/manifest.jsonl --steps 0,6,8,9 \
   --layers 12,32,44 --out /external/first-drift.json

Steps are exact checkpoint ordinals, not token counts. The default examines
only step zero across all recurrent layers; it does not read 5-GiB captures.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from compare_qsa_model_state import diff_stats, read_manifest, values

ROLES = frozenset(("gdn_r", "gdn_s", "gdn_p"))
MAX_STEPS = 16
MAX_ROWS = 512
MAX_TOTAL_BYTES = 1024 * 1024 * 1024  # aggregate three-way reads


def csv_indices(raw: str, *, name: str, maximum: int) -> tuple[int, ...]:
    parts = raw.split(",")
    if not 1 <= len(parts) <= maximum:
        raise ValueError(f"{name} needs 1..{maximum} integer indices")
    try:
        nums = tuple(int(p) for p in parts)
    except ValueError as err:
        raise ValueError(f"{name} must be comma-separated integers") from err
    if (any(i < 0 or i > 10000 for i in nums) or
            len(nums) != len(set(nums)) or
            any(not p or p.strip() != p for p in parts)):
        raise ValueError(f"{name} has duplicate, out-of-range or malformed index")
    return tuple(sorted(nums))


def _path(manifest: Path, row: dict) -> Path:
    p = Path(row["path"])
    return p if p.is_absolute() else manifest.parent / p


def _metadata_match(a: dict, b: dict, c: dict, key: tuple) -> None:
    for field in ("dtype", "shape", "logical_cell_pos"):
        if a.get(field) != b.get(field) or a.get(field) != c.get(field):
            raise ValueError(f"unaligned {field} for {key}")


def _increment(stats: dict, changed: bool, max_abs: float, nmse: float) -> None:
    stats["rows"] += 1
    stats["different_rows"] += int(changed)
    stats["max_abs"] = max(stats["max_abs"], max_abs)
    stats["max_row_nmse"] = max(stats["max_row_nmse"], nmse)


def analyze(dense_a: Path, dense_b: Path, sparse: Path, *,
            steps: tuple[int, ...] = (0,),
            layers: tuple[int, ...] | None = None,
            max_rows: int = MAX_ROWS) -> dict:
    if not steps or len(steps) > MAX_STEPS or tuple(sorted(set(steps))) != steps:
        raise ValueError("steps must be a nonempty sorted unique bounded tuple")
    if layers is not None and (
            not layers or len(layers) > 64 or tuple(sorted(set(layers))) != layers):
        raise ValueError("layers must be a nonempty sorted unique bounded tuple")
    if type(max_rows) is not int or not 1 <= max_rows <= MAX_ROWS:
        raise ValueError("max_rows must be 1..512")
    paths = tuple(p.resolve(strict=True) for p in (dense_a, dense_b, sparse))
    if len(set(paths)) != 3:
        raise ValueError("capture manifests must be distinct")
    ea, ta = read_manifest(dense_a)
    eb, tb = read_manifest(dense_b)
    ec, tc = read_manifest(sparse)
    if ea != eb or ea != ec:
        raise ValueError("target/draft candidate, acceptance, rollback or logical event history differs")
    if ta.keys() != tb.keys() or ta.keys() != tc.keys():
        raise ValueError("incomplete or mismatched tensor key sets across three captures")
    available_steps = {key[0] for key in ea}
    if not set(steps).issubset(available_steps):
        raise ValueError(f"requested checkpoint missing: {sorted(set(steps)-available_steps)}")
    chosen = sorted(
        key for key in ta if key[0] in steps
        and key[5] in ROLES and (layers is None or key[4] in layers))
    if not chosen:
        raise ValueError("no selected recurrent-state rows")
    if len(chosen) > max_rows:
        raise ValueError(f"selected recurrent rows {len(chosen)} exceed max_rows={max_rows}")
    totals = {
        "by_step": defaultdict(lambda: {"rows": 0, "different_rows": 0,
                                        "max_abs": 0.0, "max_row_nmse": 0.0}),
        "by_phase": defaultdict(lambda: {"rows": 0, "different_rows": 0,
                                         "max_abs": 0.0, "max_row_nmse": 0.0}),
        "by_layer": defaultdict(lambda: {"rows": 0, "different_rows": 0,
                                         "max_abs": 0.0, "max_row_nmse": 0.0}),
        "by_role": defaultdict(lambda: {"rows": 0, "different_rows": 0,
                                        "max_abs": 0.0, "max_row_nmse": 0.0}),
    }
    first = None
    examples = []
    bytes_read = 0
    raw_equal = 0
    for key in chosen:
        a, b, c = ta[key], tb[key], tc[key]
        _metadata_match(a, b, c, key)
        filenames = (_path(dense_a, a), _path(dense_b, b), _path(sparse, c))
        expected_bytes = math.prod(a["shape"]) * (4 if a["dtype"] == "f32" else 2)
        if not 0 < expected_bytes <= 64*1024*1024:
            raise ValueError(f"invalid bounded recurrent row size: {key}")
        if bytes_read + 3*expected_bytes > MAX_TOTAL_BYTES:
            raise ValueError("bounded total recurrent-state reads exceed 1 GiB")
        for p in filenames:
            if p.stat().st_size != expected_bytes:
                raise ValueError(f"row size mismatch: {key}")
        # Bytewise controls: float equality is NOT sufficient for signed zero,
        # different NaN payloads, or other bit-pattern changes.
        dense_bytes = filenames[0].read_bytes()
        if dense_bytes != filenames[1].read_bytes():
            raise ValueError(f"dense A/B snapshots differ bytewise: {key}")
        raw_equal += 1
        bytes_read += 3*expected_bytes
        data_a = values(dense_a, a)
        data_c = values(sparse, c)
        stats = diff_stats(data_a, data_c, 0.0, 0.0)
        changed = stats["changed_elements"] != 0
        example = {
            "step": key[0], "phase": key[1], "seq": key[2],
            "position": key[3], "layer": key[4], "role": key[5],
            "elements": stats["elements"],
            "changed_elements": stats["changed_elements"],
            "max_abs": stats["max_abs"],
            "nmse_over_dense_energy": stats["nmse_over_dense_energy"],
            "first_changed_element": stats["first_changed"],
        }
        if changed:
            if first is None:
                first = example
            if len(examples) < 16:
                examples.append(example)
        for grouping, tag in (
                ("by_step", str(key[0])),
                ("by_phase", key[1]),
                ("by_layer", str(key[4])),
                ("by_role", key[5]),
        ):
            _increment(totals[grouping][tag], changed, stats["max_abs"],
                       stats["nmse_over_dense_energy"])
    different = sum(v["different_rows"] for v in totals["by_step"].values())
    return {
        "read_only_offline_analysis": True,
        "aligned_logical_events": len(ea),
        "fully_matched_tensor_keys": len(ta),
        "selected_steps": list(steps),
        "selected_layers": list(layers) if layers is not None else "all",
        "selected_recurrent_rows": len(chosen),
        "dense_controls_exact_bytes": raw_equal,
        "sparse_different_rows": different,
        "total_input_bytes_checked": bytes_read,
        "first_observed_difference": first,
        "changed_examples": examples,
        **{name: dict(sorted(section.items())) for name, section in totals.items()},
        "interpretation": (
            "First observed checkpoint is not the origin of arithmetic error. "
            "Pre-verification old captures are after target decode. This "
            "analysis shows recurrent-state drift conditional on aligned "
            "token/acceptance history and exact dense repeats. It does not "
            "isolate the upstream attention or selector cause."
        ),
        "numerical_equivalence": "NOT_ESTABLISHED",
        "production_promotion": "NO_GO",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_a", type=Path)
    parser.add_argument("dense_b", type=Path)
    parser.add_argument("sparse", type=Path)
    parser.add_argument("--steps", default="0", help="comma-separated checkpoint ordinals, max 16")
    parser.add_argument("--layers", help="optional comma-separated layer IDs, max 64")
    parser.add_argument("--max-rows", type=int, default=MAX_ROWS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        steps = csv_indices(args.steps, name="steps", maximum=MAX_STEPS)
        layers = csv_indices(args.layers, name="layers", maximum=64) if args.layers is not None else None
        report = analyze(args.dense_a, args.dense_b, args.sparse,
                         steps=steps, layers=layers, max_rows=args.max_rows)
        with args.out.open("x", encoding="utf-8") as out:
            json.dump(report, out, indent=2, allow_nan=False)
            out.write("\n")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "selected_recurrent_rows": report["selected_recurrent_rows"],
        "dense_controls_exact_bytes": report["dense_controls_exact_bytes"],
        "sparse_different_rows": report["sparse_different_rows"],
        "first_observed_difference": report["first_observed_difference"],
        "production_promotion": "NO_GO",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
