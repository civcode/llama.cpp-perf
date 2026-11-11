#!/usr/bin/env python3
"""Compare bounded GGML_QSA_MASK_AUDIT dumps from two fresh model processes.

The two executions must have identical input tokens/configuration. This tool
compares **mask bytes/indices**, not logits, GDN state, or model correctness.
It uses only the Python 3 standard library; no GPU or large framework needed.

Capture example (in separate processes; never for benchmark timing):
    GGML_QSA_MASK_AUDIT=1000 GGML_QSA_MASK_DUMP_LIMIT=256 \
      GGML_QSA_MASK_AUDIT_DIR=/path/to/empty-output-dir <server ...>

    python3 docs/LLAMA-PERF/tools/compare_qsa_mask_dumps.py \
      dense.log sparse.log > mask-diff.json

DUMP_LIMIT=256 is per backend/device. Audit and dumping disable GPU graphs
and add synchronization; results are diagnostic only.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import json
from pathlib import Path
import struct
import sys


def _fields(line: str, marker: str) -> dict[str, str] | None:
    needle = marker + " "
    if needle not in line:
        return None
    rest = line.split(needle, 1)[1]
    return dict(chunk.split("=", 1) for chunk in rest.split() if "=" in chunk)


def _key(fields: dict[str, str]) -> tuple[str, int, int, int]:
    return (fields["backend"], int(fields["device"]), int(fields["id"]), int(fields["stream"]))


def parse_log(log: Path):
    audits: dict[tuple[str, int], list[dict]] = defaultdict(list)
    dumps: dict[tuple[str, int, int, int], Path] = {}
    with log.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            dump = _fields(line, "QSA_MASK_DUMP")
            if dump and "path" in dump:
                key = _key(dump)
                if key in dumps:
                    raise ValueError(f"duplicate dump key {key} in {log}")
                filepath = Path(dump["path"])
                dumps[key] = filepath if filepath.is_absolute() else log.parent / filepath
                continue
            audit = _fields(line, "QSA_MASK_AUDIT")
            if not audit or "id" not in audit:
                continue
            key = _key(audit)
            record = {
                "key": key,
                "backend": key[0], "device": key[1], "id": key[2], "stream": key[3],
                "kv": int(audit["kv"]), "bound": int(audit["bound"]),
                "qheads": int(audit["qheads"]), "kvheads": int(audit["kvheads"]),
                "counts": [int(n) for n in audit["counts"].split(",")],
                "union": int(audit["union"]), "capacity": int(audit["capacity"]),
                "nan": int(audit["nan"]), "posinf": int(audit["posinf"]),
            }
            if len(record["counts"]) != 4:
                raise ValueError(f"bad Q=4 count record in {log}: {line.strip()}")
            if record["nan"] or record["posinf"] or (
                record["union"] > record["capacity"]
                or max(record["counts"]) > record["bound"]
            ):
                raise ValueError(f"invalid or over-bound audit in {log}: {record}")
            audits[key[:2]].append(record)

    if not audits:
        raise ValueError(f"no QSA_MASK_AUDIT rows in {log}")
    if not dumps:
        raise ValueError(f"no QSA_MASK_DUMP paths in {log}; enable DUMP_LIMIT")
    for records in audits.values():
        for record in records:
            record["dump"] = dumps.get(record["key"])
    unused = set(dumps) - {r["key"] for items in audits.values() for r in items}
    if unused:
        raise ValueError(f"dumps without audit rows in {log}: {sorted(unused)[:3]}")
    return audits


def mask_content(record: dict) -> tuple[list[set[int]], list[array]]:
    path = record["dump"]
    if path is None:
        raise ValueError(f"missing mask dump for {record['key']}")
    raw = path.read_bytes()
    kv = record["kv"]
    if len(raw) != 4 * kv * 2:
        raise ValueError(f"bad dump size {path}: got {len(raw)}, need {4*kv*2}")
    words = array("H")
    words.frombytes(raw)
    if sys.byteorder != "little":
        words.byteswap()
    rows = [words[q*kv:(q+1)*kv] for q in range(4)]
    selected = []
    for row in rows:
        indices = set()
        for pos, half in enumerate(row):
            if (half & 0x7c00) != 0x7c00:
                indices.add(pos)
            elif (half & 0x03ff) or not (half & 0x8000):
                raise ValueError(f"non -inf masked value in {path} at position {pos}")
        selected.append(indices)
    union = set().union(*selected)
    if [len(x) for x in selected] != record["counts"] or len(union) != record["union"]:
        raise ValueError(f"binary contents disagree with audit in {path}")
    return selected, rows


def compare_pair(a: dict, b: dict) -> dict | None:
    for f in ("kv", "bound", "capacity", "qheads", "kvheads", "stream"):
        if a[f] != b[f]:
            raise ValueError(
                f"unaligned {f} on {a['backend']} device {a['device']} "
                f"at per-device row {a['ordinal']}: {a[f]} vs {b[f]}"
            )
    a_mask, a_values = mask_content(a)
    b_mask, b_values = mask_content(b)
    a_union = set().union(*a_mask)
    b_union = set().union(*b_mask)
    row_diffs = []
    value_changes = 0
    for q in range(4):
        removed = a_mask[q] - b_mask[q]
        added = b_mask[q] - a_mask[q]
        changed_positions = sorted(
            pos for pos in a_mask[q] & b_mask[q] if a_values[q][pos] != b_values[q][pos]
        )
        changed_values = len(changed_positions)
        value_changes += changed_values
        value_examples = [{
            "position": pos,
            "dense_bits": f"0x{a_values[q][pos]:04x}",
            "sparse_bits": f"0x{b_values[q][pos]:04x}",
            "dense_value": struct.unpack("<e", struct.pack("<H", a_values[q][pos]))[0],
            "sparse_value": struct.unpack("<e", struct.pack("<H", b_values[q][pos]))[0],
        } for pos in changed_positions[:12]]
        row_diffs.append({
            "query": q,
            "dense_count": len(a_mask[q]), "sparse_count": len(b_mask[q]),
            "removed_count": len(removed), "added_count": len(added),
            "removed_first": sorted(removed)[:12], "added_first": sorted(added)[:12],
            "common_selected_mask_value_changes": changed_values,
            "common_selected_mask_value_examples": value_examples,
        })

    if a_mask == b_mask and value_changes == 0:
        return None
    missing = a_union - b_union
    added = b_union - a_union
    return {
        "backend": a["backend"], "device": a["device"],
        "per_device_ordinal": a["ordinal"],
        "dense_id": a["id"], "sparse_id": b["id"],
        "stream": a["stream"], "kv": a["kv"], "bound": a["bound"],
        "dense_union": len(a_union), "sparse_union": len(b_union),
        "union_removed_count": len(missing), "union_added_count": len(added),
        "union_removed_first": sorted(missing)[:12],
        "union_added_first": sorted(added)[:12],
        "selected_mask_value_changes": value_changes,
        "rows": row_diffs,
    }


def compare_logs(dense: Path, sparse: Path) -> dict:
    a = parse_log(dense)
    b = parse_log(sparse)
    if set(a) != set(b):
        raise ValueError(f"backend/device sets differ: {sorted(a)} vs {sorted(b)}")
    summary = {"dense_log": str(dense), "sparse_log": str(sparse),
               "groups_audited": 0, "groups_with_raw_dump": 0,
               "groups_different": 0, "union_count_differences": 0,
               "first_union_count_difference_per_device": [],
               "first_difference_per_device": [],
               "by_device": []}
    for key in sorted(a):
        before, after = a[key], b[key]
        if len(before) != len(after):
            raise ValueError(f"mask row count differs for {key}: {len(before)} vs {len(after)}")
        count = 0
        first = None
        first_union_diff = None
        for ordinal, (ra, rb) in enumerate(zip(before, after)):
            ra["ordinal"] = rb["ordinal"] = ordinal
            for f in ("kv", "bound", "capacity", "qheads", "kvheads", "stream"):
                if ra[f] != rb[f]:
                    raise ValueError(f"misaligned audit row for {key} ordinal {ordinal}: {f}")
            if ra["union"] != rb["union"]:
                summary["union_count_differences"] += 1
                if first_union_diff is None:
                    first_union_diff = {
                        "backend": key[0], "device": key[1],
                        "per_device_ordinal": ordinal,
                        "dense_union": ra["union"], "sparse_union": rb["union"],
                        "dense_id": ra["id"], "sparse_id": rb["id"],
                    }
            if (ra["dump"] is None) != (rb["dump"] is None):
                raise ValueError(f"one-sided dump for {key} ordinal {ordinal}")
            if ra["dump"] is None:
                continue
            diff = compare_pair(ra, rb)
            summary["groups_with_raw_dump"] += 1
            if diff:
                count += 1
                summary["groups_different"] += 1
                if first is None:
                    first = diff
        summary["groups_audited"] += len(before)
        summary["by_device"].append({
            "backend": key[0], "device": key[1],
            "audited": len(before), "raw_differences": count,
        })
        if first_union_diff is not None:
            summary["first_union_count_difference_per_device"].append(first_union_diff)
        if first is not None:
            summary["first_difference_per_device"].append(first)
    if not summary["groups_with_raw_dump"]:
        raise ValueError("no aligned dump pairs")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dense_log", type=Path)
    parser.add_argument("sparse_log", type=Path)
    parser.add_argument("--require-identical", action="store_true",
                        help="exit 2 when any raw mask differs")
    args = parser.parse_args()
    try:
        result = compare_logs(args.dense_log, args.sparse_log)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 2 if args.require_identical and result["groups_different"] else 0


if __name__ == "__main__":
    sys.exit(main())
