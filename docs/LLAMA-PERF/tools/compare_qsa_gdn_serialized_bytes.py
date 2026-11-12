#!/usr/bin/env python3
"""Compare observer GDN rows to bytes emitted by actual recurrent serialization.

The opt-in LLAMA_QSA_GDN_SERIAL_REF_LAYER=<layer> mode emits selected R/S/P
tensor payload bytes through llama_memory_recurrent::state_write's own writer.
The observer separately reads its logical tensor rows. Both paths use the
same owning backend tensor-get API; this validates offset-selection/copy
agreement, NOT an independent GPU device-memory read implementation.

python3 docs/LLAMA-PERF/tools/compare_qsa_gdn_serialized_bytes.py \
    /external/gdn/manifest.jsonl \
    /external/gdn/serializer-reference.jsonl --require-identical
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from compare_qsa_model_state import read_manifest, tensor_key


def _path(manifest: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else manifest.parent / path


def _compare_file(a: Path, b: Path, expected: int) -> tuple[bool, int | None]:
    if a.stat().st_size != expected or b.stat().st_size != expected:
        raise ValueError(f"incorrect reference/observer file size: {a} {b}")
    offset = 0
    with a.open("rb") as f_a, b.open("rb") as f_b:
        while offset < expected:
            n = min(128 * 1024, expected - offset)
            left, right = f_a.read(n), f_b.read(n)
            if len(left) != n or len(right) != n:
                raise ValueError("truncated serialized/observer data")
            if left != right:
                index = next(i for i, (x, y) in enumerate(zip(left, right)) if x != y)
                return False, offset + index
            offset += n
    return True, None


def compare(observer_manifest: Path, serializer_manifest: Path,
            max_rows: int = 256, max_bytes: int = 512*1024*1024) -> dict:
    if max_rows < 1 or max_rows > 1024 or max_bytes < 1 or max_bytes > 2**30:
        raise ValueError("invalid reference row/byte limits")
    events, tensors = read_manifest(observer_manifest)
    if serializer_manifest.stat().st_size > 4*1024*1024:
        raise ValueError("serializer reference manifest exceeds 4 MiB")

    seen = set()
    result = {
        "observer_manifest": str(observer_manifest),
        "serializer_manifest": str(serializer_manifest),
        "reference_rows": 0, "reference_bytes": 0, "rows_identical": 0,
        "rows_different": 0, "first_difference": None, "observed_layers": [],
        "note": ("Compares actual state_write tensor payloads with GDN observer rows; "
                 "both use ggml_backend_tensor_get, so this is not an "
                 "independent backend transfer validation."),
    }
    layers = set()
    with serializer_manifest.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict) or rec["kind"] != "serializer_reference":
                    raise ValueError("unexpected serializer reference record")
                key = tensor_key(rec)
                if key in seen:
                    raise ValueError(f"duplicate serialized reference key {key}")
                seen.add(key)
                if key not in tensors:
                    raise ValueError(f"reference has no aligned observer row {key}")
                if key[:4] not in events:
                    raise ValueError("reference lacks matching logical checkpoint event")
                observer = tensors[key]
                if rec["name"] not in ("gdn_r", "gdn_s", "gdn_p"):
                    raise ValueError("unexpected tensor role")
                if rec["type"] != observer["dtype"]:
                    raise ValueError(f"mismatched serializer dtype {key}")
                if len(observer["shape"]) != 1:
                    raise ValueError(f"non-vector recurrent state row {key}")
                width = 4 if rec["type"] == "f32" else 2
                expected = math.prod(observer["shape"]) * width
                if (expected > 8*1024*1024 or
                    type(rec["offset"]) is not int or rec["offset"] < 0 or
                    type(rec["bytes"]) is not int or rec["bytes"] != expected):
                    raise ValueError(f"invalid serialized reference dimensions {key}")
                row = observer.get("logical_source_row")
                if type(row) is not int or row < 0:
                    raise ValueError(f"observer lacks logical source row {key}")
                if rec["offset"] != row * expected:
                    raise ValueError(f"serializer byte offset disagrees with logical row {key}")
                result["reference_rows"] += 1
                result["reference_bytes"] += expected
                if result["reference_rows"] > max_rows or result["reference_bytes"] > max_bytes:
                    raise ValueError("reference byte/row budget exceeded")
                same, first = _compare_file(
                    _path(observer_manifest, observer["path"]),
                    _path(serializer_manifest, rec["path"]), expected)
                if same:
                    result["rows_identical"] += 1
                else:
                    result["rows_different"] += 1
                    if result["first_difference"] is None:
                        result["first_difference"] = {
                            "step": key[0], "phase": key[1], "seq": key[2],
                            "position": key[3], "layer": key[4], "name": key[5],
                            "first_byte_offset": first,
                        }
                layers.add(rec["layer"])
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{serializer_manifest}:{lineno}: {exc}") from exc

    if not result["reference_rows"] or len(layers) != 1:
        raise ValueError("expected at least one reference row from exactly one layer")
    result["observed_layers"] = sorted(layers)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observer_manifest", type=Path)
    parser.add_argument("serializer_manifest", type=Path)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--max-bytes", type=int, default=512*1024*1024)
    parser.add_argument("--require-identical", action="store_true")
    args = parser.parse_args()
    try:
        result = compare(args.observer_manifest, args.serializer_manifest,
                         args.max_rows, args.max_bytes)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 2 if args.require_identical and result["rows_different"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
