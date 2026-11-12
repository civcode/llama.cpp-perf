#!/usr/bin/env python3
"""Generate bounded counterfactual pairs for long-context QSA quality checking.

Six pairs, each with variants A/B: retrieval of a TAG or exact JSON
extraction of STATUS. For each pair, one factual field in exactly one
ledger record changes while query text, record order and other data stay
identical. Target records are near the beginning, middle or end. No gold
answer is copied into the question suffix.

This is a deterministic synthetic probe, NOT a certified held-out or
representative deployment benchmark. All raw manifests must remain
outside Git.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random

from generate_qsa_quality_tasks import _code


def _ledger(rows: list[dict]) -> str:
    return "\n".join(
        f"REC {r['id']} | AMOUNT={r['amount']} | OWNER={r['owner']} | "
        f"TAG={r['tag']} | STATUS={r['status']} | REGION={r['region']}"
        for r in rows
    )


def make_counterfactual_tasks(seed: int = 20261114, records: int = 520) -> list[dict]:
    if not 0 <= seed < 2**32 or not 80 <= records <= 900:
        raise ValueError("seed must be uint32 and records 80..900")
    tasks = []
    locations = ("early", "middle", "late")
    kinds = ("retrieval", "instruction_following")
    for kind_idx, kind in enumerate(kinds):
        for band_idx, band in enumerate(locations):
            pair_number = kind_idx * 3 + band_idx
            rng = random.Random(seed + pair_number * 104729)
            rows = []
            for i in range(records):
                rows.append({
                    "id": f"K{i:05d}", "amount": rng.randint(100, 899),
                    "owner": _code(rng, 6), "tag": _code(rng, 10),
                    "status": rng.choice(("READY", "PAUSE", "HOLD", "CLOSED")),
                    "region": rng.choice(("NORTH", "SOUTH", "WEST", "EAST")),
                })
            target_idx = (3, records // 2, records - 4)[band_idx]
            target = rows[target_idx]
            pair_id = f"{kind}-{band}"
            a_rows = rows
            b_rows = copy.deepcopy(rows)
            changed_field = "tag" if kind == "retrieval" else "status"
            if changed_field == "tag":
                replacement = _code(rng, 10)
                while replacement == target["tag"] or any(
                        row["tag"] == replacement for row in rows):
                    replacement = _code(rng, 10)
                b_rows[target_idx]["tag"] = replacement
            else:
                b_rows[target_idx]["status"] = (
                    "PAUSE" if target["status"] != "PAUSE" else "READY")
            assert a_rows[target_idx][changed_field] != b_rows[target_idx][changed_field]
            query = (
                f"Find exactly record {target['id']} in the ledger. "
                "Return ONLY the TAG field of this record, with no explanation."
                if kind == "retrieval" else
                f"Find exactly record {target['id']} in the ledger. "
                'Return ONLY valid JSON containing exactly these two fields: '
                '"id" (a string) and "status" (a string). '
                "Use this record's ID and STATUS, with no explanation or code fence."
            )
            for variant, actual_rows in (("a", a_rows), ("b", b_rows)):
                row = actual_rows[target_idx]
                expected = (
                    {"type": "literal", "value": row["tag"]}
                    if kind == "retrieval" else
                    {"type": "json", "value": {
                        "id": row["id"], "status": row["status"]}}
                )
                prompt = (
                    "Use only the following fictional long-context ledger. "
                    "Each REC identifier occurs exactly once. "
                    "Do not use any outside information.\n"
                    "<LEDGER>\n" + _ledger(actual_rows) +
                    "\n</LEDGER>\nTASK:\n" + query
                )
                if len(prompt.encode("utf-8")) > 245760:
                    raise ValueError("generated prompt exceeds 240 KiB")
                tasks.append({
                    "id": f"{pair_id}-{variant}",
                    "kind": kind, "difficulty": "lookup",
                    "pair_id": pair_id, "variant": variant,
                    "target_band": band, "changed_field": changed_field,
                    "target_id": row["id"],
                    "prompt": prompt, "expected": expected,
                    "max_tokens": 32 if kind == "retrieval" else 64,
                    "source": "synthetic_counterfactual_ledger_not_heldout",
                })
    return tasks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--records", type=int, default=520)
    p.add_argument("--seed", type=int, default=20261114)
    args = p.parse_args()
    if not 420 <= args.records <= 900:
        p.error("GPU pilot records must be 420..900; check tokenizer and sparse KV eligibility")
    try:
        tasks = make_counterfactual_tasks(args.seed, args.records)
        with args.out.open("x", encoding="utf-8") as f:
            json.dump({
                "version": 1, "profile": "0026_counterfactual_pairs",
                "seed": args.seed, "records_per_task": args.records,
                "provenance": "synthetic_long_context_counterfactual_not_heldout",
                "tasks": tasks,
            }, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(f"Created {len(tasks)} tasks in {len(tasks)//2} counterfactual pairs: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
