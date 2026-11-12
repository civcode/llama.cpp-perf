#!/usr/bin/env python3
"""Generate 12 predeclared, calibrated long-context QSA quality tasks.

A deterministic fictional ledger supplies identical long-context distractor
material for two difficulty tiers:
  anchored: the needed exact facts are repeated at the question suffix.
            This controls formatting/basic calculation under long context;
            it is NOT a meaningful long-range retrieval test.
  lookup:   the facts are ONLY in the ledger, with one-record lookup and
            no multi-hop pointer chasing (unlike floor-limited Report 0024).
Two variations per tier and three categories = 12 fixed cases. The task
rubric is generated independently of inference and never inferred from a
dense or sparse model answer. This is synthetic difficulty calibration, not a
certified held-out deployment benchmark.

python3 docs/LLAMA-PERF/tools/generate_qsa_calibrated_tasks.py \
    --out /external/iteration-0025/tasks.json --records 520 --seed 20261113
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from generate_qsa_quality_tasks import KINDS, _code


def make_calibrated_tasks(seed: int = 20261113, records: int = 520) -> list[dict]:
    if not (0 <= seed < 2**32 and 40 <= records <= 850):
        raise ValueError("seed must be uint32 and records must be 40..850")
    tasks = []
    for k, kind in enumerate(KINDS):
        for tier_idx, tier in enumerate(("anchored", "lookup")):
            for case in (1, 2):
                idx = (k * 2 + tier_idx) * 2 + case - 1
                rng = random.Random(seed + 100019 * idx)
                rows = []
                for i in range(records):
                    rows.append({
                        "id": f"K{i:05d}",
                        "amount": rng.randint(100, 899),
                        "owner": _code(rng, 6),
                        "tag": _code(rng, 10),
                        "status": rng.choice(("READY", "HOLD", "REVIEW", "CLOSED")),
                        "region": rng.choice(("NORTH", "SOUTH", "WEST", "EAST")),
                    })
                focus_index = rng.randrange(records // 5, 4 * records // 5)
                target = rows[focus_index]
                target2 = rows[(focus_index + 13) % records]
                anchor = tier == "anchored"
                if kind == "retrieval":
                    if anchor:
                        question = (
                            "The following exact record is highlighted for the current task:\n"
                            f"ID={target['id']} OWNER={target['owner']} TAG={target['tag']}\n"
                            "Return ONLY its TAG value, without any explanatory text."
                        )
                    else:
                        question = (
                            f"Find record {target['id']} in the ledger. "
                            "Return ONLY its TAG value, without any explanatory text. "
                            "Do not follow any other record."
                        )
                    expected = {"type": "literal", "value": target["tag"]}
                elif kind == "code_reasoning":
                    if anchor:
                        question = (
                            "The required AMOUNT fields are reproduced here:\n"
                            f"AMOUNT({target['id']})={target['amount']}\n"
                            f"AMOUNT({target2['id']})={target2['amount']}\n"
                            "Calculate (AMOUNT(first) + 2 * AMOUNT(second)) % 97. "
                            "Return ONLY the decimal integer."
                        )
                        value = (target["amount"] + 2 * target2["amount"]) % 97
                    else:
                        question = (
                            f"Find the AMOUNT field of record {target['id']} in the ledger. "
                            "Calculate (3 * AMOUNT + 7) % 97. "
                            "Return ONLY the decimal integer. No code explanation."
                        )
                        value = (3 * target["amount"] + 7) % 97
                    expected = {"type": "literal", "value": str(value)}
                else:
                    if anchor:
                        question = (
                            "The following exact record is highlighted for the current task:\n"
                            f"ID={target['id']} REGION={target['region']} AMOUNT={target['amount']}\n"
                            'Return ONLY valid JSON with exactly these keys: "id", "region", "amount". '
                            "Copy the displayed ID and REGION strings and AMOUNT integer. "
                            "No code fences or extra keys."
                        )
                        val = {"id": target["id"], "region": target["region"],
                               "amount": target["amount"]}
                    else:
                        question = (
                            f"Find record {target['id']} in the ledger. "
                            'Return ONLY valid JSON with exactly these keys: "id", "owner", "status". '
                            "Use its ID, OWNER, STATUS values from that single record. "
                            "No code fences or extra keys."
                        )
                        val = {"id": target["id"], "owner": target["owner"],
                               "status": target["status"]}
                    expected = {"type": "json", "value": val}
                ledger = "\n".join(
                    f"REC {r['id']} | AMOUNT={r['amount']} | OWNER={r['owner']} | "
                    f"TAG={r['tag']} | STATUS={r['status']} | REGION={r['region']}"
                    for r in rows
                )
                prompt = (
                    "This is a fictional and fully specified ledger. "
                    "Use its values, not any external knowledge. Every REC ID is unique.\n"
                    "<LEDGER>\n" + ledger + "\n</LEDGER>\n"
                    "TASK:\n" + question
                )
                if len(prompt.encode("utf-8")) > 245760:
                    raise ValueError("prompt too large for local task runner")
                tasks.append({
                    "id": f"{kind}-{tier}-{case}",
                    "kind": kind, "difficulty": tier,
                    "prompt": prompt, "expected": expected,
                    "max_tokens": 96 if kind == "instruction_following" else 48,
                    "source": "synthetic_calibration_ledger_not_independent_heldout",
                })
    return tasks


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--records", type=int, default=520)
    p.add_argument("--seed", type=int, default=20261113)
    args = p.parse_args()
    if not (390 <= args.records <= 850):
        p.error("GPU pilot records must be in 390..850; verify tokenizer/context and sparse KV eligibility")
    try:
        tasks = make_calibrated_tasks(args.seed, args.records)
        obj = {
            "version": 1, "profile": "0025_predeclared_anchored_vs_lookup",
            "seed": args.seed, "records_per_task": args.records,
            "provenance": "synthetic_long_context_calibration_not_heldout",
            "tasks": tasks,
        }
        with args.out.open("x", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
    except (OSError, ValueError) as exc:
        p.error(str(exc))
    print(f"Created {len(tasks)} calibrated tasks: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
