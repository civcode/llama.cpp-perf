#!/usr/bin/env python3
"""Build six deterministic long-context QSA task-quality probes.

The data is newly synthesized for a reproducible *functional pilot*, not a
human task-quality benchmark or a certified held-out evaluation. Gold answers
are stored only in the external JSON manifest, never in the question suffix.

Usage:
  python3 docs/LLAMA-PERF/tools/generate_qsa_quality_tasks.py \
    --out /external/iteration-0024/tasks.json --records 900 --seed 20261112

No GPU, internet access, model execution, or external Python packages needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import string


KINDS = ("retrieval", "code_reasoning", "instruction_following")


def _code(rng: random.Random, n: int) -> str:
    return "".join(rng.choices(string.ascii_uppercase + string.digits, k=n))


def _compute_first(a: int, b: int) -> int:
    state = (a * 7 + b) % 1009
    for i in range(1, 9):
        if state % 3 == 0:
            state = (state // 3 + i * 5) % 1009
        else:
            state = (state * 2 + i + (a % 7)) % 1009
    return state


def _compute_second(values: list[int]) -> int:
    acc = 1
    for i, v in enumerate(values):
        acc = (acc * 11 + v * (i + 3)) % 997
        if (acc + v) % 5 == 0:
            acc = (acc ^ v) % 997
    return acc


def make_tasks(seed: int = 20261112, records: int = 900) -> list[dict]:
    if not 32 <= records <= 1250 or not 0 <= seed < 2**32:
        raise ValueError("records must be 32..1250 and seed must be uint32")
    tasks = []
    for task_index in range(6):
        rng = random.Random(seed + 100003 * task_index)
        rows = []
        for i in range(records):
            rows.append({
                "id": f"K{i:05d}",
                "next": f"K{(i * 17 + 23) % records:05d}",
                "amount": rng.randint(50, 950),
                "owner": _code(rng, 8),
                "tag": _code(rng, 10),
                "status": rng.choice(("READY", "REVIEW", "HOLD", "CLOSED")),
                "region": rng.choice(("NORTH", "SOUTH", "EAST", "WEST")),
            })
        by_id = {row["id"]: row for row in rows}
        ids = [r["id"] for r in rows]
        focus = rng.choice(ids[records // 8: 7 * records // 8])
        r0 = by_id[focus]
        r1 = by_id[r0["next"]]
        r2 = by_id[r1["next"]]
        ordered = rng.sample(ids, 5)
        kind = KINDS[task_index // 2]
        if task_index == 0:
            question = (
                f"Starting at record {focus}, follow exactly TWO next pointers. "
                "Return only the TAG of the final record, in uppercase, "
                "with no label, whitespace or explanation."
            )
            expected = {"type": "literal", "value": r2["tag"]}
        elif task_index == 1:
            question = (
                f"Starting at record {focus}, follow one next pointer. "
                "Add the AMOUNT of the starting and destination records. "
                "Return only the base-10 integer, without punctuation."
            )
            expected = {"type": "literal", "value": str(r0["amount"] + r1["amount"])}
        elif task_index == 2:
            question = (
                "Execute the following Python function on AMOUNT fields "
                f"a=AMOUNT({focus}), b=AMOUNT({r0['next']}). "
                "Return only the exact base-10 integer it returns.\n"
                "def calc(a, b):\n"
                "    state = (a * 7 + b) % 1009\n"
                "    for i in range(1, 9):\n"
                "        if state % 3 == 0:\n"
                "            state = (state // 3 + i * 5) % 1009\n"
                "        else:\n"
                "            state = (state * 2 + i + (a % 7)) % 1009\n"
                "    return state"
            )
            expected = {"type": "literal", "value": str(_compute_first(r0["amount"], r1["amount"]))}
        elif task_index == 3:
            question = (
                "Evaluate the following Python function with values equal "
                "to the AMOUNT fields of these records in the given order: "
                + ", ".join(ordered) + ". Return only the final base-10 integer.\n"
                "def scan(values):\n"
                "    acc = 1\n"
                "    for i, v in enumerate(values):\n"
                "        acc = (acc * 11 + v * (i + 3)) % 997\n"
                "        if (acc + v) % 5 == 0:\n"
                "            acc = (acc ^ v) % 997\n"
                "    return acc"
            )
            expected = {"type": "literal", "value": str(_compute_second([by_id[x]["amount"] for x in ordered]))}
        elif task_index == 4:
            question = (
                f"Use starting record {focus} and follow its NEXT pointer exactly once. "
                'Produce ONLY a valid JSON object with exactly these four keys: '
                '"ticket", "owner", "priority", "amount". '
                "ticket is the starting ID; owner is the destination OWNER; "
                "priority is HIGH if the destination AMOUNT is at least 600, otherwise NORMAL; "
                "amount is the destination AMOUNT as a JSON integer. "
                "No code fence, comments, extra keys or explanatory text."
            )
            expected = {"type": "json", "value": {
                "ticket": focus, "owner": r1["owner"],
                "priority": "HIGH" if r1["amount"] >= 600 else "NORMAL",
                "amount": r1["amount"],
            }}
        else:
            question = (
                "Take these five record IDs and examine their AMOUNT fields: "
                + ", ".join(ordered) + ". "
                'Return ONLY a JSON object with exactly two keys, "ids" and "total". '
                '"ids" is an array containing the five IDs sorted by descending AMOUNT, '
                "breaking ties by ascending ID. "
                '"total" is the integer sum of those five AMOUNT fields. '
                "Do not add a code fence, comments, or extra keys."
            )
            ranked = sorted(ordered, key=lambda k: (-by_id[k]["amount"], k))
            expected = {"type": "json", "value": {
                "ids": ranked, "total": sum(by_id[x]["amount"] for x in ordered),
            }}
        ledger = "\n".join(
            f"REC {r['id']} | NEXT={r['next']} | AMOUNT={r['amount']} | "
            f"OWNER={r['owner']} | TAG={r['tag']} | STATUS={r['status']} | "
            f"REGION={r['region']}"
            for r in rows
        )
        prompt = (
            "This is a controlled long-context functional evaluation. "
            "Use only the following fictional ledger. Each REC ID is unique. "
            "All pointers refer to an ID in this same ledger.\n"
            "<LEDGER>\n" + ledger + "\n</LEDGER>\n"
            "TASK:\n" + question
        )
        if len(prompt.encode("utf-8")) > 245760:
            raise ValueError("generated prompt exceeds 240 KiB")
        tasks.append({
            "id": f"{kind}-{1 + task_index % 2}",
            "kind": kind,
            "prompt": prompt,
            "expected": expected,
            "max_tokens": 128 if kind == "instruction_following" else 64,
            "source": "deterministically_generated_fictional_ledger",
        })
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--records", type=int, default=900)
    args = parser.parse_args()
    if not 600 <= args.records <= 1250:
        parser.error("GPU pilot requires 600..1250 records; check actual tokenizer and KV eligibility")
    try:
        tasks = make_tasks(args.seed, args.records)
        output = {
            "version": 1,
            "seed": args.seed,
            "records_per_task": args.records,
            "provenance": "synthetic_functional_pilot_not_human_held_out",
            "tasks": tasks,
        }
        with args.out.open("x", encoding="utf-8") as stream:
            json.dump(output, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except (OSError, ValueError) as err:
        parser.error(str(err))
    print(f"Created {len(tasks)} tasks, two per category: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
