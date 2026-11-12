#!/usr/bin/env python3
"""Freeze a pre-inference LongBench-v2 multiple-choice quality pilot.

Reads an authorized local JSON/JSONL export of official LongBench-v2 records,
selects at most 12 examples, balanced round-robin across source domains.
The manifest is compatible with existing dense-A/dense-B/sparse quality
runner, baseline calibrator and exact-answer scorer.

No source questions, model answers or private data are committed to Git.
This tool never downloads data, executes inference, truncates source contexts,
reorders answer options or alters the correct answer.
Data origin: https://huggingface.co/datasets/zai-org/LongBench-v2

Usage:
 python3 docs/LLAMA-PERF/tools/prepare_qsa_longbench_v2.py \
     /external/LongBench-v2.json --out /external/bench/tasks.json \
     --seed 20261115 --limit 12 --min-context-bytes 56000 \
     --max-prompt-bytes 210000 --difficulty easy

All choices and labels are copied unchanged. "easy" is the source dataset's
category, NOT a claim a model will necessarily answer successfully.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

CHOICES = ("A", "B", "C", "D")
KIND = "benchmark_mcq"
SOURCE = "LongBench-v2"


def _valid_text(value, limit: int) -> bool:
    return isinstance(value, str) and 1 <= len(value.encode("utf-8")) <= limit


def _records(path: Path):
    if not path.is_file() or not 0 < path.stat().st_size <= 600 * 1024 * 1024:
        raise ValueError("dataset file must exist and fit in 600 MiB")
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL record at {lineno}: {exc}") from exc
                yield obj
    else:
        # Official dataset exports are ~465 MiB, and the GPU host already
        # has resources to hold that file. Bound size and task count, and
        # prefer JSONL exports if host memory is constrained.
        with path.open(encoding="utf-8") as handle:
            obj = json.load(handle)
        if isinstance(obj, dict) and isinstance(obj.get("data"), list):
            obj = obj["data"]
        if not isinstance(obj, list):
            raise ValueError("expected dataset JSON list or JSONL objects")
        yield from obj


def _build_prompt(row: dict) -> str:
    options = "\n".join(f"{name}. {row['choice_' + name]}" for name in CHOICES)
    return (
        "Use the supplied long document(s) to answer a multiple-choice question. "
        "Do not rely on outside knowledge when it conflicts with the supplied text.\n"
        "<DOCUMENTS>\n" + row["context"] + "\n</DOCUMENTS>\n"
        "Question: " + row["question"] + "\n" + options + "\n"
        "Output exactly one line: The correct answer is (X)\n"
        "Replace X with exactly one of A, B, C or D. Do not explain."
    )


def make_manifest(source: Path, seed: int = 20261115, limit: int = 12,
                  min_context_bytes: int = 56000, max_prompt_bytes: int = 210000,
                  difficulty: str = "easy") -> dict:
    if not 0 <= seed < 2**32 or not 1 <= limit <= 12:
        raise ValueError("seed must be uint32, limit 1..12")
    if not 1 <= min_context_bytes <= max_prompt_bytes <= 245760:
        raise ValueError("invalid context/prompt length bounds")
    if difficulty not in ("easy", "hard", "any"):
        raise ValueError("difficulty must be easy, hard or any")
    by_domain: dict[str, list[dict]] = {}
    seen = set()
    eligible = 0
    for row in _records(source):
        if not isinstance(row, dict):
            raise ValueError("non-object benchmark record")
        sid = row.get("_id")
        domain = row.get("domain")
        if not isinstance(sid, str) or not sid or len(sid) > 128:
            raise ValueError("missing/invalid benchmark record ID")
        if sid in seen:
            raise ValueError(f"duplicate benchmark record ID {sid}")
        seen.add(sid)
        if not isinstance(domain, str) or not domain or len(domain) > 160:
            raise ValueError(f"missing/invalid benchmark domain {sid}")
        if (row.get("difficulty") not in ("easy", "hard") or
            row.get("length") not in ("short", "medium", "long")):
            raise ValueError(f"invalid difficulty/length bucket at {sid}")
        answer = row.get("answer")
        if not isinstance(answer, str) or answer not in CHOICES:
            raise ValueError(f"invalid MCQ answer at {sid}")
        if not _valid_text(row.get("context"), 1024 * 1024):
            raise ValueError(f"missing/oversized context for {sid}")
        if not _valid_text(row.get("question"), 8192):
            raise ValueError(f"missing/oversized question for {sid}")
        if any(not _valid_text(row.get("choice_" + c), 8192) for c in CHOICES):
            raise ValueError(f"missing/oversized choice for {sid}")
        if difficulty != "any" and row["difficulty"] != difficulty:
            continue
        context_len = len(row["context"].encode("utf-8"))
        if context_len < min_context_bytes:
            continue
        prompt = _build_prompt(row)
        if len(prompt.encode("utf-8")) > max_prompt_bytes:
            continue
        eligible += 1
        row_hash = hashlib.sha256(
            json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        record = {
            "id": f"lb2-{sid}", "kind": KIND,
            "difficulty": "benchmark_" + row["difficulty"],
            "source": SOURCE, "benchmark_id": sid,
            "domain": domain, "sub_domain": row.get("sub_domain", ""),
            "length_class": row["length"],
            "source_row_sha256": row_hash,
            "prompt": prompt,
            "expected": {"type": "choice", "value": answer},
            "max_tokens": 32,
        }
        if len(record["id"]) > 80:
            raise ValueError(f"benchmark task ID too long: {sid}")
        by_domain.setdefault(domain, []).append(record)
    if eligible < limit:
        raise ValueError(
            f"only {eligible} eligible benchmark questions for limit={limit}; "
            "change the source preflight bounds BEFORE any inference"
        )
    # Deterministic domain-stratified draw, no model outputs are inspected.
    # Choose one per domain first, then repeat until limit; within each
    # domain draw in seeded stable order.
    rng = random.Random(seed)
    for rows in by_domain.values():
        rows.sort(key=lambda v: v["benchmark_id"])
        rng.shuffle(rows)
    chosen = []
    domains = sorted(by_domain)
    rng.shuffle(domains)
    while len(chosen) < limit:
        progress = False
        for domain in domains:
            rows = by_domain[domain]
            if rows:
                chosen.append(rows.pop())
                progress = True
                if len(chosen) == limit:
                    break
        if not progress:
            raise ValueError("not enough benchmark rows")
    out = {
        "version": 1,
        "profile": "LongBench-v2-natural-long-context-pilot",
        "source_name": SOURCE,
        "source_file_sha256": _file_hash(source),
        "seed": seed, "requested_difficulty": difficulty,
        "min_context_bytes": min_context_bytes,
        "max_prompt_bytes": max_prompt_bytes,
        "eligible_candidates": eligible,
        "selected_domains": sorted({t["domain"] for t in chosen}),
        "source_dataset_license": "apache-2.0 (dataset card; check original content rights)",
        "note": (
            "Public benchmark, potentially present in model training data; "
            "not a certified held-out or representative production study."
        ),
        "tasks": chosen,
    }
    if len(json.dumps(out, ensure_ascii=False).encode("utf-8")) > 3*1024*1024:
        raise ValueError("output task manifest exceeds 3 MiB runner bound")
    return out


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--min-context-bytes", type=int, default=56000)
    parser.add_argument("--max-prompt-bytes", type=int, default=210000)
    parser.add_argument("--difficulty", choices=("easy", "hard", "any"), default="easy")
    args = parser.parse_args()
    try:
        manifest = make_manifest(
            args.source, args.seed, args.limit, args.min_context_bytes,
            args.max_prompt_bytes, args.difficulty
        )
        with args.out.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except (ValueError, OSError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "tasks": len(manifest["tasks"]),
        "eligible": manifest["eligible_candidates"],
        "domains": manifest["selected_domains"],
        "seed": manifest["seed"], "out": str(args.out)
    }, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
