#!/usr/bin/env python3
"""CPU-only integrity preflight for the interrupted Iteration 0028 LongBench run.

This script NEVER imports CUDA/ROCm modules, invokes model inference, probes
hardware, recovers devices, or starts processes. It verifies pinned source
dataset, prior and current manifests, full dense A/B control outputs and
optionally historical *incomplete* sparse JSONL. The latter is diagnostic
ONLY: DO NOT append or reuse prefix outputs in the new sparse full run.

Use only on the actual external artifacts; nothing is downloaded:
 python3 docs/LLAMA-PERF/tools/preflight_qsa_0028_resume.py \
   --source /external/0027/data.json \
   --prior-manifest /external/0027/tasks.json \
   --manifest /external/0028/tasks.json \
   --dense-a /external/0028/dense-a.jsonl \
   --dense-b /external/0028/dense-b.jsonl \
   --failed-sparse /external/0028/sparse.jsonl \
   --out /external/0028/offline-preflight.json

A green *offline* integrity check is NOT permission to start GPU activity.
Xid 79/154 requires authorized operator recovery and independent device and
driver validation before any new CUDA model load. On a healthy recovered host
use a FRESH sparse process and FRESH output path; score only its FULL 24 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from run_qsa_quality_tasks import _localhost_api, load_tasks, sha256
from score_qsa_quality_tasks import _read_results, grade

SOURCE_SHA256 = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"
PRIOR_SHA256 = "d9c55048469791d1a1b2e106a0609c8d5fca9aa4b65293623e47543d224643c1"
MANIFEST_SHA256 = "ed01ff9392f56c2e0c1e9af35c342e8cc565c71f4c8880df496469d9105d6fb5"
FROZEN_TASK_COUNT = 24
FROZEN_DENSE_SOLVED = 14
FROZEN_SOURCE_COUNT_EXCLUSIONS = 12


def hash_file(path: Path, max_bytes: int) -> str:
    if not path.is_file() or not 0 < path.stat().st_size <= max_bytes:
        raise ValueError(f"missing, empty or oversized artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pinned_manifest(path: Path, expected: str) -> dict:
    digest = hash_file(path, 12*1024*1024)
    if digest != expected:
        raise ValueError(f"frozen manifest digest mismatch: {path} {digest}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or obj.get("source_name") != "LongBench-v2":
        raise ValueError("expected a LongBench-v2 frozen manifest")
    return obj


def partial_sparse_prefix(path: Path, tasks: list[dict], manifest_digest: str,
                          dense_a: dict) -> dict:
    """Validate a historical, fully written prefix; never coalesce runs."""
    if not path.is_file() or path.stat().st_size > 2*1024*1024:
        raise ValueError("missing/oversized failed sparse output")
    found = []
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) >= len(tasks):
        raise ValueError("historical sparse file must be a proper incomplete prefix")
    for i, line in enumerate(lines):
        if not line.strip():
            raise ValueError(f"blank historical sparse record at {i}")
        try:
            row = json.loads(line)
            task = tasks[i]
            if not isinstance(row, dict) or row.get("id") != task["id"]:
                raise ValueError("historical task order or identity changed")
            if (row.get("condition") != "sparse" or
                    row.get("kind") != task["kind"] or
                    row.get("task_manifest_sha256") != manifest_digest or
                    row.get("prompt_sha256") != sha256(task["prompt"].encode("utf-8")) or
                    row.get("requested_max_tokens") != task["max_tokens"] or
                    row.get("sampling") != {
                        "temperature": 0.0, "top_p": 1.0, "seed": 42,
                    }):
                raise ValueError("historical sparse request/manifest mismatch")
            if (not isinstance(row.get("text"), str) or
                    row.get("finish_reason") not in ("stop", "length") or
                    type(row.get("prompt_tokens")) is not int or
                    row["prompt_tokens"] < 0):
                raise ValueError("malformed historical sparse completion")
            d = dense_a[task["id"]]
            found.append({
                "id": task["id"],
                "dense_a_correct": grade(task, d["text"], d["finish_reason"])[0],
                "historical_sparse_correct": grade(
                    task, row["text"], row["finish_reason"])[0],
                "exact_text_match": row["text"] == d["text"],
                "prompt_tokens_match": row["prompt_tokens"] == d["prompt_tokens"],
            })
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as err:
            raise ValueError(f"failed sparse JSONL row {i + 1}: {err}") from err
    return {
        "historical_completed_prefix_rows": len(found),
        "historical_not_a_full_run": True,
        "never_merge_with_new_sparse_results": True,
        "observed_prefix": found,
    }


def preflight(source: Path, prior: Path, manifest: Path,
              dense_a: Path, dense_b: Path,
              failed_sparse: Path | None = None) -> dict:
    source_digest = hash_file(source, 600*1024*1024)
    if source_digest != SOURCE_SHA256:
        raise ValueError("official 0027 dataset SHA-256 mismatch")
    prior_obj = _load_pinned_manifest(prior, PRIOR_SHA256)
    obj = _load_pinned_manifest(manifest, MANIFEST_SHA256)
    if prior_obj.get("source_file_sha256") != source_digest or (
            obj.get("source_file_sha256") != source_digest):
        raise ValueError("frozen manifests do not match publisher dataset SHA-256")
    digest, tasks = load_tasks(manifest)
    if digest != MANIFEST_SHA256 or len(tasks) != FROZEN_TASK_COUNT:
        raise ValueError("frozen 0028 task count/hash mismatch")
    old_tasks = prior_obj.get("tasks")
    if not isinstance(old_tasks, list) or len(old_tasks) != FROZEN_SOURCE_COUNT_EXCLUSIONS:
        raise ValueError("0027 frozen task count differs")
    prior_ids = {r.get("benchmark_id") for r in old_tasks if isinstance(r, dict)}
    new_ids = {r.get("benchmark_id") for r in tasks}
    if (len(prior_ids) != FROZEN_SOURCE_COUNT_EXCLUSIONS or
            len(new_ids) != FROZEN_TASK_COUNT or
            not all(isinstance(i, str) and i for i in prior_ids | new_ids) or
            prior_ids & new_ids):
        raise ValueError("invalid source IDs or overlap with previous sample")
    if obj.get("excluded_source_ids") != sorted(prior_ids):
        raise ValueError("0028 exclusions differ from 0027 frozen source IDs")
    for task in tasks:
        if (task.get("source") != "LongBench-v2" or
                task.get("kind") != "benchmark_mcq" or
                task.get("id") != "lb2-" + task["benchmark_id"] or
                task.get("expected", {}).get("type") != "choice" or
                task["expected"].get("value") not in ("A", "B", "C", "D")):
            raise ValueError(f"invalid frozen MCQ task: {task.get('id')}")
    a = _read_results(dense_a, "dense-a", digest, tasks)
    b = _read_results(dense_b, "dense-b", digest, tasks)
    solved = 0
    token_counts = []
    unstable = []
    for task in tasks:
        key = task["id"]
        ra, rb = a[key], b[key]
        if (ra["text"] != rb["text"] or
                ra["finish_reason"] != rb["finish_reason"] or
                type(ra["prompt_tokens"]) is not int or
                ra["prompt_tokens"] != rb["prompt_tokens"]):
            unstable.append(key)
            continue
        if ra["prompt_tokens"] < 16408 or ra["prompt_tokens"] + task["max_tokens"] > 65536:
            raise ValueError(f"0028 task context not in sparse eligibility window: {key}")
        token_counts.append(ra["prompt_tokens"])
        solved += int(grade(task, ra["text"], ra["finish_reason"])[0])
    if unstable or solved != FROZEN_DENSE_SOLVED or len(token_counts) != FROZEN_TASK_COUNT:
        raise ValueError(
            f"0028 dense repeat evidence differs: unstable={unstable}, solved={solved}")
    result = {
        "read_only_cpu_preflight": True,
        "official_dataset_sha256": source_digest,
        "previous_manifest_sha256": PRIOR_SHA256,
        "frozen_manifest_sha256": digest,
        "frozen_questions": len(tasks),
        "excluded_previous_source_ids": len(prior_ids),
        "dense_repeat_identical": True,
        "stable_dense_correct": solved,
        "min_chat_tokens": min(token_counts),
        "max_chat_tokens": max(token_counts),
        "gpu_health_verified": False,
        "host_recovery_authorized": False,
        "sparse_full_run_available": False,
        "safe_to_resume_inference": False,
        "next_step": (
            "No inference unless an operator has independently restored and "
            "verified NVIDIA GPU/driver health after Xid 79/154. Then launch a "
            "new bounded sparse process for ALL 24 frozen questions, using a "
            "new empty output path. Never merge prior incomplete sparse rows."
        ),
    }
    if failed_sparse is not None:
        result["historical_sparse"] = partial_sparse_prefix(
            failed_sparse, tasks, digest, a)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dense-a", type=Path, required=True)
    parser.add_argument("--dense-b", type=Path, required=True)
    parser.add_argument("--failed-sparse", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = preflight(args.source, args.prior_manifest, args.manifest,
                           args.dense_a, args.dense_b, args.failed_sparse)
        with args.out.open("x", encoding="utf-8") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as err:
        parser.error(str(err))
    print(json.dumps({
        "frozen_questions": result["frozen_questions"],
        "stable_dense_correct": result["stable_dense_correct"],
        "gpu_health_verified": False,
        "safe_to_resume_inference": False,
        "preflight_report": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
