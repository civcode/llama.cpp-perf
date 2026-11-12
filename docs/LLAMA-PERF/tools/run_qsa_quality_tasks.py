#!/usr/bin/env python3
"""Run deterministic task-quality prompts against an already-started LOCAL llama-server.

Runs one named condition per process. Does not start, stop, configure or time
GPU inference; agent owns separate dense A/B and sparse server processes.
No arbitrary model output is executed. All prompts/answers remain outside Git.

 python3 docs/LLAMA-PERF/tools/run_qsa_quality_tasks.py \
   /external/tasks.json --condition dense-a --url http://127.0.0.1:8080 \
   --out /external/dense-a.jsonl

Only localhost URLs are supported. Existing output files are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from generate_qsa_quality_tasks import KINDS


CONDITIONS = ("dense-a", "dense-b", "sparse")
SYSTEM_PROMPT = (
    "You are answering a deterministic functional evaluation. "
    "Use the user-provided fictional data and instructions exactly. "
    "Output only the requested value or JSON; no explanation or code fences."
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_tasks(path: Path) -> tuple[str, list[dict]]:
    data = path.read_bytes()
    if not 1 <= len(data) <= 3 * 1024 * 1024:
        raise ValueError("task manifest exceeds 3 MiB or is empty")
    obj = json.loads(data)
    if not isinstance(obj, dict) or obj.get("version") != 1 or not isinstance(obj.get("tasks"), list):
        raise ValueError("expected version-1 task manifest")
    tasks = obj["tasks"]
    if not 1 <= len(tasks) <= 12:
        raise ValueError("expected 1..12 tasks")
    seen = set()
    for t in tasks:
        if not isinstance(t, dict):
            raise ValueError("malformed task object")
        key = t.get("id")
        if not isinstance(key, str) or not key or len(key) > 80 or key in seen:
            raise ValueError("invalid/duplicate task identifier")
        seen.add(key)
        if t.get("kind") not in KINDS:
            raise ValueError(f"invalid task kind {key}")
        prompt = t.get("prompt")
        if not isinstance(prompt, str) or not (1 <= len(prompt.encode("utf-8")) <= 245760):
            raise ValueError(f"missing/oversized task prompt {key}")
        expected = t.get("expected")
        if not isinstance(expected, dict) or expected.get("type") not in ("literal", "json"):
            raise ValueError(f"invalid grading rubric for {key}")
        max_tokens = t.get("max_tokens")
        if type(max_tokens) is not int or not (1 <= max_tokens <= 256):
            raise ValueError(f"invalid token budget for {key}")
    return sha256(data), tasks


def _localhost_api(url: str) -> str:
    parts = urlsplit(url)
    if (parts.scheme != "http" or parts.hostname not in ("localhost", "127.0.0.1", "::1")
            or parts.username or parts.password or parts.query or parts.fragment
            or parts.path not in ("", "/")):
        raise ValueError("server URL must be an http://localhost:<port> root")
    if parts.port is None or not 1 <= parts.port <= 65535:
        raise ValueError("server URL requires a port")
    return url.rstrip("/") + "/v1/chat/completions"


def request_completion(url: str, prompt: str, max_tokens: int, timeout: int = 600) -> dict:
    payload = {
        "model": "local",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": False,
    }
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=encoded, method="POST",
                  headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
    except (HTTPError, URLError, TimeoutError, socket.timeout) as err:
        raise RuntimeError(f"local server request failed: {err}") from err
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("response exceeds 4 MiB")
    message = json.loads(raw)
    if not isinstance(message, dict):
        raise ValueError("server returned non-object response")
    choices = message.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("response must contain exactly one choice")
    choice = choices[0]
    output = choice.get("message", {}).get("content")
    if not isinstance(output, str) or len(output.encode("utf-8")) > 16384:
        raise ValueError("missing or excessively long model output")
    finish = choice.get("finish_reason")
    if finish not in ("stop", "length"):
        raise ValueError(f"unexpected completion finish reason {finish!r}")
    usage = message.get("usage")
    if usage is None:
        usage = {}
    if not isinstance(usage, dict):
        raise ValueError("invalid response usage")
    prompt_tokens = usage.get("prompt_tokens")
    if prompt_tokens is not None and (type(prompt_tokens) is not int or prompt_tokens < 0):
        raise ValueError("invalid prompt_tokens usage")
    return {
        "text": output,
        "finish_reason": finish,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": usage.get("completion_tokens"),
        "response_model": message.get("model"),
    }


def run(tasks_file: Path, url: str, condition: str, output: Path,
        timeout: int = 600) -> int:
    if condition not in CONDITIONS:
        raise ValueError("condition must be dense-a, dense-b or sparse")
    if not 10 <= timeout <= 1800:
        raise ValueError("timeout must be 10..1800 seconds")
    manifest_sha, tasks = load_tasks(tasks_file)
    endpoint = _localhost_api(url)
    with output.open("x", encoding="utf-8") as out:
        for task in tasks:
            # Local monotonic markers correlate each request with the backend's
            # QSA audit timestamps; they are not a throughput measurement.
            start_ns = time.monotonic_ns()
            result = request_completion(endpoint, task["prompt"], task["max_tokens"], timeout)
            end_ns = time.monotonic_ns()
            record = {
                "id": task["id"],
                "kind": task["kind"],
                "condition": condition,
                "task_manifest_sha256": manifest_sha,
                "prompt_sha256": sha256(task["prompt"].encode("utf-8")),
                "requested_max_tokens": task["max_tokens"],
                "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 42},
                "request_start_monotonic_ns": start_ns,
                "request_end_monotonic_ns": end_ns,
                **result,
            }
            out.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            out.flush()
            print(f"{condition}: {task['id']} complete, "
                  f"prompt_tokens={result['prompt_tokens']}, finish={result['finish_reason']}")
    return len(tasks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--url", required=True, help="localhost llama-server root URL")
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    try:
        count = run(args.manifest, args.url, args.condition, args.out, args.timeout)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, json.JSONDecodeError) as err:
        parser.error(str(err))
    print(f"Saved {count} task outputs to {args.out} (not a throughput benchmark)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
