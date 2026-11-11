#!/usr/bin/env python3
"""Read-only ROCprofiler-SDK 1.4.x sync/transfer summary for a chosen time window.

Uses stable ROCPD SQLite *views*, not UUID-suffixed storage tables.
The 0-second origin is the earliest recorded AMD GPU kernel dispatch.
The report identifies correlates, not removable wall-time overhead.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def union_duration(intervals):
    merged = []
    for begin, end in sorted(intervals):
        if end <= begin:
            continue
        if not merged or begin > merged[-1][1]:
            merged.append([begin, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - begin for begin, end in merged)


def rank_counter(counter, n=12):
    return [{"name": key, "calls": count} for key, count in counter.most_common(n)]


def analyze(conn, start_seconds=6.0, end_seconds=None):
    available = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
    )}
    required = {"kernels", "regions_and_samples", "rocpd_memory_copy", "rocpd_info_agent"}
    missing = required - available
    if missing:
        raise ValueError(f"Missing ROCPD tables/views: {', '.join(sorted(missing))}")

    first, last = conn.execute("SELECT MIN(start), MAX(end) FROM kernels").fetchone()
    if first is None or last is None:
        raise ValueError("No GPU kernel events in the database")
    window_start = first + round(start_seconds * 1_000_000_000)
    window_end = last if end_seconds is None else min(last, first + round(end_seconds * 1_000_000_000))
    if window_end <= window_start:
        raise ValueError("Invalid time window; --start-seconds must be before the trace ends")
    duration = window_end - window_start

    agents = {r[0]: {"name": r[1], "type": r[2], "absolute_index": r[3]}
              for r in conn.execute("SELECT id, name, type, absolute_index FROM rocpd_info_agent")}

    kernel_summary = []
    for agent_abs, calls, gpu_ns in conn.execute("""
        SELECT agent_abs_index, COUNT(*), SUM(duration) FROM kernels
        WHERE start >= ? AND start < ? GROUP BY agent_abs_index ORDER BY agent_abs_index
    """, (window_start, window_end)):
        top_kernels = conn.execute("""
            SELECT name, COUNT(*), SUM(duration) FROM kernels
            WHERE agent_abs_index = ? AND start >= ? AND start < ?
            GROUP BY name ORDER BY SUM(duration) DESC LIMIT 10
        """, (agent_abs, window_start, window_end)).fetchall()
        kernel_summary.append({"agent_abs_index": agent_abs, "calls": calls,
                               "accumulated_ms": round(gpu_ns / 1e6, 3),
                               "top": [{"name": name, "calls": k_calls, "ms": round(ns / 1e6, 3)}
                                       for name, k_calls, ns in top_kernels]})

    raw_syncs = conn.execute("""
        SELECT tid, start, end, category FROM regions_and_samples
        WHERE name = 'hipStreamSynchronize' AND start >= ? AND start < ? AND end > start
        ORDER BY tid, start
    """, (window_start, window_end)).fetchall()
    per_thread = defaultdict(list)
    sync_intervals = []
    categories = Counter()
    for tid, begin, end, category in raw_syncs:
        categories[category] += 1
        clipped = (max(begin, window_start), min(end, window_end))
        if clipped[1] > clipped[0]:
            per_thread[tid].append(clipped)
            sync_intervals.append(clipped)

    threads = []
    for tid, intervals in per_thread.items():
        lengths = sorted(end - begin for begin, end in intervals)
        total = sum(lengths)
        p95 = lengths[max(0, (95 * len(lengths) + 99) // 100 - 1)]
        threads.append({"tid": tid, "calls": len(lengths),
                        "total_ms": round(total / 1e6, 3),
                        "avg_us": round(total / len(lengths) / 1e3, 3),
                        "p95_us": round(p95 / 1e3, 3),
                        "max_ms": round(max(lengths) / 1e6, 3),
                        "window_pct": round(100 * total / duration, 2)})
    threads.sort(key=lambda t: t["total_ms"], reverse=True)

    total_sync_ns = sum(end - begin for begin, end in sync_intervals)
    union_sync_ns = union_duration(sync_intervals)

    neighbors = []
    # Only read sequences for threads with synchronizations.
    for tid in per_thread:
        calls = conn.execute("""
            SELECT name, start, end FROM regions_and_samples
            WHERE tid = ? AND start >= ? AND start < ?
            ORDER BY start, id
        """, (tid, window_start, window_end)).fetchall()
        pattern_count = Counter()
        pattern_durations = defaultdict(list)
        for idx, (name, start, end) in enumerate(calls):
            if name != "hipStreamSynchronize":
                continue
            before = calls[idx-1][0] if idx > 0 else "<START>"
            after = calls[idx+1][0] if idx + 1 < len(calls) else "<END>"
            label = f"{before} -> hipStreamSynchronize -> {after}"
            pattern_count[label] += 1
            pattern_durations[label].append(max(0, min(end, window_end) - start))
        neighbors.append({"tid": tid, "patterns": [
            {"sequence": r["name"], "calls": r["calls"],
             "avg_sync_us": round(statistics.mean(pattern_durations[r["name"]]) / 1e3, 3),
             "sum_sync_ms": round(sum(pattern_durations[r["name"]]) / 1e6, 3)}
            for r in rank_counter(pattern_count)]})

    copy_groups = defaultdict(lambda: [0, 0, 0])
    copy_sizes = defaultdict(Counter)
    for src, dst, size, begin, end in conn.execute("""
        SELECT src_agent_id, dst_agent_id, size, start, end FROM rocpd_memory_copy
        WHERE start >= ? AND start < ?
    """, (window_start, window_end)):
        group = copy_groups[(src, dst)]
        group[0] += 1
        group[1] += size or 0
        group[2] += max(0, min(end, window_end) - begin)
        copy_sizes[(src, dst)][size] += 1
    copies = [{"src": src, "dst": dst, "calls": counts[0],
               "mib": round(counts[1] / 2**20, 3),
               "accumulated_ms": round(counts[2] / 1e6, 3),
               "most_common_sizes_bytes": copy_sizes[(src, dst)].most_common(3)}
              for (src, dst), counts in sorted(copy_groups.items(), key=lambda kv: kv[1][2], reverse=True)]

    api_top = conn.execute("""
        SELECT name, COUNT(*), SUM(end - start) AS ns
        FROM regions_and_samples
        WHERE category LIKE 'HIP_%' AND start >= ? AND start < ? AND end > start
        GROUP BY name ORDER BY ns DESC LIMIT 15
    """, (window_start, window_end)).fetchall()

    return {
        "window": {"start_seconds": start_seconds,
                   "end_seconds": round((window_end-first)/1e9, 6),
                   "duration_ms": round(duration/1e6, 3)},
        "agents": agents,
        "kernel_summary": kernel_summary,
        "sync": {"calls": len(sync_intervals), "categories": dict(categories),
                 "sum_ms": round(total_sync_ns/1e6, 3),
                 "union_ms": round(union_sync_ns/1e6, 3),
                 "union_window_pct": round(100*union_sync_ns/duration, 2),
                 "threads": threads},
        "neighbor_patterns": neighbors,
        "memory_copies": copies,
        "top_hip_apis": [{"name": name, "calls": calls, "accumulated_ms": round(ns/1e6, 3)}
                         for name, calls, ns in api_top],
    }


def render(summary):
    print("=== ROCPD SYNCHRONIZATION ANALYSIS ===")
    w = summary["window"]
    print(f"Window: {w['start_seconds']:.3f} to {w['end_seconds']:.3f} s after first GPU kernel; {w['duration_ms']:.1f} ms")
    print("\n=== GPU AGENTS ===")
    for aid, agent in sorted(summary["agents"].items(), key=lambda item: int(item[0])):
        print(f"{aid}: {agent['type']} {agent['name']} (absolute {agent['absolute_index']})")
    print("\n=== GPU WORK (ACCUMULATED, NOT CRITICAL PATH) ===")
    for gpu in summary["kernel_summary"]:
        print(f"Agent {gpu['agent_abs_index']}: {gpu['calls']} kernels, {gpu['accumulated_ms']} ms")
        for k in gpu["top"][:5]:
            print(f"  {k['ms']:8.3f} ms | {k['calls']:6} | {k['name'][:105]}")
    s = summary["sync"]
    print("\n=== HIP STREAM SYNCHRONIZATION ===")
    print(f"Calls: {s['calls']} | sum: {s['sum_ms']} ms | union: {s['union_ms']} ms "
          f"| window: {s['union_window_pct']}% | categories: {s['categories']}")
    for t in s["threads"][:10]:
        print(f"TID {t['tid']}: {t['calls']} calls, sum {t['total_ms']} ms, "
              f"average {t['avg_us']} us, p95 {t['p95_us']} us, max {t['max_ms']} ms")
    print("\n=== IMMEDIATE SAME-THREAD API NEIGHBORS ===")
    for n in summary["neighbor_patterns"][:3]:
        print(f"TID {n['tid']}")
        for p in n["patterns"][:12]:
            print(f"  {p['calls']:5} | avg sync {p['avg_sync_us']:8.2f} us "
                  f"| total sync {p['sum_sync_ms']:8.2f} ms | {p['sequence'][:150]}")
    print("\n=== MEMORY COPIES ===")
    for c in summary["memory_copies"]:
        print(f"{c['src']} -> {c['dst']}: {c['calls']} copies, {c['mib']} MiB, "
              f"{c['accumulated_ms']} ms, common sizes {c['most_common_sizes_bytes']}")
    print("\n=== HIP API DURATION LEADERS ===")
    for api in summary["top_hip_apis"][:10]:
        print(f"{api['calls']:6} | {api['accumulated_ms']:10.3f} ms | {api['name']}")
    print("\nCAVEAT: profile covers only AMD/HIP, not NVIDIA CUDA. API blocking includes GPU completion ")
    print("work; synchronization duration is NOT automatically removable overhead.")
    print("Kernel and memcpy durations overlap with other GPU/CPU activities.")
    print("A neighboring API does not establish a causal dependency or C++ callsite.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Path to *_results.db")
    parser.add_argument("--start-seconds", type=float, default=6.0, help="Window start after first AMD GPU kernel (default: 6)")
    parser.add_argument("--end-seconds", type=float, help="Optional end after first GPU kernel; default: last kernel")
    parser.add_argument("--json", type=Path, help="Optional destination for compact JSON summary")
    args = parser.parse_args()
    if not args.db.is_file():
        parser.error(f"Database does not exist: {args.db}")
    try:
        with sqlite3.connect(f"{args.db.resolve().as_uri()}?mode=ro", uri=True) as conn:
            result = analyze(conn, args.start_seconds, args.end_seconds)
    except (sqlite3.DatabaseError, ValueError) as error:
        parser.error(str(error))
    render(result)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
