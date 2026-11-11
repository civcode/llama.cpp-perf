#!/usr/bin/env python3
"""Join same-process ROCPD HIP intervals to MTP host-phase timestamps.

The tool reports AMD GPU kernel/copy *overlap* with host target-sync windows;
overlap is NOT proof of dependency or critical-path contribution. NVIDIA CUDA
activity is not present in ROCprofiler. Profiler and CLOCK_MONOTONIC timestamps
may have different clock origins. This tool refuses unvalidated joins.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import csv
import json
from pathlib import Path
import sqlite3
import statistics

from rocpd_operator_compare import kernel_family
from server_step_trace_analysis import match_intervals, parse_events, quantile


def busy_union_ns(ranges):
    """Elapsed union of execution intervals, not summed concurrent GPU time."""
    if not ranges:
        return 0
    ranges = sorted(ranges)
    start, end = ranges[0]
    busy = 0
    for a, b in ranges[1:]:
        if a <= end:
            end = max(end, b)
        else:
            busy += end - start
            start, end = a, b
    return busy + end - start


def load_kernels(db):
    objects = {name for (name,) in db.execute(
        "SELECT name FROM sqlite_master WHERE type IN ('view','table')")}
    required = {"kernels", "regions_and_samples"}
    if not required.issubset(objects):
        raise ValueError("Missing required ROCPD views: " + ", ".join(sorted(required - objects)))

    # 'kernels' is the ROCprofiler 1.4.1 normalized view verified in Reports 0001-0004.
    result = []
    for agent, start, end, name in db.execute(
            "SELECT agent_abs_index,start,end,name FROM kernels ORDER BY start"):
        if start is None or end is None or end <= start:
            continue
        result.append((int(start), int(end), str(agent), kernel_family(name)))
    if not result:
        raise ValueError("No valid AMD GPU kernel intervals found")
    return result


def clock_bounds(windows, clock, offset_ns=0):
    if clock == "raw":
        if any(w.get("start_raw_ns") is None or w.get("end_raw_ns") is None
               for w in windows):
            raise ValueError("Host trace has no CLOCK_MONOTONIC_RAW samples; rebuild tracer")
        return [(w["start_raw_ns"] + offset_ns, w["end_raw_ns"] + offset_ns)
                for w in windows]
    return [(w["start_ns"] + offset_ns, w["end_ns"] + offset_ns) for w in windows]


def choose_clock(windows, kernels, requested, offset_ns=0):
    begin = kernels[0][0]
    finish = max(x[1] for x in kernels)
    choices = [requested] if requested != "auto" else ["mono", "raw"]
    scored = []
    for clock in choices:
        try:
            bounds = clock_bounds(windows, clock, offset_ns)
        except ValueError:
            if requested != "auto":
                raise
            continue
        hits = sum(a < finish and b > begin for a, b in bounds)
        scored.append((hits, clock))
    if not scored:
        raise ValueError("No usable host clock domain")
    scored.sort(reverse=True)
    best_hits, best = scored[0]
    # Some kernels must occur during actual host waits. A mere overlap with
    # total process lifetime is insufficient for dependable correlation.
    if best_hits < max(4, len(windows) // 2):
        raise ValueError(f"ROCPD and host timestamp domains do not overlap enough "
                         f"(best={best}: {best_hits}/{len(windows)} intervals). "
                         "Check clock origins and matching process before joining.")
    if requested == "auto" and len(scored) > 1 and scored[0][0] == scored[1][0]:
        raise ValueError("Ambiguous MONOTONIC vs RAW overlap; inspect timestamps "
                         "and rerun with --clock mono or --clock raw")
    return best, best_hits


def prefixed_maximum_end(records):
    result = []
    max_end = -1
    for _, end, *_ in records:
        max_end = max(max_end, end)
        result.append(max_end)
    return result


def clipped_for_window(records, starts, ends_prefix, a, b):
    if b <= a:
        return []
    lo = bisect_right(ends_prefix, a)
    hi = bisect_left(starts, b)
    return [(max(start, a), min(end, b), device, family)
            for start, end, device, family in records[lo:hi]
            if end > a and start < b]


def aggregate_overlap(kernels):
    grouped = defaultdict(list)
    family_sum = defaultdict(int)
    for a, b, gpu, family in kernels:
        grouped[gpu].append((a, b))
        family_sum[(gpu, family)] += b - a
    return grouped, family_sum


def load_hip_sync_calls(db, begin, finish):
    """Return HIP host API wait intervals, indexed by Linux OS thread ID.

    HIP API timestamps must already share the calibrated host clock. These
    records do NOT label the GPU/stream waited on; kernel overlap is only a
    temporal association, not an event dependency.
    """
    calls = defaultdict(list)
    for tid, start, end in db.execute("""
        SELECT tid,start,end FROM regions_and_samples
        WHERE name = 'hipStreamSynchronize' AND start < ? AND end > ?
        ORDER BY tid,start
    """, (finish, begin)):
        if tid is None or end is None or start is None or end <= start:
            continue
        calls[int(tid)].append((int(start), int(end)))
    return calls


def wait_breakdown(records, starts, ends_prefix, a, b):
    """Kernel activity clipped to one specific HIP API wait on the host."""
    clipped = clipped_for_window(records, starts, ends_prefix, a, b)
    groups, families = aggregate_overlap(clipped)
    last = max(clipped, key=lambda x: x[1], default=None)
    return {
        "duration_ms": round((b - a)/1e6, 5),
        "gpu_busy_union_ms": {
            gpu: round(busy_union_ns(ranges)/1e6, 5)
            for gpu, ranges in groups.items()
        },
        "gpu_families_ms": {
            gpu: {
                family: round(sum(ns for (dev, fam), ns in families.items()
                                  if dev == gpu and fam == family)/1e6, 5)
                for family in sorted({fam for (dev, fam) in families if dev == gpu})
            }
            for gpu in groups
        },
        "last_amd_agent": last[2] if last else None,
        "last_amd_kernel_family": last[3] if last else None,
        "last_amd_kernel_to_wait_end_ms": round((b-last[1])/1e6, 5) if last else None,
    }


def read_hip_apis(db, begin, finish):
    # Use only known stable view columns. CPU API timestamps reveal *host*
    # activity, not the device kernel a synchronization waited for.
    return [(str(name), int(count), float(ns)/1e6)
            for name, count, ns in db.execute("""
                SELECT name, COUNT(*), SUM(end-start)
                FROM regions_and_samples
                WHERE category LIKE 'HIP_%' AND start < ? AND end > ?
                GROUP BY name ORDER BY SUM(end-start) DESC LIMIT 12
            """, (finish, begin))]


def analyze(host_path, db_path, clock, rows_path, phase="target_sync", offset_ns=0):
    events = parse_events(host_path)
    windows, unpaired = match_intervals(events)
    targets = sorted((w for w in windows
                      if w["phase"] == phase and
                      (w["draft_tokens"] > 0 or phase == "draft")),
                     key=lambda w: w["start_ns"])
    if len(targets) < 4:
        raise ValueError(f"Only {len(targets)} speculative {phase} intervals; "
                         "check MTP and marker capture")
    with sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True) as db:
        kernels = load_kernels(db)
        domain, clock_hits = choose_clock(targets, kernels, clock, offset_ns)
        times = clock_bounds(targets, domain, offset_ns)
        starts = [row[0] for row in kernels]
        prefix_ends = prefixed_maximum_end(kernels)
        sync_api_by_tid = load_hip_sync_calls(
            db, min(a for a, _ in times), max(b for _, b in times))
        timeline = []
        for target, (a, b) in zip(targets, times):
            observed = clipped_for_window(kernels, starts, prefix_ends, a, b)
            gpu_ranges, by_family = aggregate_overlap(observed)
            gpu_busy = {gpu: busy_union_ns(ranges)/1e6
                        for gpu, ranges in gpu_ranges.items()}
            latest_gpu_end = max((end for _, end, _, _ in observed), default=None)
            waits = []
            for start_api, end_api in sync_api_by_tid.get(target["start_tid"], []):
                aa, bb = max(a, start_api), min(b, end_api)
                if bb <= aa:
                    continue
                breakdown = wait_breakdown(kernels, starts, prefix_ends, aa, bb)
                breakdown["start_offset_ms"] = round((aa-a)/1e6, 5)
                waits.append(breakdown)
            waits.sort(key=lambda w: w["start_offset_ms"])
            timeline.append({
                "step_before": target["step_before"],
                "start_ns": a,
                "end_ns": b,
                "host_sync_ms": target["duration_ms"],
                "kernels_clipped": len(observed),
                "hip_stream_sync_waits": waits,
                "amd_gpu_union_ms": gpu_busy,
                "amd_flash_ms": {gpu: round(ns/1e6, 4)
                                 for (gpu, family), ns in by_family.items()
                                 if family == "flash_attention"},
                "last_amd_kernel_end_minus_sync_end_ms":
                    (latest_gpu_end - b)/1e6 if latest_gpu_end is not None else None,
            })
        begin, finish = min(a for a, _ in times), max(b for _, b in times)
        apis = read_hip_apis(db, begin, finish)

    if rows_path:
        rows_path.parent.mkdir(parents=True, exist_ok=True)
        with rows_path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "step_before", "start_ns", "end_ns", "host_sync_ms",
                "kernels_clipped", "hip_stream_sync_waits", "amd_gpu_union_ms", "amd_flash_ms",
                "last_amd_kernel_end_minus_sync_end_ms",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for item in timeline:
                writer.writerow({key: json.dumps(item[key]) if isinstance(item[key], dict)
                                 else item[key] for key in fields})

    # A pair of HIP waits is common for the two Radeon backends, but this
    # analysis never assumes fixed call count or assigns an API to a device.
    wait_count_histogram = defaultdict(int)
    waits_by_position = defaultdict(list)
    for step in timeline:
        wait_count_histogram[len(step["hip_stream_sync_waits"])] += 1
        for position, call in enumerate(step["hip_stream_sync_waits"], 1):
            waits_by_position[position].append(call)
    wait_position_summary = {}
    for position, calls in sorted(waits_by_position.items()):
        wait_position_summary[str(position)] = {
            "observed_steps": len(calls),
            "median_api_duration_ms": statistics.median(w["duration_ms"] for w in calls),
            "median_offset_ms": statistics.median(w["start_offset_ms"] for w in calls),
            "last_amd_agent_counts": dict(
                (agent, sum(1 for w in calls if w["last_amd_agent"] == agent))
                for agent in sorted({w["last_amd_agent"] for w in calls
                                     if w["last_amd_agent"] is not None})),
            "per_gpu_median_busy_overlap_ms": {
                agent: statistics.median(
                    w["gpu_busy_union_ms"].get(agent, 0.0) for w in calls)
                for agent in sorted({agent for w in calls for agent
                                     in w["gpu_busy_union_ms"]})
            },
            "per_gpu_median_flash_overlap_ms": {
                agent: statistics.median(
                    w["gpu_families_ms"].get(agent, {}).get("flash_attention", 0.0)
                    for w in calls)
                for agent in sorted({agent for w in calls for agent
                                     in w["gpu_families_ms"]})
            },
        }

    devices = sorted({gpu for w in timeline for gpu in w["amd_gpu_union_ms"]})
    aggregate = {}
    for gpu in devices:
        busy = [w["amd_gpu_union_ms"].get(gpu, 0.0) for w in timeline]
        flash = [w["amd_flash_ms"].get(gpu, 0.0) for w in timeline]
        aggregate[gpu] = {
            "median_amd_busy_overlap_ms": statistics.median(busy),
            "p90_amd_busy_overlap_ms": quantile(busy, .9),
            "median_flash_overlap_ms": statistics.median(flash),
        }

    return {
        "host_trace": str(host_path),
        "rocpd_db": str(db_path),
        "clock_used": domain,
        "clock_offset_ns": offset_ns,
        "clock_intersections": clock_hits,
        "phase": phase,
        "host_sync_windows": len(targets),
        "unpaired_host_events": len(unpaired),
        "median_host_sync_ms": statistics.median(w["duration_ms"] for w in targets),
        "per_gpu_amd_kernel_overlap": aggregate,
        "hip_sync_wait_counts_per_step": dict(sorted(wait_count_histogram.items())),
        "hip_sync_wait_positions": wait_position_summary,
        "top_hip_host_apis": [{"name": name, "count": count, "sum_ms": round(ms, 3)}
                              for name, count, ms in apis],
        "steps": timeline,
        "caveat": "AMD overlap only; not a CUDA timeline, a GPU dependency proof or a causal latency decomposition",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host_tsv", type=Path)
    parser.add_argument("rocpd_db", type=Path)
    parser.add_argument("--clock", choices=("auto", "mono", "raw"), default="auto")
    parser.add_argument("--clock-offset-ns", type=int, default=0,
                        help="Explicit calibrated offset: ROCPD timestamp = host timestamp + offset. "
                             "Never guess this number.")
    parser.add_argument("--phase", choices=("target_sync", "target_process",
                                            "spec_reprocess", "post_decode", "draft"),
                        default="target_sync")
    parser.add_argument("--rows-out", type=Path, help="Write per-verification sync windows CSV")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    try:
        report = analyze(args.host_tsv, args.rocpd_db, args.clock, args.rows_out,
                         args.phase, args.clock_offset_ns)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))

    print(f"Clock: {report['clock_used']} with {report['clock_intersections']}/"
          f"{report['host_sync_windows']} windows inside GPU capture")
    print(f"{report['phase']} host median: {report['median_host_sync_ms']:.3f} ms "
          f"({report['host_sync_windows']} speculative windows)")
    for gpu, info in report["per_gpu_amd_kernel_overlap"].items():
        print(f"  GPU {gpu}: median busy overlap "
              f"{info['median_amd_busy_overlap_ms']:.3f} ms, "
              f"median flash-attention overlap {info['median_flash_overlap_ms']:.3f} ms")
    print("HIP stream synchronizations inside each phase:")
    print(f"  Wait count distribution: {report['hip_sync_wait_counts_per_step']}")
    for position, info in report["hip_sync_wait_positions"].items():
        print(f"  Wait #{position}: median {info['median_api_duration_ms']:.3f} ms, "
              f"AMD tail agents {info['last_amd_agent_counts']}; "
              f"median GPU busy overlap {info['per_gpu_median_busy_overlap_ms']}")
    print("IMPORTANT: HIP host API position does not by itself establish which "
          "GPU stream or backend was synchronized.")
    print("Top HIP host APIs:")
    for api in report["top_hip_host_apis"][:5]:
        print(f"  {api['name']}: {api['count']} calls, {api['sum_ms']:.1f} ms")
    print(f"Clock offset (ns): {report['clock_offset_ns']}")
    print("CLOCK VALIDATION REQUIRED: corroborate selected domain and offset "
          "against actual HIP API timestamps, process/thread IDs, and known marker "
          "boundaries before inferring device/host relationships.")
    print("CAUTION: " + report["caveat"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
