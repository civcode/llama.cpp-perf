#!/usr/bin/env python3
"""Summarize LLAMA_SERVER_STEP_TRACE buffered host markers.

Output is a host-side breakdown of draft generation, target submission,
target sync, speculative reprocessing, post-decode and sampling. These intervals
DO NOT measure GPU kernel durations, and the host stage sums may overlap other
device work. Times are CLOCK_MONOTONIC ns on Linux; validate time domains
before overlaying intervals on ROCprofiler/Nsight traces.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics
import sys


PAIRS = {
    "draft_begin": "draft_end",
    "target_process_begin": "target_process_end",
    "target_sync_begin": "target_sync_end",
    "spec_reprocess_begin": "spec_reprocess_end",
    "post_decode_begin": "post_decode_end",
    "verification_sample_begin": "verification_sample_end",
}
COLUMNS = {
    "mono_ns", "os_tid", "phase", "slot_id",
    "verification_steps_before", "generation_tokens_before",
    "batch_tokens", "draft_tokens",
}
PHASE_NAMES = {name.removesuffix("_begin"): (name, end) for name, end in PAIRS.items()}


def parse_events(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not reader.fieldnames or not COLUMNS.issubset(reader.fieldnames):
            raise ValueError("Missing required LLAMA_SERVER_STEP_TRACE TSV headers")
        records = []
        for line_no, row in enumerate(reader, 2):
            try:
                records.append({
                    "mono_ns": int(row["mono_ns"]),
                    "raw_ns": int(row["raw_ns"]) if row.get("raw_ns") else None,
                    "os_tid": int(row["os_tid"]),
                    "phase": row["phase"],
                    "slot_id": int(row["slot_id"]),
                    "verification_steps_before": int(row["verification_steps_before"]),
                    "generation_tokens_before": int(row["generation_tokens_before"]),
                    "batch_tokens": int(row["batch_tokens"]),
                    "draft_tokens": int(row["draft_tokens"]),
                })
            except (TypeError, ValueError) as err:
                raise ValueError(f"Bad event data at line {line_no}: {err}") from err
    return sorted(records, key=lambda r: (r["mono_ns"], r["os_tid"]))


def match_intervals(events):
    # Each CPU source operation has a single begin/end pair for each slot;
    # pairing by phase avoids the step counter increment in post_decode().
    active = defaultdict(list)
    windows, unpaired = [], []
    for event in events:
        phase = event["phase"]
        if phase in PAIRS:
            active[(event["slot_id"], phase)].append(event)
            continue
        begin = next((name for name, end in PAIRS.items() if end == phase), None)
        if begin is None:
            continue
        key = (event["slot_id"], begin)
        if not active[key]:
            unpaired.append(event)
            continue
        first = active[key].pop()
        duration_ns = event["mono_ns"] - first["mono_ns"]
        if duration_ns < 0:
            unpaired.append(event)
            continue
        windows.append({
            "phase": begin.removesuffix("_begin"),
            "slot_id": first["slot_id"],
            "step_before": first["verification_steps_before"],
            "generation_tokens_before": first["generation_tokens_before"],
            "batch_tokens": first["batch_tokens"],
            "draft_tokens": first["draft_tokens"],
            "start_ns": first["mono_ns"],
            "end_ns": event["mono_ns"],
            "start_raw_ns": first.get("raw_ns"),
            "end_raw_ns": event.get("raw_ns"),
            "start_tid": first["os_tid"],
            "end_tid": event["os_tid"],
            "duration_ms": duration_ns / 1e6,
        })
    for incomplete in active.values():
        unpaired.extend(incomplete)
    return sorted(windows, key=lambda w: w["start_ns"]), unpaired


def quantile(values, q):
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(xs) - 1)
    return xs[lower] + (xs[upper] - xs[lower]) * (pos - lower)


def analyze(windows, only_speculative):
    selected = [
        w for w in windows
        if not only_speculative or w["phase"] == "draft"
        or w["draft_tokens"] > 0
    ]
    groups = defaultdict(list)
    for w in selected:
        groups[w["phase"]].append(w["duration_ms"])
    summary = {}
    for phase in PHASE_NAMES:
        ms = groups.get(phase, [])
        if not ms:
            continue
        summary[phase] = {
            "count": len(ms),
            "mean_ms": statistics.mean(ms),
            "median_ms": statistics.median(ms),
            "p90_ms": quantile(ms, .90),
            "sum_ms": sum(ms),
            "max_ms": max(ms),
        }
    return summary


def print_summary(label, events, windows, unpaired, summary):
    print(f"\n=== {label} ===")
    print(f"{len(events)} events; {len(windows)} paired intervals; {len(unpaired)} unpaired events")
    print("Phase                 count     mean ms   median ms      p90 ms    sum ms")
    for phase, r in summary.items():
        print(f"{phase:21} {r['count']:5d} {r['mean_ms']:11.3f}"
              f" {r['median_ms']:11.3f} {r['p90_ms']:11.3f} {r['sum_ms']:11.1f}")
    print("WARNING: only host boundaries; target_process is submission, target_sync"
          " waits for device completion; stage durations are not additive GPU critical path.")
    print("WARNING: verify generation/decode separation by draft_tokens and step_before;"
          " draft_begin/end spans may run on a different thread from target decode.")


def write_windows(path, windows):
    fields = [
        "phase", "slot_id", "step_before", "generation_tokens_before",
        "batch_tokens", "draft_tokens", "start_ns", "end_ns",
        "start_raw_ns", "end_raw_ns", "start_tid", "end_tid", "duration_ms",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(windows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+", help="Labels matching input traces")
    parser.add_argument("--speculative-only", action="store_true",
                        help="Exclude non-draft target stages with draft_tokens=0")
    parser.add_argument("--windows-out", type=Path,
                        help="Write first trace's paired host intervals (TSV, MONOTONIC ns)")
    parser.add_argument("--json", type=Path, help="Write JSON summaries")
    args = parser.parse_args(argv)
    if len(args.traces) > 2:
        parser.error("At most two traces may be compared")
    if args.labels and len(args.labels) != len(args.traces):
        parser.error("Provide one label per trace")
    try:
        data = []
        for i, path in enumerate(args.traces):
            events = parse_events(path)
            windows, unpaired = match_intervals(events)
            summary = analyze(windows, args.speculative_only)
            label = args.labels[i] if args.labels else path.stem
            print_summary(label, events, windows, unpaired, summary)
            data.append({"label": label, "events": len(events),
                         "intervals": len(windows), "unpaired": len(unpaired),
                         "summary": summary})
            if i == 0 and args.windows_out:
                write_windows(args.windows_out, windows)
        if len(data) == 2:
            print("\n=== 98K-vs-32K comparison, median host duration differences ===")
            for phase in PHASE_NAMES:
                a = data[0]["summary"].get(phase)
                b = data[1]["summary"].get(phase)
                if a and b:
                    print(f"{phase:21} {a['median_ms']:10.3f} ->"
                          f" {b['median_ms']:10.3f} ms (delta {b['median_ms']-a['median_ms']:+.3f})")
            print("Caution: this is not an NVIDIA+AMD GPU critical-path calculation.")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
