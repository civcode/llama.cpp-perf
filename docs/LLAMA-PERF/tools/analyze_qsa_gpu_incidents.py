#!/usr/bin/env python3
"""Read-only, privacy-minimizing timeline of NVIDIA/PCIe/AMD incident logs.

Consumes a few externally saved kernel/app log files (prefer
journalctl -k -o short-iso --no-pager), classifies diagnostic lines,
and reports Xid 79/154 sequencing and nearby PCIe/AER/DPC events.
No hardware access, inference, GPU resets, privileged operations,
automatic journal export, network requests, or log data committed to Git.

This is incident CORRELATION, not root-cause attribution. Empty/missing
journals cannot be interpreted as no hardware errors.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import re

MAX_SOURCES = 6
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_MATCHED_LINES = 10000

XID_RE = re.compile(r"\bXid\s*(?:\([^)]*\)\s*)?:\s*(\d{1,3})\b", re.I)
XID_ALTERNATE_RE = re.compile(r"\bXid\s+(\d{1,3})\b", re.I)
ISOTIME_RE = re.compile(
    r"^(20\d{2}-\d\d-\d\d[T ]\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|[+-]\d\d:?\d\d)?)\b"
)


def classify(message: str) -> tuple[str, str | None] | None:
    xid = XID_RE.search(message) or XID_ALTERNATE_RE.search(message)
    if xid:
        return "nvidia_xid", xid.group(1)
    low = message.lower()
    if ("pci" in low or "pcie" in low) and (
            re.search(r"\baer:\s*(?:.*\b(?:error|fatal|uncorrected|corrected)\b)", low)
            or "pcie bus error" in low):
        return "pcie_aer", None
    if ("pci" in low or "pcie" in low) and re.search(
            r"\bdpc:\s*(?:.*\b(?:containment|trigger|error|fatal)\b)", low):
        return "pcie_dpc", None
    if ("pcie" in low or "pci " in low) and (
            "link down" in low or "link failure" in low or
            "link error" in low or "training failed" in low or
            "surprise down" in low):
        return "pcie_link_problem", None
    if ("nvrm" in low or "nvidia" in low) and (
            "fallen off" in low or "unknown error" in low or
            "gpu has fallen" in low):
        return "nvidia_access_error", None
    if "cuda" in low and (
            "unspecified launch failure" in low or "illegal memory access" in low or
            "device-side assert" in low):
        return "cuda_error", None
    if ("amdgpu" in low or "kfd" in low) and (
            "ring timeout" in low or "gpu reset" in low or "vm fault" in low or
            "device lost" in low):
        return "amd_fault", None
    if "iommu" in low and re.search(r"\b(?:fault|error)\b", low):
        return "iommu_error", None
    return None


def time_of(line: str) -> datetime | None:
    match = ISOTIME_RE.match(line)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(match.group(1).replace("Z", "+00:00"))
    except ValueError:
        return None


def analyze_lines(lines, label: str) -> dict:
    if not isinstance(label, str) or not 1 <= len(label) <= 64:
        raise ValueError("invalid source label")
    found = []
    scanned = 0
    for scanned, line in enumerate(lines, 1):
        if scanned > 500000:
            raise ValueError("source log exceeds 500000 lines")
        if len(line) > 20000:
            continue
        result = classify(line)
        if result is None:
            continue
        kind, code = result
        stamp = time_of(line)
        found.append({
            "line_number": scanned, "type": kind, "xid": code,
            "timestamp": stamp.isoformat() if stamp is not None else None,
        })
        if len(found) > MAX_MATCHED_LINES:
            raise ValueError("too many matching diagnostic log lines")
    counts = Counter(x["type"] for x in found)
    xids = Counter(x["xid"] for x in found if x["type"] == "nvidia_xid")
    incidents = []
    for index, ev in enumerate(found):
        if ev["type"] != "nvidia_xid" or ev["xid"] != "79":
            continue
        # Select nearest relevant preceding lines by *log ordering*.
        # Only compute timed proximity if both lines provide compatible ISO
        # timestamps, and never conflate events from separate source files.
        before = found[max(0, index - 12):index]
        after = found[index + 1:index + 13]
        stamp = datetime.fromisoformat(ev["timestamp"]) if ev["timestamp"] else None
        near = []
        for other in before:
            observed = datetime.fromisoformat(other["timestamp"]) if other["timestamp"] else None
            delta = None
            if stamp is not None and observed is not None:
                try:
                    elapsed = (stamp - observed).total_seconds()
                    if 0 <= elapsed <= 600:
                        delta = round(elapsed, 3)
                except TypeError:
                    pass
            near.append({
                "type": other["type"], "xid": other["xid"],
                "seconds_before_xid79": delta,
            })
        paired_154 = False
        for ev2 in after:
            if ev2["type"] == "nvidia_xid" and ev2["xid"] == "79":
                break
            if ev2["type"] == "nvidia_xid" and ev2["xid"] == "154":
                paired_154 = True
                break
        incidents.append({
            "timestamp": ev["timestamp"],
            "line_number": ev["line_number"],
            "xid154_after_before_next_xid79": paired_154,
            "preceding_relevant_events": near,
        })
    return {
        "source_label": label,
        "lines_scanned": scanned,
        "matched_diagnostic_lines": len(found),
        "event_type_counts": dict(sorted(counts.items())),
        "xid_code_counts": dict(sorted(xids.items())),
        "xid79_incidents": incidents,
        "timestamps_available": sum(x["timestamp"] is not None for x in found),
        "caveat": ("Only supplied logs are inspected. Missing logs, filtered "
                   "journal access and no matching entries cannot certify "
                   "device health or exclude a hardware trigger."),
    }


def analyze_sources(sources: list[tuple[str, Path]]) -> dict:
    if not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError("supply 1..6 source files")
    if len({label for label, _ in sources}) != len(sources):
        raise ValueError("duplicate source label")
    reports = []
    for label, path in sources:
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_FILE_BYTES:
            raise ValueError(f"missing, empty or oversized source file for {label}")
        with path.open(encoding="utf-8", errors="replace") as stream:
            reports.append(analyze_lines(stream, label))
    return {
        "scope": "offline_observation_not_causal_proof",
        "sources": reports,
        "total_xid79_in_supplied_logs": sum(
            len(r["xid79_incidents"]) for r in reports),
        "total_xid154_in_supplied_logs": sum(
            r["xid_code_counts"].get("154", 0) for r in reports),
        "interpretation": (
            "GPU fell off PCIe bus is an observed failure state, not a unique "
            "root cause. Look for timestamped preceding AER/DPC and correlate "
            "with workload, tracer activity, and driver/card/link settings. "
            "One successful PCIe Gen3 run cannot establish a fix. No new GPU "
            "activity is justified by this report alone."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", action="append", required=True, metavar="LABEL=PATH",
        help="one to six independently collected historical kernel/app logs")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        sources = []
        for item in args.source:
            if "=" not in item:
                raise ValueError("source must be LABEL=PATH")
            label, path = item.split("=", 1)
            if not label or not path:
                raise ValueError("source must have a label and path")
            sources.append((label, Path(path)))
        report = analyze_sources(sources)
        with args.out.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(json.dumps({
        "sources": len(report["sources"]),
        "xid79": report["total_xid79_in_supplied_logs"],
        "xid154": report["total_xid154_in_supplied_logs"],
        "full_report": str(args.out),
        "causal_root_cause": "NOT_DETERMINED",
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
