#!/usr/bin/env python3
"""Read-only comparison of AMD GPU kernel families in ROCPD trace windows.

Auto windows are bounded by first/last float32-logits-sized GPU->CPU readbacks.
They are *not* an exact decode/verification phase boundary. For truncated ROCprof
captures use --start-s/--end-s after verifying the local timeline.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3


def kernel_family(name):
    n = (name or "").lower()
    if "lightning_indexer" in n:
        return "qsa_lightning_indexer"
    if "top_k_mask" in n:
        return "qsa_top_k_mask"
    if "flash_attn" in n or "fattn" in n or "flash_attention" in n:
        return "flash_attention"
    if "gated_delta" in n or "gdn" in n or "gla_" in n or "ssm_" in n:
        return "recurrent_gdn_ssm"
    if "mul_mat_vec_q" in n or "mul_mat_q" in n or "vec_dot_q" in n:
        return "quantized_matmul"
    if "mul_mat_vec_f" in n or "mul_mat_f" in n:
        return "floating_matmul"
    if "top_k" in n or "topk" in n:
        return "other_top_k"
    if "rope" in n:
        return "rope"
    return "other"


def validate_views(db):
    objects = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type IN ('view','table')")}
    required = {"kernels", "rocpd_info_agent", "rocpd_memory_copy", "regions_and_samples"}
    missing = required - objects
    if missing:
        raise ValueError("Missing ROCPD views: " + ", ".join(sorted(missing)))


def read_trace(path, label, logits_bytes, start_s, end_s):
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        validate_views(db)
        min_start, max_end = db.execute("SELECT MIN(start), MAX(end) FROM kernels").fetchone()
        if min_start is None:
            raise ValueError(f"No kernels in {path}")
        agents = {r[0]: {"name": r[1], "type": r[2], "absolute_index": r[3]}
                  for r in db.execute("SELECT id,name,type,absolute_index FROM rocpd_info_agent")}
        cpus = [a for a, meta in agents.items() if meta["type"] == "CPU"]
        gpus = [a for a, meta in agents.items() if meta["type"] == "GPU"]
        if not cpus or not gpus:
            raise ValueError(f"No CPU or GPU agents in {path}")
        # Select logits-sized readbacks only (other transfers are intentionally excluded).
        placeholders_cpu = ",".join("?" for _ in cpus)
        placeholders_gpu = ",".join("?" for _ in gpus)
        readbacks = db.execute(f"""
            SELECT start,end,src_agent_id,dst_agent_id
            FROM rocpd_memory_copy
            WHERE size = ? AND src_agent_id IN ({placeholders_gpu})
                           AND dst_agent_id IN ({placeholders_cpu})
            ORDER BY start
        """, (logits_bytes, *gpus, *cpus)).fetchall()
        if (start_s is None and end_s is None):
            if len(readbacks) < 8:
                raise ValueError(f"Found {len(readbacks)} logits-sized readbacks in {path}; "
                                 "specify --start-s/--end-s or review trace size")
            begin, finish = readbacks[0][0], readbacks[-1][1]
            window_mode = "first-to-last logits readback (approximate, excludes initial compute)"
        else:
            begin = min_start + int(start_s * 1e9) if start_s is not None else min_start
            finish = min_start + int(end_s * 1e9) if end_s is not None else max_end
            window_mode = "explicit seconds from first collected kernel"
        if begin >= finish:
            raise ValueError(f"Empty or inverted window in {path}")
        window_ns = finish - begin
        in_window_readbacks = sum(1 for start, end, _, _ in readbacks if begin <= start < finish)

        # Sorted by GPU and time to calculate stream-union busy time without
        # retaining hundreds of thousands of kernel intervals in memory.
        by_gpu = {}
        last_span = {}
        names_by_gpu = defaultdict(Counter)
        categories_by_gpu = defaultdict(Counter)
        counts_by_gpu = Counter()
        for gpu, start, end, duration, name in db.execute("""
            SELECT agent_abs_index,start,end,duration,name FROM kernels
            WHERE start < ? AND end > ?
            ORDER BY agent_abs_index,start
        """, (finish, begin)):
            start, end = max(start, begin), min(end, finish)
            if end <= start:
                continue
            # Dispatches at boundary are counted, and duration is clipped for consistency.
            observed_ns = end - start
            counts_by_gpu[gpu] += 1
            names_by_gpu[gpu][name or "<unnamed>"] += observed_ns
            categories_by_gpu[gpu][kernel_family(name)] += observed_ns
            if gpu not in last_span:
                by_gpu[gpu] = 0
                last_span[gpu] = (start, end)
            else:
                old_start, old_end = last_span[gpu]
                if start <= old_end:
                    last_span[gpu] = (old_start, max(old_end, end))
                else:
                    by_gpu[gpu] += old_end - old_start
                    last_span[gpu] = (start, end)
        for gpu, (start, end) in last_span.items():
            by_gpu[gpu] += end - start

        all_families = Counter()
        gpu_summary = []
        for gpu, cats in sorted(categories_by_gpu.items()):
            total_ns = sum(cats.values())
            all_families.update(cats)
            top = [{"name": name, "accumulated_ms": round(ns / 1e6, 3)}
                   for name, ns in names_by_gpu[gpu].most_common(8)]
            gpu_summary.append({"agent_absolute_index": gpu, "kernel_dispatches": counts_by_gpu[gpu],
                                "accumulated_ms": round(total_ns / 1e6, 3),
                                "busy_union_ms": round(by_gpu[gpu] / 1e6, 3),
                                "busy_union_window_pct": round(100 * by_gpu[gpu] / window_ns, 2),
                                "families_ms": {k: round(v / 1e6, 3) for k, v in cats.most_common()},
                                "top_kernels": top})

        api = db.execute("""
            SELECT name,COUNT(*),SUM(end-start) FROM regions_and_samples
            WHERE category LIKE 'HIP_%' AND start >= ? AND start < ?
            GROUP BY name ORDER BY SUM(end-start) DESC LIMIT 12
        """, (begin, finish)).fetchall()
        return {
            "label": label, "path": str(path), "logits_bytes": logits_bytes,
            "window": {"mode": window_mode, "start_s": round((begin-min_start)/1e9, 3),
                       "end_s": round((finish-min_start)/1e9, 3),
                       "duration_ms": round(window_ns/1e6, 3),
                       "logits_readbacks_total": len(readbacks),
                       "logits_readbacks_in_window": in_window_readbacks},
            "gpu_agents": gpu_summary,
            "families_ms": {k: round(v / 1e6, 3) for k, v in all_families.most_common()},
            "hip_api_leaders": [{"name": name, "calls": calls, "sum_ms": round(ns/1e6, 3)}
                                for name, calls, ns in api]
        }


def present(reports):
    for report in reports:
        w = report["window"]
        print(f"\n=== {report['label']} ===")
        print(f"Window {w['start_s']}-{w['end_s']} s ({w['duration_ms']} ms); {w['mode']}")
        print(f"Logits-sized readbacks: {w['logits_readbacks_in_window']} in window, {w['logits_readbacks_total']} total")
        print("WARNING: GPU time is accumulated device work, not elapsed CPU or end-to-end time.")
        print("WARNING: readback-bounded windows omit first-step work; cross-run placement can differ.")
        for gpu in report["gpu_agents"]:
            print(f"  GPU {gpu['agent_absolute_index']}: {gpu['kernel_dispatches']} kernels, "
                  f"{gpu['accumulated_ms']} ms accumulated, {gpu['busy_union_window_pct']}% union busy")
            for item in gpu["top_kernels"][:6]:
                print(f"    {item['accumulated_ms']:9.3f} ms  {item['name'][:100]}")
        print("  Family                    Accumulated ms   ms/readback")
        for family, ms in report["families_ms"].items():
            count = w["logits_readbacks_in_window"]
            normalized = f"{ms/count:10.3f}" if count else "     n/a"
            print(f"  {family:25} {ms:12.3f} {normalized}")
        print("  Leading HIP APIs:")
        for api in report["hip_api_leaders"][:5]:
            print(f"    {api['name']}: {api['calls']} calls, {api['sum_ms']} ms summed host API time")

    if len(reports) == 2:
        a,b = reports
        print(f"\n=== {b['label']} - {a['label']} (ms per observed readback) ===")
        a_n=a['window']['logits_readbacks_in_window']
        b_n=b['window']['logits_readbacks_in_window']
        if not a_n or not b_n:
            print("Cannot normalize: at least one window has no logits readbacks")
        else:
            for family in sorted(set(a['families_ms'])|set(b['families_ms'])):
                x=a['families_ms'].get(family,0)/a_n
                y=b['families_ms'].get(family,0)/b_n
                print(f"{family:25} {x:9.3f} -> {y:9.3f} ms/readback "
                      f"(delta {y-x:+.3f}, ratio {y/x:.2f}x)" if x else
                      f"{family:25} {x:9.3f} -> {y:9.3f} ms/readback (new category)")
            print("IMPORTANT: counts of readbacks need not equal generated tokens or MTP verification steps.")
            print("Compare matching workloads with same model, placement and decode shape.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", nargs="+", type=Path, help="One or two ROCPD SQLite databases")
    parser.add_argument("--labels", nargs="+", help="Labels for the database files")
    parser.add_argument("--logits-bytes", type=int, default=993280,
                        help="Logits copy bytes (default: 248320 float32 entries = 993280)")
    parser.add_argument("--start-s", type=float, help="Explicit window start from first recorded kernel (per trace)")
    parser.add_argument("--end-s", type=float, help="Explicit window end from first recorded kernel (per trace)")
    parser.add_argument("--json", type=Path, help="Write structured summary (not raw trace) to this path")
    args = parser.parse_args()
    if len(args.db) > 2:
        parser.error("Compare at most two traces at once")
    if args.labels and len(args.labels) != len(args.db):
        parser.error("--labels must specify exactly one label per DB")
    if args.logits_bytes <= 0:
        parser.error("--logits-bytes must be positive")
    for path in args.db:
        if not path.is_file():
            parser.error(f"Database not found: {path}")
    try:
        data = [read_trace(path, args.labels[i] if args.labels else path.stem,
                           args.logits_bytes, args.start_s, args.end_s)
                for i,path in enumerate(args.db)]
    except (sqlite3.Error, ValueError) as err:
        parser.error(str(err))
    present(data)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(data, indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
