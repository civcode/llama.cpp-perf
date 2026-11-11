#!/usr/bin/env python3
"""Estimate *logical* K/V scan rate for one flash-attention operation.

The estimate assumes one read of every K and V element across all KV heads.
It is not measured DRAM traffic: caches, redundant reads, mask pruning,
quantization/dequantization and parallel reduction may change actual work.
"""
import argparse
import json


def estimate(kv, kv_heads, dim_k, dim_v, bytes_k, bytes_v, latency_ms, peak_gbs=None):
    if any(x <= 0 for x in (kv, kv_heads, dim_k, dim_v, bytes_k, bytes_v, latency_ms)):
        raise ValueError('All workload dimensions, bytes per element and latency must be positive')
    if peak_gbs is not None and peak_gbs <= 0:
        raise ValueError('Peak GB/s must be positive')
    logical_bytes = kv * kv_heads * (dim_k * bytes_k + dim_v * bytes_v)
    nominal_gbs = logical_bytes / (latency_ms * 1e-3) / 1e9
    return {
        'logical_kv_bytes': logical_bytes,
        'logical_kv_mib': logical_bytes / (2**20),
        'latency_ms': latency_ms,
        'nominal_scan_gbs': nominal_gbs,
        'peak_gbs': peak_gbs,
        'nominal_fraction_of_peak': nominal_gbs / peak_gbs if peak_gbs else None,
        'ideal_single_scan_floor_ms': logical_bytes / (peak_gbs * 1e9) * 1e3 if peak_gbs else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kv', type=int, required=True, help='K/V sequence positions')
    p.add_argument('--kv-heads', type=int, required=True)
    p.add_argument('--dim-k', type=int, required=True)
    p.add_argument('--dim-v', type=int, required=True)
    p.add_argument('--bytes-k', type=float, default=2, help='Storage bytes per K element; F16=2')
    p.add_argument('--bytes-v', type=float, default=2, help='Storage bytes per V element; F16=2')
    p.add_argument('--latency-ms', type=float, required=True, help='Measured whole attention operation latency')
    p.add_argument('--peak-gbs', type=float, help='Rated GPU peak external memory bandwidth GB/s')
    args = p.parse_args()
    try:
        result = estimate(args.kv, args.kv_heads, args.dim_k, args.dim_v, args.bytes_k, args.bytes_v, args.latency_ms, args.peak_gbs)
    except ValueError as exc:
        p.error(str(exc))
    print(json.dumps(result, indent=2))
    print('CAUTION: logical single-pass scan rate is NOT an observed DRAM throughput counter.')


if __name__ == '__main__':
    main()
