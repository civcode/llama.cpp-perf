#!/usr/bin/env python3
"""Offline audit of HIP QSA dense A/B top-k order vs membership.

The HIP radix gather uses atomicAdd for strictly-above-threshold picks; the
threshold-tie tail is placed deterministically by pool index. Verify this
contract against two existing external QSA_TOPK_TRACE captures. Never run GPU
inference here. Source input is immutable. Outputs contain aggregate stats.
"""
from __future__ import annotations

import argparse
from array import array
import json
import math
from pathlib import Path
import struct


def ordered_key(x: float) -> int:
    """Match top_k_float_to_ordered in ggml-cuda/top-k.cu, including +/-0."""
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    return bits ^ (0xffffffff if bits & 0x80000000 else 0x80000000)


def _equal_raw(a: array | None, b: array | None) -> bool:
    return (a is None and b is None) or (
        a is not None and b is not None and a.typecode == b.typecode
        and a.tobytes() == b.tobytes())


def _mapped(record: dict, query: int, picks: list[int]) -> set[int] | None:
    if record['blocks'] is None:
        return None
    scores = record['scores']
    n_pool = record['n_pool']
    kv = record['kv']
    block_size = record['bshape'][0]
    result = set()
    for pool in picks:
        if scores[query*n_pool + pool] <= -math.inf:
            continue
        for offset in range(block_size):
            cell = int(record['blocks'][pool * block_size + offset])
            if 0 <= cell < kv:
                result.add(cell)
    if record['extra'] is not None:
        n_extra = record['eshape'][0]
        for cell in record['extra'][query*n_extra:(query+1)*n_extra]:
            if 0 <= cell < kv:
                result.add(int(cell))
    return result


def audit_records(left: dict, right: dict, max_rows: int = 64) -> dict:
    """Fail closed if raw inputs or selected membership differs.

    Existing compare_qsa_topk_trace.read_log supplies validated group tensors.
    This function intentionally accepts those in-memory records for CPU tests.
    """
    if not 0 <= max_rows <= 512:
        raise ValueError('max_rows must be 0..512')
    if not left or left.keys() != right.keys():
        raise ValueError('missing or misaligned QSA top-k groups')
    result = {
        'interpretation': 'HIP radix top-k index order is unspecified; only membership is consumed by fused mask',
        'groups': 0, 'queries': 0, 'positions': 0, 'different_order_positions': 0,
        'queries_with_order_changes': 0, 'different_positions_above_cutoff': 0,
        'different_positions_in_tie_tail': 0, 'cutoff_queries_with_ties': 0,
        'descending_score_order_violations_a': 0, 'descending_score_order_violations_b': 0,
        'selected_pool_sets_identical': True, 'derived_kv_sets_identical': True,
        'score_and_mapping_bytes_identical': True, 'membership_matches_radix_policy': True,
        'rows': [], 'graph_safety': 'NOT_TESTED', 'numerical_equivalence': 'NOT_ESTABLISHED',
        'production_promotion': 'NO_GO',
    }
    for key in sorted(left):
        a, b = left[key], right[key]
        if key != a['key'] or key != b['key']:
            raise ValueError(f'unexpected group key {key}')
        for field in ('n_pool', 'n_top', 'nq', 'kv', 'kind', 'bshape', 'eshape'):
            if a[field] != b[field]:
                raise ValueError(f'dense control {field} mismatch in {key}')
        if key[0] != 'HIP' or key[1] != 0 or key[2] not in ('mask', 'topk'):
            raise ValueError(f'unexpected backend/group {key}')
        for field in ('scores', 'blocks', 'extra'):
            if not _equal_raw(a[field], b[field]):
                raise ValueError(f'dense control {field} bytes differ in {key}')
        n_pool, k, nq = a['n_pool'], a['n_top'], a['nq']
        if not 0 < k <= n_pool <= 262144 or not 0 < nq <= 4096:
            raise ValueError(f'invalid group shape in {key}')
        if len(a['scores']) != n_pool * nq or len(a['selected']) != k*nq or len(b['selected']) != k*nq:
            raise ValueError(f'bad tensor payload dimensions in {key}')
        result['groups'] += 1
        for q in range(nq):
            scores = a['scores'][q*n_pool:(q+1)*n_pool]
            if any(math.isnan(x) or x == math.inf for x in scores):
                raise ValueError(f'nonfinite invalid score in {key} query {q}')
            keys = [ordered_key(x) for x in scores]
            rank = sorted(range(n_pool), key=lambda p: (-keys[p], p))[:k]
            expected = set(rank)
            picks_a = list(a['selected'][q*k:(q+1)*k])
            picks_b = list(b['selected'][q*k:(q+1)*k])
            for picks in (picks_a, picks_b):
                if len(picks) != k or len(set(picks)) != k or any(p < 0 or p >= n_pool for p in picks):
                    raise ValueError(f'invalid selected indices in {key} query {q}')
                if set(picks) != expected:
                    raise ValueError(f'index set conflicts with exact GPU radix score/tie policy in {key} query {q}')
            threshold = keys[rank[-1]]
            above = sum(x > threshold for x in keys)
            tie_ids = [i for i, x in enumerate(keys) if x == threshold]
            tie_tail = tie_ids[:k-above]
            if any(picks[above:] != tie_tail for picks in (picks_a, picks_b)):
                raise ValueError(f'GPU radix threshold tie tail differs from low-pool-ID policy in {key} query {q}')
            if set(picks_a) != set(picks_b):
                raise ValueError(f'dense control selected membership differs in {key} query {q}')
            if _mapped(a, q, picks_a) != _mapped(b, q, picks_b):
                raise ValueError(f'order-dependent mapped KV set in {key} query {q}')
            changed_positions = [i for i, (x, y) in enumerate(zip(picks_a, picks_b)) if x != y]
            changed_above = sum(i < above for i in changed_positions)
            changed_tail = len(changed_positions) - changed_above
            inv_a = sum(keys[picks_a[i]] < keys[picks_a[i+1]] for i in range(k-1))
            inv_b = sum(keys[picks_b[i]] < keys[picks_b[i+1]] for i in range(k-1))
            result['queries'] += 1
            result['positions'] += k
            result['different_order_positions'] += len(changed_positions)
            result['queries_with_order_changes'] += bool(changed_positions)
            result['different_positions_above_cutoff'] += changed_above
            result['different_positions_in_tie_tail'] += changed_tail
            result['cutoff_queries_with_ties'] += len(tie_ids) > 1
            result['descending_score_order_violations_a'] += inv_a
            result['descending_score_order_violations_b'] += inv_b
            if len(result['rows']) < max_rows:
                result['rows'].append({
                    'group': list(key), 'query': q, 'n_pool': n_pool, 'k': k,
                    'radix_expected_for_HIP': nq > 1 and n_pool > 1024,
                    'order_mismatch_positions': len(changed_positions),
                    'different_positions_above_cutoff': changed_above,
                    'different_positions_in_tie_tail': changed_tail,
                    'above_threshold_picks': above, 'threshold_tie_count': len(tie_ids),
                    'threshold_tie_selected': k-above,
                    'score_descending_inversions_a': inv_a,
                    'score_descending_inversions_b': inv_b,
                    'mapped_kv_count': len(_mapped(a, q, picks_a)) if a['blocks'] is not None else None,
                })
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('dense_a_log', type=Path)
    p.add_argument('dense_b_log', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-rows', type=int, default=32)
    args = p.parse_args()
    try:
        if args.dense_a_log.resolve(strict=True) == args.dense_b_log.resolve(strict=True):
            raise ValueError('independent dense A/B log paths required')
        from compare_qsa_topk_trace import read_log
        report = audit_records(read_log(args.dense_a_log), read_log(args.dense_b_log), args.max_rows)
        # The original logs remain immutable, and the caller chooses an external output.
        with args.out.open('x', encoding='utf-8') as fp:
            json.dump(report, fp, indent=2, allow_nan=False)
            fp.write('\n')
    except (OSError, ValueError, KeyError, TypeError, OverflowError) as err:
        p.error(str(err))
    print(json.dumps({k: report[k] for k in (
        'groups', 'queries', 'positions', 'different_order_positions',
        'different_positions_in_tie_tail', 'selected_pool_sets_identical',
        'derived_kv_sets_identical', 'production_promotion')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
