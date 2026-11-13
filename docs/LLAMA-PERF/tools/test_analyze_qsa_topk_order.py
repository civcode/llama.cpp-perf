#!/usr/bin/env python3
"""CPU-only synthetic tests for exact score, membership and radix ordering."""
from array import array
import copy
import unittest

from analyze_qsa_topk_order import audit_records, ordered_key

KEY = ('HIP', 0, 'mask', 0)

def record(scores, picks, *, blocks=True):
    n_pool, nq = len(scores), 1
    k = len(picks)
    a = {
        'key': KEY, 'kind': 'mask' if blocks else 'topk',
        'n_pool': n_pool, 'n_top': k, 'nq': nq, 'kv': 128,
        'scores': array('f', scores), 'selected': array('i', picks),
        'blocks': array('i', [i for p in range(n_pool) for i in (p*2, p*2+1)]) if blocks else None,
        'extra': array('i', [100]) if blocks else None,
        'bshape': (2,n_pool,1,1) if blocks else None,
        'eshape': (1,nq,1,1) if blocks else None,
    }
    return a

class RadixOrderingTests(unittest.TestCase):
    def pair(self, a, b):
        return audit_records({KEY:a},{KEY:b})

    def test_permutation_of_above_threshold_is_expected(self):
        a=record([9,8,7,6,5],[0,1,2])
        b=record([9,8,7,6,5],[1,0,2])
        d=self.pair(a,b)
        self.assertEqual(d['different_order_positions'],2)
        self.assertEqual(d['different_positions_in_tie_tail'],0)
        self.assertEqual(d['different_positions_above_cutoff'],2)
        self.assertEqual(d['queries_with_order_changes'],1)
        self.assertTrue(d['derived_kv_sets_identical'])
        self.assertEqual(d['rows'][0]['above_threshold_picks'],2)
        self.assertEqual(d['rows'][0]['mapped_kv_count'],7)

    def test_threshold_tie_stable_lower_ids_win(self):
        a=record([10,9,8,8,8,7],[0,1,2,3])
        b=record([10,9,8,8,8,7],[1,0,2,3])
        d=self.pair(a,b)
        self.assertEqual(d['cutoff_queries_with_ties'],1)
        self.assertEqual(d['rows'][0]['threshold_tie_count'],3)
        self.assertEqual(d['rows'][0]['threshold_tie_selected'],2)

    def test_threshold_tie_tail_misorder_rejected(self):
        a=record([10,9,8,8,8,7],[0,1,2,3])
        b=record([10,9,8,8,8,7],[1,0,3,2])
        with self.assertRaisesRegex(ValueError,'tie tail'):
            self.pair(a,b)

    def test_bad_selection_membership_rejected(self):
        a=record([9,8,7,6,5],[0,1,2])
        b=record([9,8,7,6,5],[0,1,3])
        with self.assertRaisesRegex(ValueError,'radix score/tie policy'):
            self.pair(a,b)

    def test_duplicate_selected_indices_rejected(self):
        a=record([9,8,7,6,5],[0,1,2])
        b=record([9,8,7,6,5],[0,0,2])
        with self.assertRaisesRegex(ValueError,'invalid selected'):
            self.pair(a,b)

    def test_byte_mismatched_score_negative_zero_rejected(self):
        a=record([1.,0.,-1.],[0,1])
        b=record([1.,-0.,-1.],[0,1])
        with self.assertRaisesRegex(ValueError,'scores bytes differ'):
            self.pair(a,b)
        self.assertLess(ordered_key(-0.), ordered_key(0.))

    def test_mapping_mismatch_rejected(self):
        a=record([9,8,7,6,5],[0,1,2])
        b=record([9,8,7,6,5],[1,0,2])
        b['blocks'][1]=101
        with self.assertRaisesRegex(ValueError,'blocks bytes differ'):
            self.pair(a,b)

    def test_extra_mapping_mismatch_rejected(self):
        a=record([9,8,7,6,5],[0,1,2])
        b=record([9,8,7,6,5],[1,0,2])
        b['extra'][0]=99
        with self.assertRaisesRegex(ValueError,'extra bytes differ'):
            self.pair(a,b)

    def test_nonfused_matches_index_set(self):
        a=record([9,8,7,6,5],[0,1,2],blocks=False)
        b=record([9,8,7,6,5],[1,0,2],blocks=False)
        d=self.pair(a,b)
        self.assertTrue(d['derived_kv_sets_identical'])
        self.assertIsNone(d['rows'][0]['mapped_kv_count'])

    def test_incorrect_group_alignment_rejected(self):
        a=record([9,8,7,6,5],[0,1,2]);b=record([9,8,7,6,5],[0,1,2]);b['n_pool']=6
        with self.assertRaisesRegex(ValueError,'n_pool mismatch'):
            self.pair(a,b)

    def test_max_rows_zero_still_aggregates(self):
        a=record([9,8,7,6,5],[0,1,2]);b=record([9,8,7,6,5],[1,0,2])
        d=audit_records({KEY:a},{KEY:b},max_rows=0)
        self.assertEqual(d['rows'],[])
        self.assertEqual(d['different_order_positions'],2)

    def test_negative_infinity_padding(self):
        a=record([9,8,float('-inf'),float('-inf')],[0,1,2])
        b=record([9,8,float('-inf'),float('-inf')],[1,0,2])
        d=self.pair(a,b)
        self.assertEqual(d['rows'][0]['threshold_tie_count'],2)

    def test_all_identity(self):
        a=record([9,8,7,6,5],[0,1,2]);b=copy.deepcopy(a)
        d=self.pair(a,b)
        self.assertEqual(d['different_order_positions'],0)

if __name__ == '__main__':
    unittest.main()
