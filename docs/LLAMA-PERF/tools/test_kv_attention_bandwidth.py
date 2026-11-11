import unittest
from kv_attention_bandwidth import estimate


class KVBandwidthTest(unittest.TestCase):
    def test_f16_32k(self):
        r = estimate(32768, 2, 256, 256, 2, 2, 0.11504, 640)
        self.assertEqual(r['logical_kv_bytes'], 67108864)
        self.assertAlmostEqual(r['logical_kv_mib'], 64.0)
        self.assertAlmostEqual(r['nominal_scan_gbs'], 583.351, places=2)
        self.assertAlmostEqual(r['ideal_single_scan_floor_ms'], 0.1048576)

    def test_kv_triples(self):
        a = estimate(32768, 2, 256, 256, 2, 2, .12)
        b = estimate(98304, 2, 256, 256, 2, 2, .36)
        self.assertEqual(b['logical_kv_bytes'], 3*a['logical_kv_bytes'])
        self.assertAlmostEqual(b['nominal_scan_gbs'], a['nominal_scan_gbs'])
        self.assertIsNone(a['nominal_fraction_of_peak'])

    def test_invalid(self):
        with self.assertRaises(ValueError):
            estimate(0, 2, 256, 256, 2, 2, .1)
        with self.assertRaises(ValueError):
            estimate(1, 2, 256, 256, 2, 2, .1, -640)

if __name__ == '__main__':
    unittest.main()
