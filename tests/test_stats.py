import unittest

from src.stats import Stats
from src.config import *


class TestStats(unittest.TestCase):
    def test_create_Stats_object(self):
        stats = Stats(LOOPS)

        self.assertIsNotNone(stats)

    def test_records_has_len_loops(self):
        stats = Stats(LOOPS)
        self.assertEqual(stats.records.shape, (2, LOOPS))

    def test_stats_getLoss(self):
        stats = Stats(42)
        losses = (11, 22, 33, 34, 122, 244, -12, 0.42)
        for l in losses:
            with self.subTest():
                stats.records[RSQUARED][-1] = l
                self.assertEqual(stats.getRsquared(), l)

    def test_stats_getRsquared(self):
        stats = Stats(42)
        rs = (1, 2, 3, 4, 12, 24, -2, 0.42)
        for r in rs:
            with self.subTest():
                stats.records[RSQUARED][-1] = r
                self.assertEqual(stats.getRsquared(), r)
