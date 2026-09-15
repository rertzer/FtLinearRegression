import unittest
import numpy as np

from src.stats import Stats
from src.config import LOOPS


class TestStats(unittest.TestCase):
    def test_create_Stats_object(self):
        stats = Stats(LOOPS)

        self.assertIsNotNone(stats)

    def test_records_has_len_loops(self):
        stats = Stats(LOOPS)

        self.assertEqual(len(stats.records), LOOPS)
