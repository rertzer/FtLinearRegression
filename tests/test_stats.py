import unittest
import numpy as np

from src import stats
from src.model import Model
from src.config import *


class TestStats(unittest.TestCase):
    def test_stats_sst(self):
        data = np.array(((-1, 1), (1, 2), (2, 3), (3, 2)))
        with self.subTest():
            self.assertEqual(stats.sst(data), 2)
            data = np.array(((1, 0), (1, 0)))
            self.assertEqual(stats.sst(data), 0)
            data = np.array(((1, 1), (1, 2), (1, 6)))
            self.assertEqual(stats.sst(data), 14)

    def test_stats_y_mean(self):
        data = np.array(((-1, 1), (1, 2), (2, 3), (3, 2)))
        with self.subTest():
            self.assertEqual(stats.y_mean(data), 2)
            data = np.array(((1, 0), (1, 0)))
            self.assertEqual(stats.y_mean(data), 0)
            data = np.array(((1, 1), (1, 2), (1, 6)))
            self.assertEqual(stats.y_mean(data), 3)

    def test_stats_sse(self):
        data = np.array(((-1, 1), (1, 2), (2, 3), (3, 2)))
        model = Model((1, 1))
        model.setParams((1, 1))
        with self.subTest():
            self.assertEqual(stats.sse(data, model), 9)

    def test_stats_r_squared(self):
        data = np.array(((-1, 1), (1, 2), (2, 3), (3, 2)))
        model = Model((1, 1))
        self.assertEqual(stats.r_squared(data, model), 4.5)

    def test_stats_variance(self):
        with self.subTest():
            data = np.array(((1, 0), (33, 0), (-12, 0)))
            self.assertAlmostEqual(stats.variance(data), 0)

            data = np.array(((1, 17), (33, 44), (-12, 42)))
            self.assertAlmostEqual(stats.variance(data), 150.8888, 3)

            data = np.array(((1, 1), (1, 2), (1, 6)))
            self.assertAlmostEqual(stats.variance(data), 14 / 3, 3)
