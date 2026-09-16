import pandas as pd
import numpy as np
import unittest

from src.linearregression import LinearRegression
from src.config import *


class TestLinearRegression(unittest.TestCase):
    def test_LinearRegression_exists(self):
        lr = LinearRegression()
        self.assertIsNotNone(lr)

    def test_LinearRegresstion_default_learning_step_value(self):
        lr = LinearRegression()
        self.assertEqual(lr.learning_step, 0.01)

    def test_LinearRegression_init_with_learning_step_value(self):
        lr = LinearRegression(0.2)
        self.assertEqual(lr.learning_step, 0.2)

    def test_LinearRegression_set_data(self):
        data = np.array(((1, 2), (3, 4), (5, 6)))
        lr = LinearRegression()
        lr.setData(data)
        assert lr.data is not None
        np.testing.assert_allclose(lr.data, data)

    def test_LinearRegression_predictedDistance(self):
        data = ((1, 2), (3, 3), (5, 18))
        expected_distances = np.array((1, 4, -7))
        lr = LinearRegression()
        lr.model.setParams((1, 2))
        lr.setData(data)
        gaps = lr.getPredictedDistance()
        assert gaps is not None
        np.testing.assert_allclose(gaps, expected_distances)

    def test_LinearRegression_Delta(self):

        data = ((1, 2), (3, 3), (5, 18))
        expected_delta = np.array((-2 / 3 * 0.01, -22 / 3 * 0.01))
        lr = LinearRegression()
        lr.model.setParams((1, 2))
        lr.setData(data)
        delta = lr.getDelta()
        np.testing.assert_allclose(delta, expected_delta)

    def test_LinearRegression_Loss(self):
        data = ((0, 1), (1, 2), (2, 3), (3, 4))
        lr = LinearRegression()
        lr.setData(data)
        self.assertEqual(lr.getLoss(), 7.5)

    def test_LinearRegression_setSST(self):
        data = ((-1, 1), (1, 2), (2, 3), (3, 2))
        lr = LinearRegression()
        lr.setData(data)
        self.assertEqual(lr.sst, 2)

    def test_LinearRegression_getSSE(self):
        data = ((-1, 1), (1, 2), (2, 3), (3, 2))
        lr = LinearRegression()
        lr.setData(data)
        lr.model.setParams((1, 1))
        self.assertEqual(lr.getSSE(), 9)

    def test_LinearRegression_Rsquared(self):
        data = ((-1, 1), (1, 2), (2, 3), (3, 2))
        lr = LinearRegression()
        lr.model.setParams((1, 1))
        lr.setData(data)
        self.assertEqual(lr.getRsquared(), 4.5)

    def test_LinearRegression_Train(self):
        data = ((0, 1), (1, 2), (2, 3), (3, 4))
        lr = LinearRegression()
        lr.setData(data)
        lr.train()
        np.testing.assert_allclose(lr.model.params, np.array((1, 1)))

    def test_LinearRegression_LossStats(self):
        data = ((0, 1), (1, 2), (2, 3), (3, 4))
        lr = LinearRegression()
        lr.loops = 4
        lr.setData(data)
        lr.train()
        print(lr.stats.records)
        for i in range(lr.loops - 1):
            with self.subTest():
                self.assertGreater(
                    lr.stats.records[LOSS][i], lr.stats.records[LOSS][i + 1]
                )

    def test_LinearRegresssion_NormData(self):
        data = ((10, -42), (100, 17), (55, 958), (0, 33))
        norm_data = np.array(((0.1, 0), (1, 0.059), (0.55, 1.0), (0, 0.075)))
        lr = LinearRegression()
        lr.setData(data)
        lr.normData()
        assert lr.data is not None
        np.testing.assert_allclose(lr.data, norm_data)

    def test_LinearRegression_OnData(self):
        expected_params = np.array((8481, -0.02127))
        df = pd.read_csv("tests/data/data.csv")
        lr = LinearRegression()
        lr.setData(df.values)
        lr.normData()
        lr.train()
        print(lr.model.params)
        print(lr.getParams())
        np.testing.assert_allclose(lr.getParams(), expected_params, rtol=1e-02)
