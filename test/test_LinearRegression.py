import numpy as np
import unittest

from src.linearregression import LinearRegression


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
        data = ((1, 2), (3, 4), (5, 6))
        lr = LinearRegression()
        lr.setData(data)
        data = np.array(data)

        np.testing.assert_allclose(lr.data, data)

    def test_LinearRegression_predictedGap(self):
        data = ((1, 2), (3, 3), (5, 18))
        expected_gaps = np.array((1, 4, 7))
        lr = LinearRegression()
        lr.model.setParams((1, 2))
        lr.setData(data)
        gaps = lr.getPredictedGap()
        assert gaps is not None
        np.testing.assert_allclose(gaps, expected_gaps)
