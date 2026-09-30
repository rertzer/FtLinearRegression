import re
import unittest
import numpy as np


import src.randomator as rd
from src.model import Model


class TestRandomator(unittest.TestCase):
    def setUp(self):
        self.randomator = rd.Randomator()

    def dummyModel(self):
        model = Model((0.2, 0.5))
        model.norm_mins = np.array([0, 0])
        model.norm_range = np.array([100, 100])
        model.normalized = True
        model.setParams(model.getRawParams())
        print("Dummy params: ", model.params)
        self.randomator.model = model
        self.randomator.nb_points = 4

    def test_Randomator_exist(self):
        self.assertIsNotNone(self.randomator)

    def test_Randomator_file_name(self):
        self.randomator.setFileName()
        pattern = re.search(r"^randata\d{9}", self.randomator.file)
        self.assertIsNotNone(pattern)

    def test_Randomator_nb_points(self):
        self.assertTrue(6 <= self.randomator.nb_points <= 666)

    def test_Randomator_thetas(self):
        with self.subTest():
            self.assertIsNotNone(self.randomator.model)
            # thetas = self.randomator.model.params
            # self.assertTrue(0 <= thetas[0] <= 1)
            # self.assertTrue(0 <= thetas[1] <= 1)

    def test_Randomator_MinMax(self):
        print("mins: ", self.randomator.model.norm_mins)
        print("ranges: ", self.randomator.model.norm_range)
        with self.subTest():
            self.assertTrue(
                np.array([1, 1]).all()
                <= self.randomator.model.norm_range.all()
                <= np.array([1000000, 1000000]).all()
            )
            self.assertTrue(
                np.array([0, 0]).all()
                <= self.randomator.model.norm_mins.all()
                <= (self.randomator.model.norm_range / 2).all()
            )

    def test_Randomator_Data(self):
        self.dummyModel()
        self.randomator.create_data()
        with self.subTest():
            self.assertEqual(2 * self.randomator.nb_points, self.randomator.data.size)
            self.assertTrue(0 <= np.all(self.randomator.data <= 100))
            self.assertTrue(
                np.allclose(self.randomator.data[1], 0.5 * self.randomator.data[0] + 20)
            )
