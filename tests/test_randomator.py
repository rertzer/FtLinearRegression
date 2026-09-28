import re
import unittest


import src.randomator as rd


class TestRandomator(unittest.TestCase):
    def setUp(self):
        self.randomator = rd.randomator()

    def test_Randomator_exist(self):
        self.assertIsNotNone(self.randomator)

    def test_Randomator_file_name(self):
        pattern = re.search(r"^randata\d{9}", self.randomator.file)
        self.assertIsNotNone(pattern)

    def test_Randomator_nb_points(self):
        self.assertTrue(6 <= self.randomator.nb_points <= 666)

    def test_Randomator_thetas(self):
        with self.subTest():
            self.assertEqual(len(self.randomator.thetas), 2)
            self.assertTrue(0 <= self.randomator.thetas[0] <= 1)
            self.assertTrue(0 <= self.randomator.thetas[1] <= 1)
