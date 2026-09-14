import unittest
import numpy as np

import src.ft_plotting as ftp


class TestPlotting(unittest.TestCase):
    def test_get_one_arg(self):
        expected_args = ("./ft_plotting.py", "toto.csv", None)
        args = ftp.get_args(expected_args[:2])
        self.assertEqual(args, expected_args[1:])

    def test_get_two_args(self):
        expected_args = ("./ft_plotting.py", "toto.csv", "tutu.txt")
        args = ftp.get_args(expected_args)
        self.assertEqual(args, expected_args[1:])

    def test_get_vars(self):
        args = (
            "./ft_plotting.py",
            "tests/data/data.csv",
            "tests/data/thetas.txt",
        )
        vars = ftp.get_vars(args)

        with self.subTest():
            self.assertEqual(len(vars[0]), 24)  # number of lines
        with self.subTest():
            np.testing.assert_array_equal(vars[1], np.array([1111, 2222]))


if __name__ == "__main__":
    unittest.main()
