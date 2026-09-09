import unittest
import numpy as np

import src.ft_training as ft


class TestFtTraining(unittest.TestCase):
    def test_FtTraining_NoArg(self):
        status = ft.ft_training(
            ("./ft_training"),
        )
        self.assertEqual(status, 1)

    def test_FtTraining_BadArg(self):
        status = ft.ft_training(
            ("./ft_training", "toto"),
        )
        self.assertEqual(status, 1)

    def test_FtTraining_EmptyFile(self):
        status = ft.ft_training(("./ft_training", "test/data/empty_file"))
        self.assertEqual(status, 1)
