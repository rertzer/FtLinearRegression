import unittest
import pandas as pd

import src.ft_training as ft


class TestFtTraining(unittest.TestCase):
    def test_FtTraining_NoArg(self):
        with self.assertRaises(SystemExit) as cm:
            ft.ft_training(
                ("./ft_training"),
            )
        self.assertEqual(cm.exception.code, 1)

    def test_FtTraining_BadArg(self):
        with self.assertRaises(SystemExit) as cm:
            ft.ft_training(
                ("./ft_training", "toto"),
            )
        self.assertEqual(cm.exception.code, 1)

    def test_FtTraining_EmptyFile(self):
        with self.assertRaises(SystemExit) as cm:
            ft.ft_training(("./ft_training", "tests/data/empty_file"))
        self.assertEqual(cm.exception.code, 1)

    def test_FtTraining_DataFile(self):
        status = ft.ft_training(("./ft_training", "tests/data/data.csv"))
        self.assertEqual(status, 0)

    def test_FtTraining_Train(self):
        df = pd.read_csv("tests/data/data.csv")
        self.assertIsNotNone(df)
