import unittest
import pandas as pd
import numpy as np

import src.ft_predict as ft


class TestFtPredict(unittest.TestCase):
    def test_FtPredict_getArguments_missingArg(self):
        with self.assertRaises(SystemExit) as cm:
            ft.get_arguments(("toto",))
        self.assertEqual(cm.exception.code, 1)

    def test_FtPredict_getArguments(self):
        args = ft.get_arguments(("toto", "42.42"))
        self.assertEqual(args, (42.42, "thetas.txt"))

    def test_FtPredict_getArgumentsWithFile(self):
        args = ft.get_arguments(("toto", "42.42", "toto.txt"))
        self.assertEqual(args, (42.42, "toto.txt"))
