import unittest

import src.ft_predict as ft


class TestFtPredict(unittest.TestCase):
    def test_FtPredict_getArguments(self):
        args = ft.get_arguments(("toto",))
        self.assertEqual(args, ("thetas.json"))

    def test_FtPredict_getArgumentsWithFile(self):
        args = ft.get_arguments(("toto", "toto.json"))
        self.assertEqual(args, ("toto.json"))

    def test_FtPredict_bad_txt_file(self):
        with self.assertRaises(SystemExit) as cm:
            _ = ft.get_data_json("tests/data/badThetas.txt")
        self.assertEqual(cm.exception.code, 1)

    def test_FtPredict_missing_thetha_in_txt_file(self):
        with self.assertRaises(SystemExit) as cm:
            _ = ft.get_data_json("tests/data/MissingThetas.txt")
        self.assertEqual(cm.exception.code, 1)
