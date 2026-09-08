import unittest

from src.model import Model


class TestModel(unittest.TestCase):
    def test_create_Model_object(self):
        model = Model()

        self.assertIsNotNone(model)

    def test_Model_default_params_are_null(self):
        model = Model()

        self.assertEqual(model.params, [0, 0])

    def test_Model_init_params(self):
        params = (42, 24)
        model = Model(params)

        self.assertEqual(model.params, [42, 24])

    def test_Model_cannot_have_only_one_param(self):
        params = (42,)
        with self.assertRaises(ValueError):
            Model(params)

    def test_Model_cannot_have_three_params(self):
        params = (1, 2, 3)
        with self.assertRaises(ValueError):
            Model(params)

    def test_Model_returns_null_eval_by_default(self):
        model = Model()
        predict = model.eval(42)

        self.assertEqual(predict, 0)

    def test_Model_return_correct_eval(self):
        test_cases = (
            ((0, 1), ((1, 1), (42, 42), (665, 665), (-555, -555), (42.42, 42.42))),
            ((1, 0), ((1, 1), (42, 1), (665, 1), (-555, 1), (42.42, 1))),
            ((1, 1), ((1, 2), (42, 43), (665, 666), (-555, -554), (42.42, 43.42))),
            ((2, 3), ((1, 5), (42, 128), (665, 1997), (-555, -1663), (42.42, 129.26))),
        )
        for params, data in test_cases:
            model = Model(params)
            for value, expected in data:
                with self.subTest(value=value, expected=expected):
                    predict = model.eval(value)
                    self.assertEqual(predict, expected)


if __name__ == "__main__":
    unittest.main()
