import unittest
import datetime
import sys

sys.path.append("..")
from scripts.others import elapsed_time, ask_boost_round


class TestScripts(unittest.TestCase):
    def test_ask_boost_round(self):
        # Test the ask_boost_round function
        from unittest.mock import patch
        from main import ask_boost_round

        with patch("builtins.input", return_value="50"):
            num_boost_round = ask_boost_round()
            self.assertEqual(num_boost_round, 50)

        with patch("builtins.input", return_value=""):
            num_boost_round = ask_boost_round()
            self.assertEqual(num_boost_round, 100)

    def test_elapsed_time_decorator(self):
        # Test the elapsed_time decorator
        @elapsed_time
        def dummy_function():
            return "Completed"

        start_time = datetime.datetime.now()
        result = dummy_function()
        end_time = datetime.datetime.now()
        self.assertEqual(result, "Completed")
        self.assertLess(end_time - start_time, datetime.timedelta(seconds=5))

    def test_r2_score(self):
        # Test the r2_score function
        from scripts.others import r2_score

        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 5]
        self.assertEqual(r2_score(y_true, y_pred), 1.0)

        y_true = [1, 2, 3, 4, 5]
        y_pred = [5, 4, 3, 2, 1]
        self.assertLess(r2_score(y_true, y_pred), 0)  # It should be negative

        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 6]
        self.assertEqual(r2_score(y_true, y_pred), 0.9)

        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 3, 4, 4]
        self.assertEqual(r2_score(y_true, y_pred), 0.9)

        y_true = [3, -0.5, 2, 7]
        y_pred = [2.5, 0.0, 2, 8]
        self.assertEqual(r2_score(y_true, y_pred), 0.9486081370449679)

        y_true = [3, 3, 3, 3, 3]  # No variance in true values
        y_pred = [1, 2, 3, 4, 5]
        try:
            result = r2_score(y_true, y_pred)
            self.assertTrue(result != result)  # Check for NaN
        except ZeroDivisionError:
            pass  # Expected exception

        y_true = [1, 2, 3, 4, 5]
        mean_prediction = [3, 3, 3, 3, 3]  # Model predicting the mean of y_true
        self.assertEqual(r2_score(y_true, mean_prediction), 0.0)


if __name__ == "__main__":
    unittest.main()
