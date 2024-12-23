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


if __name__ == "__main__":
    unittest.main()