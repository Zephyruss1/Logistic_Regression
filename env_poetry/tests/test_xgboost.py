import unittest
import numpy as np
import sys

sys.path.append("..")
from scripts.squared_error_objective import SquaredErrorObjective
from src.xgboost_scratch import XGBoostModel, TreeBooster


class TestXGBoostModel(unittest.TestCase):
    def setUp(self):
        # Set up mock data for testing
        self.X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.y = np.array([1.0, 2.0, 3.0, 4.0])
        self.params = {
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "reg_lambda": 1.0,
            "gamma": 0.0,
            "min_child_weight": 1.0,
            "base_score": 0.0,
            "colsample_bynode": 1.0,
        }
        self.objective = SquaredErrorObjective()
        self.model = XGBoostModel(self.params, self.X, self.y, random_seed=42)

    def test_initialization(self):
        # Test that model initializes correctly
        self.assertEqual(self.model.learning_rate, 0.1)
        self.assertEqual(self.model.max_depth, 3)
        self.assertEqual(self.model.subsample, 1.0)

    def test_fit(self):
        # Test that fit does not raise errors and updates boosters
        self.model.fit(self.objective, num_boost_round=10, verboose=False)
        self.assertEqual(len(self.model.boosters), 10)

    def test_predict(self):
        # Test predictions after training
        self.model.fit(self.objective, num_boost_round=10, verboose=False)
        predictions = self.model.predict(self.X)
        self.assertEqual(len(predictions), len(self.y))
        self.assertTrue(np.all(predictions >= 0.0))

    def test_loss_calculation(self):
        # Test the SquaredErrorObjective loss
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        loss = self.objective.loss(self.y, predictions)
        self.assertEqual(loss, 0.0)  # Since predictions match actual

    def test_gradient_and_hessian(self):
        # Test the gradient and hessian computation
        predictions = np.array([1.0, 2.0, 3.0, 4.0])
        gradient = self.objective.gradient(self.y, predictions)
        hessian = self.objective.hessian(self.y, predictions)
        self.assertTrue(np.all(gradient == 0.0))  # Since predictions match actual
        self.assertTrue(np.all(hessian == 1.0))  # Hessian should be constant

    def test_tree_booster_initialization(self):
        # Test that TreeBooster initializes correctly
        gradients = self.objective.gradient(self.y, np.zeros_like(self.y))
        hessians = self.objective.hessian(self.y, np.zeros_like(self.y))
        booster = TreeBooster(self.X, gradients, hessians, self.params, max_depth=3)
        self.assertIsNotNone(booster)
        self.assertTrue(booster.is_leaf)

    def test_tree_split(self):
        # Test that tree splitting works
        gradients = self.objective.gradient(self.y, np.zeros_like(self.y))
        hessians = self.objective.hessian(self.y, np.zeros_like(self.y))
        booster = TreeBooster(self.X, gradients, hessians, self.params, max_depth=3)
        booster._maybe_insert_child_nodes()
        self.assertTrue(booster.is_leaf)  # Expect at least one split

    def test_training_with_default_params(self):
        # Test training the model with default parameters
        self.model.fit(self.objective, num_boost_round=5, verboose=False)
        self.assertEqual(len(self.model.boosters), 5)


if __name__ == "__main__":
    unittest.main()
