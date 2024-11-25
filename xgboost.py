from collections import defaultdict

import numpy as np
import pandas as pd


class XGBoostModel:
    """ XGBoost model class """
    def __init__(self, params, random_seed=None):
        self.boosters = []
        self.params = defaultdict(lambda: None, params)
        self.subsample = self.params['subsample'] \
            if self.params['subsample'] is not None else 1.0
        self.learning_rate = self.params['learning_rate'] \
            if self.params['learning_rate'] is not None else 0.3
        self.base_prediction = self.params['base_score'] \
            if self.params['base_score'] is not None else 0.5
        self.max_depth = self.params['max_depth'] \
            if self.params['max_depth'] is not None else 5
        self.rng = np.random.default_rng(seed=random_seed)

    def fit(self, X, y, objective, num_boost_round, verboose=False):
        current_predictions = np.zeros_like(y) + self.base_prediction

        for i in range(num_boost_round):
            gradients = objective.gradient(y, current_predictions)
            hessians = objective.hessian(y, current_predictions)
            sample_idxs = None if self.subsample == 1.0 else self.rng.choice(
                len(y), size=int(np.floor(self.subsample * len(y))), replace=False)
            booster = TreeBooster(X, gradients, hessians, self.params,
                                  self.max_depth, sample_idxs)
            current_predictions += self.learning_rate * booster.predict(X)
            self.boosters.append(booster)
            if verboose:
                print(f'Round: {i} | train loss: {objective.loss(y, current_predictions)}')
            else: print("Training (verboose == False)...")

    def predict(self, X):
        booster_preds = np.array([booster.predict(X) for booster in self.boosters])
        return self.base_prediction + self.learning_rate * booster_preds.sum(axis=0)

class TreeBooster:
    """Tree booster class"""
    def __init__(self, X, g, h, params, max_depth, idxs=None):
        self.params = params
        self.max_depth = max_depth
        assert self.max_depth >= 0, 'max_depth must be non-negative'
        self.min_child_weight = params['min_child_weight'] \
            if params['min_child_weight'] is not None else 1.0
        self.reg_lambda = params['reg_lambda'] \
            if params['reg_lambda'] is not None else 1.0
        self.gamma = params['gamma'] \
            if params['gamma'] is not None else 0.0
        self.colsample_bynode = params['colsample_bynode'] \
            if params['colsample_bynode'] is not None else 1.0

        if isinstance(g, pd.Series): g = g.values
        if isinstance(h, pd.Series): h = h.values
        if idxs is None: idxs = np.arange(len(g))
        self.X, self.g, self.h, self.idxs = X, g, h, idxs
        self.n, self.c, = len(idxs), X.shape[1]
        self.value = -g[idxs].sum() / (h[idxs].sum() + self.reg_lambda)
        self.best_score_so_far = 0.
        self.left, self.right = None, None
        if self.max_depth > 0:
            self._maybe_insert_child_nodes()


    def _maybe_insert_child_nodes(self):

        # First try to find a better split for each feature
        for i in range(self.c):
            self._find_better_split(i)

        # If this node is a leaf (no better split found), return immediately
        if self.is_leaf:
            return

        # Create child nodes by splitting on the best feature and threshold
        x = self.X[self.idxs, self.split_feature_idx]
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]

        # Recursively create left and right child nodes
        if len(left_idx) > 0:  # Check if there are data points for the left node
            self.left = TreeBooster(self.X, self.g, self.h, self.params,
                                        self.max_depth - 1, self.idxs[left_idx])
        else:
            self.left = None

        if len(right_idx) > 0:  # Check if there are data points for the right node
            self.right = TreeBooster(self.X, self.g, self.h, self.params,
                                        self.max_depth - 1, self.idxs[right_idx])
        else:
            self.right = None

    @property
    def is_leaf(self): return self.left is None and self.right is None

    def _find_better_split(self, feature_idx):
        g, h, x = self.g[self.idxs], self.h[self.idxs], self.X[self.idxs, feature_idx]
        sort_idx = np.argsort(x)
        sort_g, sort_h, sort_x = g[sort_idx], h[sort_idx], x[sort_idx]
        sum_g, sum_h = g.sum(), h.sum()
        sum_g_right, sum_h_right = sum_g, sum_h
        sum_g_left, sum_h_left = 0., 0.

        for i in range(0, self.n - 1):
            # Update gradients and hessians for the current split
            g_i, h_i, x_i, x_i_next = sort_g[i], sort_h[i], sort_x[i], sort_x[i + 1]

            # Update the sums for left and right
            sum_g_left += g_i
            sum_g_right -= g_i
            sum_h_left += h_i
            sum_h_right -= h_i

            # Check if the left or right side has insufficient weight to continue
            if sum_h_left < self.min_child_weight or x_i == x_i_next:
                continue
            if sum_h_right < self.min_child_weight:
                break

            # Calculate the gain of this potential split
            gain = 0.5 * ((sum_g_left ** 2 / (sum_h_left + self.reg_lambda)) +
                          (sum_g_right ** 2 / (sum_h_right + self.reg_lambda)) -
                          (sum_g ** 2 / (sum_h + self.reg_lambda))) - self.gamma / 2

            # If the gain is better than the best score so far, update the best score
            if gain > self.best_score_so_far:
                self.split_feature_idx = feature_idx
                self.best_score_so_far = gain
                self.threshold = (x_i + x_i_next) / 2

            # Early stopping condition if the gain is significantly lower than the best score
            if gain < self.best_score_so_far * 0.01:  # Example: 1% threshold of the best score
                break

    def predict(self, X): return np.array([self._predict_row(row) for row in X])

    def _predict_row(self, row):
        if self.is_leaf: return self.value
        child = self.left if row[self.split_feature_idx] <= self.threshold else self.right
        return child._predict_row(row)

if __name__ == "__main__":
    from options import args_parser
    from datasets.data_preprocess import data_preprocess
    _args = args_parser()
    (X_train, y_train), (X_test, y_test) = data_preprocess(_args)

    class Objective:
        def loss(self, y, pred): raise NotImplementedError
        def gradient(self, y, pred): raise NotImplementedError
        def hessian(self, y, pred): raise NotImplementedError

    class SquaredErrorObjective(Objective):
        def loss(self, y, pred): return np.mean((y - pred) ** 2)
        def gradient(self, y, pred): return pred - y
        def hessian(self, y, pred): return np.ones(len(y))

    params = {
        'learning_rate': 0.1,
        'max_depth': 10,
        'subsample': 0.7,
        'reg_lambda': 1.3,
        'gamma': 0.001,
        'min_child_weight': 25,
        'base_score': 0.0,
        'tree_method': 'exact',
    }

    def elapsed_time(func):
        def wrapper(*args, **kwargs):
            import datetime
            start_time = datetime.datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            print("===" * 15)
            print("Elapsed time: ", end_time - start_time)
            print("===" * 15)
            return result
        return wrapper

    def ask_boost_round():
        ask_boost_round = input("Do you want to change the number of boosting rounds? [100]: ")
        if ask_boost_round:
            try:
                num_boost_round = int(ask_boost_round)
                print(f"Number of boosting rounds: {num_boost_round}")
            except ValueError:
                raise ValueError("Please enter an integer value.")
        else:
            num_boost_round = 100
            print(f"Number of boosting rounds: {num_boost_round}")
        return num_boost_round

    @elapsed_time
    def xgboost_scratch(param: dict):
        num_boost_round = ask_boost_round()
        # train the from-scratch XGBoost model
        model_scratch = XGBoostModel(param, random_seed=42)
        model_scratch.fit(X_train, y_train, SquaredErrorObjective(), num_boost_round,
                          verboose=True)
        pred_scratch = model_scratch.predict(X_test)
        print(f'Loss Score: {SquaredErrorObjective().loss(y_test, pred_scratch)}')

    optuna_msg = input(str('Do you want to train optuna with the xgboost scratch model? (y/n): '))

    from pprint import pprint
    if optuna_msg.lower() == 'y':
        import find_best_parameters
        best_params = find_best_parameters.main()
        pprint({"Best Parameters": best_params})
        print("Running xgboost from scratch with best parameters...")
        xgboost_scratch(best_params)
    else:
        pprint({"Default Parameters": params})
        print("---" * 15)
        xgboost_scratch(params)