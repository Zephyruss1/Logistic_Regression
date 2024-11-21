import numpy as np
import pandas as pd
import math
from collections import defaultdict

class XGBoostModel:
    """ XGBoost model class """
    def __init__(self, params, random_seed=None):
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

    def fit(self, X, y, objective, num_boost_round, verboose=True):
        current_predictions = self.base_prediction * np.ones(shape=y.shape)
        self.boosters = []
        for i in range(num_boost_round):
            gradients = objective.gradient(y, current_predictions)
            hessians = objective.hessian(y, current_predictions)
            sample_idxs = None if self.subsample == 1.0 else self.rng.choice(
                len(y), size=math.floor(self.subsample * len(y)), replace=False)
            booster = TreeBooster(X, gradients, hessians, self.params,
                                  self.max_depth, sample_idxs)
            current_predictions += self.learning_rate * booster.predict(X)
            self.boosters.append(booster)
            if verboose:
                print(f'Round: {i} | train loss: {objective.loss(y, current_predictions)}')

    def predict(self, X):
        return (self.base_prediction + self.learning_rate *
                np.sum([booster.predict(X) for booster in self.boosters], axis=0))

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
        if self.max_depth > 0:
            self._maybe_insert_child_nodes()

    def _maybe_insert_child_nodes(self):
        for i in range(self.c): self._find_better_split(i)
        if self.is_leaf: return
        x = self.X.values[self.idxs, self.split_feature_idx]
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]
        self.left = TreeBooster(self.X, self.g, self.h, self.params,
                               self.max_depth - 1, self.idxs[left_idx])
        self.right = TreeBooster(self.X, self.g, self.h, self.params,
                               self.max_depth - 1, self.idxs[right_idx])

    @property
    def is_leaf(self): return self.best_score_so_far == 0.

    def _find_better_split(self, feature_idx):
        x = self.X.values[self.idxs, feature_idx]
        g, h = self.g[self.idxs], self.h[self.idxs]
        sort_idx = np.argsort(x)
        sort_g, sort_h, sort_x = g[sort_idx], h[sort_idx], x[sort_idx]
        sum_g, sum_h = g.sum(), h.sum()
        sum_g_right, sum_h_right = sum_g, sum_h
        sum_g_left, sum_h_left = 0., 0.

        for i in range(0, self.n -1):
            g_i, h_i, x_i, x_i_next = sort_g[i], sort_h[i], sort_x[i], sort_x[i + 1]
            sum_g_left += g_i; sum_g_right -= g_i
            sum_h_left += h_i; sum_h_right -= h_i
            if sum_h_left < self.min_child_weight or x_i == x_i_next:continue
            if sum_h_right < self.min_child_weight:break

            gain = 0.5 * ((sum_g_left**2 / (sum_h_left + self.reg_lambda))
                            + (sum_g_right**2 / (sum_h_right + self.reg_lambda))
                            - (sum_g**2 / (sum_h + self.reg_lambda))
                          ) - self.gamma / 2
            if gain > self.best_score_so_far:
                self.split_feature_idx = feature_idx
                self.best_score_so_far = gain
                self.threshold = (x_i + x_i_next) / 2

    def predict(self, X): return np.array([self._predict_row(row) for i, row in X.iterrows()])

    def _predict_row(self, row):
        if self.is_leaf: return self.value
        child = self.left if row.iloc[self.split_feature_idx] <= self.threshold else self.right
        return child._predict_row(row)

if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    X, y = fetch_california_housing(as_frame=True, return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                        random_state=43)

    class SquaredErrorObjective:
        def loss(self, y, pred): return np.mean((y - pred) ** 2)
        def gradient(self, y, pred): return pred - y
        def hessian(self, y, pred): return np.ones(len(y))

    import xgboost as xgb

    params = {
        'learning_rate': 0.01,
        'max_depth': 10,
        'subsample': 0.7,
        'reg_lambda': 1.3,
        'gamma': 0.001,
        'min_child_weight': 25,
        'base_score': 0.0,
        'tree_method': 'exact',
    }
    num_boost_round = 50

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

    @elapsed_time
    def xgboost_library():
        # train the library XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model_xgb = xgb.train(params, dtrain, num_boost_round)
        pred_xgb = model_xgb.predict(dtest)
        print(f'xgboost score: {SquaredErrorObjective().loss(y_test, pred_xgb)}')

    @elapsed_time
    def xgboost_scratch():
        # train the from-scratch XGBoost model
        model_scratch = XGBoostModel(params, random_seed=42)
        model_scratch.fit(X_train, y_train, SquaredErrorObjective(), num_boost_round)
        pred_scratch = model_scratch.predict(X_test)
        print(f'Accuracy Score: {np.mean(pred_scratch == y_test) * 100} | Loss score: {SquaredErrorObjective().loss(y_test, pred_scratch)}')

    system_msg = input(str('Do you want to train with the xgboost library model? (y/n): '))

    if system_msg.lower() == 'y': xgboost_library(), xgboost_scratch()
    else: xgboost_scratch()