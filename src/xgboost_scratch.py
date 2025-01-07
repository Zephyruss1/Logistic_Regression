from collections import defaultdict

import numpy as np
import pandas as pd


class XGBoostModel:
    """
    A class implementing a simplified version of the XGBoost algorithm.

    This class supports gradient-boosted decision trees with custom objectives, allowing for flexible
    control over hyperparameters, subsampling, and tree depth.

    Attributes:
        X (np.ndarray): Feature matrix for training.
        y (np.ndarray): Target values for training.
        boosters (list): List of trained `TreeBooster` objects.
        params (dict): Dictionary containing hyperparameters, with defaults applied using `defaultdict`:
            - `subsample` (float): Fraction of data to subsample for each boosting round. Default is 1.0.
            - `learning_rate` (float): Learning rate for weight updates. Default is 0.3.
            - `base_score` (float): Initial base prediction for all observations. Default is 0.5.
            - `max_depth` (int): Maximum depth of each tree. Default is 5.
        subsample (float): Fraction of the dataset to use in each boosting round.
        learning_rate (float): Step size shrinkage for updates.
        base_prediction (float): Initial prediction for all targets.
        max_depth (int): Maximum depth of the decision trees.
        rng (np.random.Generator): Random number generator for reproducible subsampling.

    Methods:
        __init__(self, params, X, y, random_seed=None):
            Initializes the model with hyperparameters, data, and an optional random seed.

        fit(self, objective, num_boost_round, verboose=False):
            Trains the model using gradient-boosted trees.

            Args:
                objective: An objective function object with `gradient`, `hessian`, and `loss` methods.
                num_boost_round (int): Number of boosting rounds to perform.
                verboose (bool): If `True`, prints loss at each boosting round.

        predict(self, X):
            Generates predictions for the input data using the trained model.

            Args:
                X (np.ndarray): Feature matrix for prediction.

            Returns:
                np.ndarray: Predicted values for the input data.

    Notes:
        - The `fit` method trains the model by iteratively creating `TreeBooster` objects for each boosting round.
        - Subsampling is applied during training to improve generalization and reduce overfitting.
        - The objective must implement three methods:
            - `gradient(y, predictions)`: Compute gradients.
            - `hessian(y, predictions)`: Compute hessians.
            - `loss(y, predictions)`: Calculate loss for monitoring.

    Limitations:
        - This implementation is a simplified version and may lack certain optimizations and features of
          a full-fledged XGBoost library.
    """

    def __init__(self, params, X, y, random_seed=None):
        self.X, self.y = X, y
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

    def fit(self, objective, num_boost_round, verboose=False):
        current_predictions = np.zeros_like(self.y) + self.base_prediction
        if verboose is False: print("Verboose: False")
        for i in range(num_boost_round):
            gradients = objective.gradient(self.y, current_predictions)
            hessians = objective.hessian(self.y, current_predictions)
            sample_idxs = None if self.subsample == 1.0 else self.rng.choice(
                len(self.y), size=int(np.floor(self.subsample * len(self.y))), replace=False)
            booster = TreeBooster(self.X, gradients, hessians, self.params,
                                  self.max_depth, sample_idxs)
            current_predictions += self.learning_rate * booster.predict(self.X)
            self.boosters.append(booster)
            if verboose:
                print(f'Round: {i} | train loss: {objective.loss(self.y, current_predictions)}')

    def predict(self, X):
        booster_preds = np.array([booster.predict(X) for booster in self.boosters])
        return self.base_prediction + self.learning_rate * booster_preds.sum(axis=0)


class TreeBooster:
    """
    A class implementing a decision tree booster for gradient-boosted tree algorithms.

    The `TreeBooster` class constructs and manages individual decision tree nodes. It supports
    gradient and hessian-based optimization, with options for regularization and constraints.

    Attributes:
        params (dict): Dictionary containing hyperparameters for the tree booster:
            - `min_child_weight` (float): Minimum sum of hessian values required for a child node.
            - `reg_lambda` (float): L2 regularization term for weight optimization.
            - `gamma` (float): Minimum gain required to make a split.
            - `colsample_bynode` (float): Fraction of features to consider at each split.
        max_depth (int): Maximum depth of the tree. Must be non-negative.
        min_child_weight (float): Extracted from `params`, defaulting to 1.0.
        reg_lambda (float): L2 regularization parameter, defaulting to 1.0.
        gamma (float): Minimum gain required to make a split, defaulting to 0.0.
        colsample_bynode (float): Fraction of columns to sample per split, defaulting to 1.0.
        X (np.ndarray): Feature matrix.
        g (np.ndarray): Gradient values of the loss function.
        h (np.ndarray): Hessian values of the loss function.
        idxs (np.ndarray): Indices of the data points in the current node.
        n (int): Number of data points in the current node.
        c (int): Number of features in the dataset.
        value (float): Predicted value of the node, computed as:
            `-sum(gradients) / (sum(hessians) + reg_lambda)`
        best_score_so_far (float): Best split gain observed so far for the current node.
        left (TreeBooster or None): Left child node (created if a split occurs).
        right (TreeBooster or None): Right child node (created if a split occurs).

    Methods:
        __init__(self, X, g, h, params, max_depth, idxs=None):
            Initializes the tree booster node. If `max_depth > 0`, attempts to insert child nodes.

    Notes:
        - The tree uses gradients and hessians for optimization to support gradient-boosted
          decision trees.
        - Regularization parameters (`reg_lambda`, `gamma`) are used to prevent overfitting.
        - If `max_depth` is greater than 0, the `_maybe_insert_child_nodes` method is called to
          construct child nodes recursively.
    """

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
        """
        Creates child nodes for the current node by finding and applying the best possible splits.

        This function is a core part of the decision tree construction process. It evaluates each
        feature to find the best split that maximizes the gain. If a split is found, it partitions
        the data into left and right subsets and recursively creates child nodes for further splits.

        Key Steps:
        1. **Find the Best Split**: Iterates over all features to evaluate potential splits and
           updates the `split_feature_idx` and `threshold` for the best gain.
        2. **Check for Leaf Node**: If no valid split is found (leaf node condition met), the
           function terminates.
        3. **Create Child Nodes**: Splits the data at the best threshold for the chosen feature
           and initializes left and right child nodes if sufficient data points exist in each subset.

        Attributes Used:
            - `self.c`: Number of features to evaluate.
            - `self.is_leaf`: Property to check if the node is a leaf.
            - `self.X`: Feature matrix.
            - `self.idxs`: Indices of the data points in the current node.
            - `self.split_feature_idx`: Index of the feature chosen for the split.
            - `self.threshold`: Threshold value for the split.

        Attributes Modified:
            - `self.left`: Left child node, created if sufficient data exists in the left subset.
            - `self.right`: Right child node, created if sufficient data exists in the right subset.

        Notes:
            - If a node becomes a leaf, no child nodes are created.
            - Splits are performed based on maximizing the gain calculated in `_find_better_split`.

        Returns:
            None
        """
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
    def is_leaf(self):
        return self.left is None and self.right is None

    def _find_better_split(self, feature_idx):
        """
        Identifies the best split for a given feature to maximize the gain.

        This method evaluates potential split points for the specified feature by iterating
        through all unique values in the feature. The goal is to find a split that optimizes
        the gain, which is a measure of how much a split improves the decision tree's performance.

        Parameters:
            feature_idx (int): Index of the feature being evaluated for the split.

        Key Steps:
        1. **Prepare Feature Data**:
           - Extract gradients (`g`), hessians (`h`), and feature values (`x`) for the current node.
           - Sort these values to facilitate efficient split evaluation.
        2. **Iterate Over Splits**:
           - Iterate through all possible split points for the feature.
           - Update the left and right sums of gradients and hessians as the split moves.
        3. **Check Validity**:
           - Ensure the left and right child nodes have sufficient weight (`min_child_weight`).
           - Skip invalid splits or redundant splits (e.g., consecutive identical feature values).
        4. **Calculate Gain**:
           - Compute the gain for each valid split using the gradients, hessians, and regularization terms.
           - Update the best split parameters (`split_feature_idx`, `threshold`, `best_score_so_far`) if a better split is found.
        5. **Early Stopping**:
           - Stop iteration early if the gain drops significantly below the current best score.

        Attributes Used:
            - `self.g`: Gradient values for the loss function.
            - `self.h`: Hessian values for the loss function.
            - `self.X`: Feature matrix.
            - `self.idxs`: Indices of data points in the current node.
            - `self.min_child_weight`: Minimum weight required for a child node to be valid.
            - `self.reg_lambda`: Regularization parameter for gain calculation.
            - `self.gamma`: Regularization term for split gain.
            - `self.best_score_so_far`: Current best gain value.
            - `self.n`: Number of data points in the current node.

        Attributes Modified:
            - `self.split_feature_idx`: Index of the feature chosen for the best split.
            - `self.threshold`: Threshold value of the best split.
            - `self.best_score_so_far`: Updated best gain value.

        Notes:
            - A valid split must ensure that both child nodes have sufficient hessian weight.
            - Gains are calculated using the formula:
              ```
              gain = 0.5 * (sum_g_left^2 / (sum_h_left + reg_lambda) +
                            sum_g_right^2 / (sum_h_right + reg_lambda) -
                            sum_g^2 / (sum_h + reg_lambda)) - gamma / 2
              ```
            - Early stopping is applied to prevent unnecessary calculations when gains drop significantly.

        Returns:
            None
        """
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
        if self.is_leaf:
            return self.value
        child = self.left if row[self.split_feature_idx] <= self.threshold else self.right
        return child._predict_row(row)
