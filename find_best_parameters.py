import optuna
from xgboost import XGBoostModel, TreeBooster
import numpy as np
from options import args_parser
from datasets.data_preprocess import data_preprocess

_args = args_parser()
(X_train, y_train), (X_test, y_test) = data_preprocess(_args)

ask_boost_round = input(str("Do you want to change the number of boosting rounds? [50]: "))
if ask_boost_round:
    if isinstance(ask_boost_round, int):
        NUM_BOOST_ROUND = ask_boost_round
        print(f"Number of boosting rounds: {NUM_BOOST_ROUND}")
    else:
        raise ValueError("Please enter an integer value.")
else:
    NUM_BOOST_ROUND = 50
    print(f"Number of boosting rounds: {NUM_BOOST_ROUND}")

ask_n_trials = input(str("Do you want to change the number of n_trials? [50]: "))
if ask_n_trials:
    if isinstance(ask_n_trials, int):
        N_TRIALS = ask_n_trials
        print(f"Number of boosting rounds: {N_TRIALS}")
    else:
        raise ValueError("Please enter an integer value.")
else:
    N_TRIALS = 50
    print(f"Number of boosting rounds: {N_TRIALS}")

class Objective:
    def loss(self, y, pred): raise NotImplementedError
    def gradient(self, y, pred): raise NotImplementedError
    def hessian(self, y, pred): raise NotImplementedError


class SquaredErrorObjective(Objective):
    def loss(self, y, pred): return np.mean((y - pred) ** 2)
    def gradient(self, y, pred): return pred - y
    def hessian(self, y, pred): return np.ones(len(y))


def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),  # Adjusted bounds
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),             # Safer lower bound
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
        'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),                 # Adjusted bounds
        'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),  # Safer range
    }

    xgb = XGBoostModel(params, random_seed=42)
    xgb.fit(X_train, y_train, SquaredErrorObjective(), NUM_BOOST_ROUND, verboose=True)
    pred_scratch = xgb.predict(X_test)
    return SquaredErrorObjective().loss(y_test, pred_scratch)

def main():
    print("-- [DEV] OPTUNA HYPERPARAMETER TUNING --")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    print("Best Parameters:", study.best_params)
    print("Best Loss Value:", study.best_value)