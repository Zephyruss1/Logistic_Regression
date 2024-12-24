import optuna
from xgboost_scratch import XGBoostModel
from scripts.squared_error_objective import SquaredErrorObjective
from scripts.options import args_parser
from datasets.data_preprocess import data_preprocess
from scripts.others import ask_boost_round, ask_n_trials

_args = args_parser()
(X_train, y_train), (X_test, y_test) = data_preprocess(_args)

NUM_BOOST_ROUND = ask_boost_round()
N_TRIALS = ask_n_trials()

def objective(trial):
    try:
        params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'min_child_weight': trial.suggest_float('min_child_weight', 0.1, 10.0),
        }

        xgb = XGBoostModel(params, X_train, y_train, random_seed=42)
        xgb.fit(SquaredErrorObjective(), NUM_BOOST_ROUND, verboose=True)
        pred_scratch = xgb.predict(X_test)
        return SquaredErrorObjective().loss(y_test, pred_scratch)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return float('inf')  # Mark this trial as failed

def main() -> dict:
    from pprint import pprint
    print("-- OPTUNA HYPERPARAMETER TUNING --")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=N_TRIALS)
    pprint({"Best Parameters": study.best_params})
    print("Best Loss Value:", study.best_value)
    return study.best_params