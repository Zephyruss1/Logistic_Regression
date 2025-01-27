import sys
import os
sys.path.append("datasets")
sys.path.append("scripts")
sys.path.append("models")

import numpy as np
import pickle as pkl
from datasets.data_preprocess import data_preprocess
from scripts.options import args_parser
from scripts.others import elapsed_time, ask_boost_round
from scripts.plot import plot_logreg
import logging
from scripts.squared_error_objective import SquaredErrorObjective

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
current_work_dir = os.path.dirname(__file__)

weight_diff_list = []
obj_diff_list = []

parser = args_parser()
_args, unknown = parser.parse_known_args()


@elapsed_time
def main_run():
    if __name__ == "__main__":
        try:
            (x_train, y_train), (x_test, y_test) = data_preprocess(_args)

            ask_model = input("List of available models:\n1. Logistic Regression\n2. XGBoost\n3. Exit\n->: ")
            if ask_model == "1":
                from src import logistic_regression
                print("learning rate: ", _args.lr)
                print("Optimizer: ", _args.optimizer)
                print("-------------------------")

                Model = logistic_regression.LogisticRegression(
                    args=_args, X_train=x_train, Y_train=y_train, X_test=x_test
                )

                weight_diff, obj_diff = Model.diff_cal(Model.weights)
                print("\n------------ Initial ------------")
                print("weight error: {:.4e}".format(weight_diff))
                print("objective error: {:.4e}".format(obj_diff))

                Eigvals = np.linalg.eigvals(Model.pre_Hessian)
                print("\nmax eigenvalue of Hessian:{:.4f}".format(np.max(Eigvals)))
                print("min eigenvalue of Hessian:{:.4f}".format(np.min(Eigvals)))

                for i in range(_args.iteration):
                    weight_diff, obj_diff = Model.update()
                    print("\n------------ Iteration {} ------------".format(i + 1))
                    print("weight error: {:.4e}".format(weight_diff))
                    print("objective error: {:.4e}".format(obj_diff))
                    weight_diff_list.append(weight_diff)
                    obj_diff_list.append(obj_diff)

                    if weight_diff / np.sqrt(Model.dimension) <= 1e-5:
                        break

                val = Model.getTest() > 0.5
                weights = Model.weights
                val2 = y_test > 0.5
                percent_correct = np.mean(val == val2) * 100
                print("Accuracy: {:.1f}%".format(percent_correct))
                with open(os.path.join(current_work_dir, "models", "logistic_regression_weights.pkl"), "wb") as weight_file:
                    pkl.dump(weights, weight_file)

                print("Weights saved successfully to logistic_regression_weights.pkl.")
                file_name = "optimization_results/{}_{}.pkl".format("logreg", _args.optimizer)
                file_name = os.path.join(current_work_dir, file_name)
                with open(file_name, "wb") as f:
                    pkl.dump([weight_diff_list, obj_diff_list], f)
                plot_logreg()
            elif ask_model == "2":
                def xgboost_scratch(param: dict):
                    from src.xgboost_scratch import XGBoostModel
                    # train the from-scratch XGBoost model
                    model_scratch = XGBoostModel(param, x_train, y_train, random_seed=42)
                    model_scratch.fit(SquaredErrorObjective(), ask_boost_round(),
                                      verboose=True)
                    model_scratch.save_weights()
                    pred_scratch = model_scratch.predict(x_test)
                    print(f'Loss Score: {SquaredErrorObjective().loss(y_test, pred_scratch)}')

                optuna_msg = input(str('Do you want to train optuna with the xgboost scratch model? (y/n): '))

                from pprint import pprint
                if optuna_msg.lower() == 'y':
                    from src import find_best_parameters
                    best_params = find_best_parameters.main()
                    pprint({"Best Parameters": best_params})
                    print("Running xgboost from scratch with best parameters...")
                    xgboost_scratch(best_params)
                else:
                    default_params = {
                        'learning_rate': 0.1,
                        'max_depth': 10,
                        'subsample': 0.7,
                        'reg_lambda': 1.3,
                        'gamma': 0.001,
                        'min_child_weight': 25,
                        'base_score': 0.0,
                        'tree_method': 'exact',
                    }
                    pprint({"Default Parameters": default_params})
                    print("---" * 15)
                    xgboost_scratch(default_params)
            elif ask_model == "3":
                sys.exit() 
            else:
                raise ValueError("Please enter a valid model.")

        except Exception as e:
            logger.error(
                f"{__file__} | Line {e.__traceback__.tb_lineno} | An error occurred: {e} "
            )


main_run()
