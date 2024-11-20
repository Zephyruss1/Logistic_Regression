import sys

sys.path.append("./datasets")

import os
import numpy as np
import pickle as pkl
from datasets.data_preprocess import data_preprocess
import logistic_regression
from options import args_parser
import datetime
from plot import plot_logreg
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
current_work_dir = os.path.dirname(__file__)


def elapsed_time(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        start_time.strftime("%H:%M:%S")
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        print("===" * 15)
        print("Elapsed time: ", end_time - start_time)
        print("===" * 15)
        return result

    return wrapper


weight_diff_list = []
obj_diff_list = []
_args = args_parser()


@elapsed_time
def main_run():
    if __name__ == "__main__":
        try:
            (x_train, y_train), (x_test, y_test) = data_preprocess(_args)

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
            val2 = y_test > 0.5
            percent_correct = np.mean(val == val2) * 100
            print("Accuracy: {:.1f}%".format(percent_correct))

        except Exception as e:
            logger.error(
                f"{__file__} | Line {e.__traceback__.tb_lineno} | An error occurred: {e} "
            )


main_run()

file_name = "./results/{}_{}.pkl".format("logreg", _args.optimizer)
file_name = os.path.join(current_work_dir, file_name)
with open(file_name, "wb") as f:
    pkl.dump([weight_diff_list, obj_diff_list], f)
plot_logreg()
