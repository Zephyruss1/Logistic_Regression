import sys
sys.path.append('./datasets')

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

# UPDATE: elapsed_time function.
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
args = args_parser()

@elapsed_time
def main_run():
    if __name__ == '__main__':
        try:
            (x_train, y_train), (x_test, y_test) = data_preprocess(args)

            print('learning rate: ', args.lr)
            print('Optimizer: ', args.optimizer)

            Model = logistic_regression.LogisticRegression(args=args,
                                                            X_train=x_train,
                                                            Y_train=y_train,
                                                            X_test=x_test)

            weight_diff, obj_diff = Model.diff_cal(Model.weights)
            print("\n------------ Initial ------------")
            print("weight error: {:.4e}".format(weight_diff))
            print("objective error: {:.4e}".format(obj_diff))

            Eigvals = np.linalg.eigvals(Model.pre_Hessian)
            print("\nmax eigenvalue of Hessian:{:.4f}".format(np.max(Eigvals)))
            print("min eigenvalue of Hessian:{:.4f}".format(np.min(Eigvals)))
            
            '''
            update
            '''
            for i in range(args.iteration):

                weight_diff, obj_diff = Model.update()
                print("\n------------ Iteration {} ------------".format(i + 1))
                print("weight error: {:.4e}".format(weight_diff))
                print("objective error: {:.4e}".format(obj_diff))
                weight_diff_list.append(weight_diff)
                obj_diff_list.append(obj_diff)

                if weight_diff / np.sqrt(Model.dimension) <= 1e-5:
                    break

            # UPDATE: simplifying the accuracy percent.
            val = Model.getTest() > 0.5
            val2 = y_test > 0.5
            percent_correct = np.mean(val == val2) * 100
            print(percent_correct, 'Accuracy: %')

        except Exception as e:
            logger.error(f"{__file__} | Line: {e.__traceback__.tb_lineno} | An error occurred: {e} ")

main_run()

# args = args_parser()
file_name = './results/{}_{}.pkl'.format('logreg', args.optimizer)
file_name = os.path.join(current_work_dir, file_name)
with open(file_name, 'wb') as f:
    pkl.dump([weight_diff_list, obj_diff_list], f)
plot_logreg()