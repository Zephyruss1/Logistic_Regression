import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from matplotlib import rcParams
from options import args_parser

rcParams.update({'font.size': 18, 'text.usetex': True})

def plot_logreg():
    """
        Plot the logistic regression weights and objective values.

        This function generates and saves plots comparing the weights and objective values
        of different optimization algorithms used in logistic regression. It reads the
        precomputed weights and objective values from pickle files and creates visual
        comparisons. The plots are saved as PNG files in the `./results` directory.

        The function handles the following optimizers:
        - GD
        - GDArmijo
        - ModifiedNewton
        - ModifiedNewtonArmijo
        - ConjugateGradient
        - ConjugateGDArmijo
        - LevenbergMarquardt
        - BFGS
        - Adam
        - AdamW
        - SGD
        - SGDW

        If the required pickle files are not found, it prompts the user to run `main.py`
        with the appropriate optimization settings.

        Raises:
            FileNotFoundError: If the required pickle files are not found.
            ValueError: If an invalid optimizer is specified in the arguments.
    """
    args = args_parser()
    rcParams.update({'text.usetex': False})
    logreg_dimension = 785

    try:
        logreg_GD_weights, logreg_GD_objective =\
            pkl.load(open('/results/logreg_GD.pkl', 'rb'))
        logreg_GDArmijo_weights, logreg_GDArmijo_objective =\
            pkl.load(open('/results/logreg_GDArmijo.pkl', 'rb'))
        logreg_BFGS_weights, logreg_BFGS_objective =\
            pkl.load(open('/results/logreg_BFGS.pkl', 'rb'))
        logreg_modifiedNewton_weights, logreg_modifiedNewton_objective =\
            pkl.load(open('/results/logreg_ModifiedNewton.pkl', 'rb'))
        logreg_modifiedNewtonArmijo_weights, logreg_modifiedNewtonArmijo_objective =\
            pkl.load(open('/results/logreg_ModifiedNewtonArmijo.pkl', 'rb'))
        logreg_levenbergMarquardt_weights, logreg_levenbergMarquardt_objective =\
            pkl.load(open('/results/logreg_LevenbergMarquardt.pkl', 'rb'))
        logreg_ConjugateGradient_weights, logreg_ConjugateGradient_objective =\
            pkl.load(open('/results/logreg_ConjugateGradient.pkl', 'rb'))
        logreg_ConjugateGradientArmijo_weights, logreg_ConjugateGradientArmijo_objective = \
            pkl.load(open('/results/logreg_ConjugateGDArmijo.pkl', 'rb'))
        logreg_adam_weights, logreg_adam_objective =\
            pkl.load(open('/results/logreg_adam.pkl', 'rb'))
        logreg_adamw_weights, logreg_adamw_objective =\
            pkl.load(open('/results/logreg_AdamW.pkl', 'rb'))
        logreg_sgd_weights, logreg_sgd_objective =\
            pkl.load(open('/results/logreg_sgd.pkl', 'rb'))
        logreg_sgdw_weights, logreg_sgdw_objective =\
            pkl.load(open('/results/logreg_sgdw.pkl', 'rb'))
    except FileNotFoundError:
        print("[PLOT ERROR]: File not found. Please run main.py\
            first with relating optimization.")
        return

    plt.figure()
    try:
        if logreg_adam_weights is not None\
            and logreg_adamw_weights is not None\
            and logreg_sgd_weights is not None\
            and logreg_sgdw_weights is not None:
            indices1 = range(len(logreg_adam_weights))
            indices2 = range(len(logreg_adamw_weights))
            indices3 = range(len(logreg_sgd_weights))
            indices4 = range(len(logreg_sgdw_weights))

            plt.figure(figsize=(10, 6))

            plt.plot(indices1, logreg_adam_weights, label='Adam')
            plt.plot(indices2, logreg_adamw_weights, label='AdamW')
            plt.plot(indices3, logreg_sgd_weights, label='SGD')
            plt.plot(indices4, logreg_sgdw_weights, label='SGDW')

            plt.title('Model Weights Comparison')
            plt.xlabel('Weight Index')
            plt.ylabel('Weight Value')
            plt.legend()
            plt.savefig('./results/comparison_weights.png', dpi=1200)

        if logreg_adam_objective is not None\
            and logreg_adamw_objective is not None\
            and logreg_sgd_objective is not None\
            and logreg_sgdw_objective is not None:
            _indices1 = range(len(logreg_adam_objective))
            _indices2 = range(len(logreg_adamw_objective))
            _indices3 = range(len(logreg_sgd_objective))
            _indices4 = range(len(logreg_sgdw_objective))

            plt.figure(figsize=(10, 6))

            plt.plot(_indices1, logreg_adam_objective, label='Adam')
            plt.plot(_indices2, logreg_adamw_objective, label='AdamW')
            plt.plot(_indices3, logreg_sgd_objective, label='SGD')
            plt.plot(_indices4, logreg_sgdw_objective, label='SGDW')

            plt.title('Model Objective Comparison')
            plt.xlabel('Objective Index')
            plt.ylabel('Objective Value')
            plt.legend()
            plt.savefig('./results/comparison_objective.png',
                        dpi=1200)
    except FileNotFoundError:
        raise FileNotFoundError("File not found. Please run main.py first with relating optimization.")

    # Optimizer objectives
    plot_objectives = {
        'GD'                    : logreg_GD_objective,
        'GDArmijo'              : logreg_GDArmijo_objective,
        'ModifiedNewton'        : logreg_modifiedNewton_objective,
        'ModifiedNewtonArmijo'  : logreg_modifiedNewtonArmijo_objective,
        'ConjugateGradient'     : logreg_ConjugateGradient_objective,
        'ConjugateGDArmijo'     : logreg_ConjugateGradientArmijo_objective,
        'LevenbergMarquardt'    : logreg_levenbergMarquardt_objective,
        'BFGS'                  : logreg_BFGS_objective,
        'Adam'                  : logreg_adam_objective,
        'AdamW'                 : logreg_adamw_objective,
        'SGD'                   : logreg_sgd_objective,
        'SGDW'                  : logreg_sgdw_objective
    }

    if args.optimizer in plot_objectives:
        optimizer_objective = plot_objectives[args.optimizer]
        plt.plot(range(len(optimizer_objective)), np.array(optimizer_objective)\
                / np.sqrt(logreg_dimension),
                 label=args.optimizer)
        plt.savefig(f'./results/logreg_objectives_{args.optimizer}.png',
                    dpi=1200)
    else:
        raise ValueError("Invalid optimizer: {}".format(args.optimizer))

    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$\frac{1}{\sqrt{d}}\|x^{(k)}-x^{\star}\|_2$')
    plt.title('Logistic Regression weights')

    plt.yscale('log')
    plt.tight_layout()
    plt.show()
    plt.pause(5)
    plt.figure()
    # Optimizer Weights
    plot_weights = {
        'GD'                    : logreg_GD_weights,
        'GDArmijo'              : logreg_GDArmijo_weights,
        'ModifiedNewton'        : logreg_modifiedNewton_weights,
        'ModifiedNewtonArmijo'  : logreg_modifiedNewtonArmijo_weights,
        'ConjugateGradient'     : logreg_ConjugateGradient_weights,
        'ConjugateGDArmijo'     : logreg_ConjugateGradientArmijo_weights,
        'LevenbergMarquardt'    : logreg_levenbergMarquardt_weights,
        'BFGS'                  : logreg_BFGS_weights,
        'Adam'                  : logreg_adam_weights,
        'AdamW'                 : logreg_adamw_weights,
        'SGD'                   : logreg_sgd_weights,
        'SGDW'                  : logreg_sgdw_weights
    }
    if args.optimizer in plot_weights:
        optimizer_weights = plot_weights[args.optimizer]
        plt.plot(range(len(optimizer_weights)), optimizer_weights,
                label=args.optimizer)
        plt.savefig(f'./results/logreg_weights_{args.optimizer}.png',
                    dpi=1200)
    else:
        raise ValueError("Invalid optimizer: {}".format(args.optimizer))
    plt.legend()

    plt.xlabel('Iterations')
    plt.ylabel(r'$f(x^{(k)}) - p^{\star}$')
    plt.title('Logistic Regression objective')

    plt.yscale('log')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_logreg()