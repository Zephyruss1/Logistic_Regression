import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
from matplotlib import rcParams
from options import args_parser
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
PKL_PATH = ROOT_PATH / "optimization_results"

rcParams.update({"font.size": 18, "text.usetex": True})

OPTIMIZER_PKL_FILES = {
    "GD": "logreg_GD.pkl",
    "GDArmijo": "logreg_GDArmijo.pkl",
    "LBFGS": "logreg_LBFGS.pkl",
    "BFGS": "logreg_BFGS.pkl",
    "ModifiedNewton": "logreg_ModifiedNewton.pkl",
    "ModifiedNewtonArmijo": "logreg_ModifiedNewtonArmijo.pkl",
    "LevenbergMarquardt": "logreg_LevenbergMarquardt.pkl",
    "ConjugateGradient": "logreg_ConjugateGradient.pkl",
    "ConjugateGDArmijo": "logreg_ConjugateGDArmijo.pkl",
    "Adam": "logreg_Adam.pkl",
    "AdamW": "logreg_AdamW.pkl",
    "SGD": "logreg_SGD.pkl",
    "SGDW": "logreg_SGDW.pkl",
    "NelderMead": "logreg_NelderMead.pkl",

}
COMPARISON_GROUP = ["Adam", "AdamW", "SGD", "SGDW"]


def load_result(name: str):
    path = PKL_PATH / OPTIMIZER_PKL_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"[PLOT WARNING]: '{path.name}' not found. "
            f"Run main.py with optimizer='{name}' first."
        )
    with open(path, "rb") as f:
        return pkl.load(f)

def plot_comparison(results, attr_index, title, xlabel, ylabel, filename):
    """attr_index: 0 = weights series, 1 = objective series"""
    if not all(name in results for name in COMPARISON_GROUP):
        return  # skip silently
    plt.figure(figsize=(10, 6))
    for name in COMPARISON_GROUP:
        series = results[name][attr_index]
        plt.plot(range(len(series)), series, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PKL_PATH / filename, dpi=1200)


def plot_logreg():
    """
    Plot the logistic regression weights and objective values.

    This function generates and saves plots comparing the weights and objective values
    of different optimization algorithms used in logistic regression. It reads the
    precomputed weights and objective values from pickle files and creates visual
    comparisons. The plots are saved as PNG files in the `./optimization_results` directory.

    The function handles the following optimizers:
    - GD
    - GDArmijo
    - ModifiedNewton
    - ModifiedNewtonArmijo
    - ConjugateGradient
    - ConjugateGDArmijo
    - LevenbergMarquardt
    - LBFGS
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
    rcParams.update({"text.usetex": False})
    logreg_dimension = 785
    if args.comparison == 1:
        try: results = {name: load_result(name) for name in OPTIMIZER_PKL_FILES}
        except FileNotFoundError:
            raise FileNotFoundError(
                '[PLOT WARNING]: File not found.'
            )
        except Exception as e:
            raise ValueError(f"[PLOT ERROR]:If you are running this script alone,\
                please run 'main.py' first with relating optimization settings: {str(e)}")
    else: results = {args.optimizer: load_result(args.optimizer)}
    
    plt.figure()

    try:
        plot_comparison(results, 0, "Model Weights Comparison",
                        "Weight Index", "Weight Value", "comparison_weights.png")
        plot_comparison(results, 1, "Model Objective Comparison",
                        "Objective Index", "Objective Value", "comparison_objective.png")
    except FileNotFoundError: raise FileNotFoundError("File not found. Please run main.py first with relating optimization.")
    except Exception as e: raise ValueError(f"[PLOT ERROR]: Error occurred while plotting: {e}")


    if args.optimizer not in results: raise ValueError(f"Invalid optimizer: {args.optimizer}")

    optimizer_weights, optimizer_objective = results[args.optimizer]

    # --- Single-optimizer plot #1 ---
    plt.figure()
    plt.plot(
        range(len(optimizer_objective)),
        np.array(optimizer_objective) / np.sqrt(logreg_dimension),
        label=args.optimizer,
    )
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel(r"$\frac{1}{\sqrt{d}}\|x^{(k)}-x^{\star}\|_2$")
    plt.title("Logistic Regression weights")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(PKL_PATH / f"logreg_objectives_{args.optimizer}.png", dpi=1200)
    plt.show()
    plt.pause(5)

    # --- Single-optimizer plot #2 ---
    plt.figure()
    plt.plot(range(len(optimizer_weights)), optimizer_weights, label=args.optimizer)
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel(r"$f(x^{(k)}) - p^{\star}$")
    plt.title("Logistic Regression objective")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(PKL_PATH / f"logreg_weights_{args.optimizer}.png", dpi=1200)
    plt.show()

if __name__ == "__main__":
    plot_logreg()
