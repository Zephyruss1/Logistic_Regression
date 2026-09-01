# Logistic Regression & XGBoost (From Scratch)

This repository provides modular, high-performance implementations of **Logistic Regression** and **XGBoost** algorithms built entirely from scratch in Python. The codebase provides an in-depth, transparent look at the mathematical underpinnings, optimization algorithms, gradient-boosting mechanics, and tree-building logic of these machine learning techniques.

---

## Table of Contents

- [Features](#features)
  - [Logistic Regression](#logistic-regression)
  - [XGBoost (From Scratch)](#xgboost-from-scratch)
  - [Interactive Streamlit Web App](#interactive-streamlit-web-app)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Environment Setup with uv (Recommended)](#environment-setup-with-uv-recommended)
  - [Alternative Setup with pip](#alternative-setup-with-pip)
- [Dataset](#dataset)
- [Usage Guide](#usage-guide)
  - [1. CLI Training (main.py)](#1-cli-training-mainpy)
  - [2. Interactive Web Application (app.py)](#2-interactive-web-application-apppy)
  - [3. Hyperparameter Tuning with Optuna](#3-hyperparameter-tuning-with-optuna)
  - [4. Convergence & Weight Plotting](#4-convergence--weight-plotting)
- [Testing & Quality Assurance](#testing--quality-assurance)
- [CI/CD Pipeline](#cicd-pipeline)
- [Maintainers & Contributing](#maintainers--contributing)
  - [Development Setup & Dependency Groups](#development-setup--dependency-groups)
  - [Contribution Workflow](#contribution-workflow)

---

## Features

### Logistic Regression

Implements binary classification with **14 first-order, second-order, quasi-Newton, and derivative-free optimization methods**:

- **First-Order Optimizers:**
  - Standard Gradient Descent (`GD`)
  - Gradient Descent with Armijo Backtracking Line Search (`GDArmijo`)
  - Conjugate Gradient (`ConjugateGradient`) & with Armijo search (`ConjugateGDArmijo`)
  - Stochastic Gradient Descent (`SGD`) and SGD with decoupled weight decay (`SGDW`)
  - Momentum-based Optimizers: `Adam` and `AdamW` (with PyTorch tensor acceleration)
- **Second-Order & Quasi-Newton Methods:**
  - Modified Newton's Method (`ModifiedNewton`) & with Armijo search (`ModifiedNewtonArmijo`)
  - Levenberg-Marquardt Method (`LevenbergMarquardt`)
  - BFGS (`BFGS`) and Memory-efficient L-BFGS (`LBFGS`)
- **Derivative-Free Methods:**
  - Nelder-Mead Simplex Algorithm (`NelderMead`)
- **Analysis & Diagnostics:**
  - Ground truth comparison using **CVXPY** convex optimization solvers (ECOS, CVXOPT).
  - Dynamic **Armijo condition** step-size backtracking.
  - Convergence monitoring: Objective suboptimality ($f(x^{(k)}) - p^\star$), weight difference ($\frac{1}{\sqrt{d}}\|x^{(k)} - x^\star\|_2$), and Hessian eigenvalue spectrum diagnostics.

### XGBoost (From Scratch)

A custom Gradient Boosted Decision Tree (GBDT) engine featuring:

- **Tree Booster (`TreeBooster`):** Recursive binary tree construction using exact (`exact`) and approximate (`approx`) histogram-based split-finding algorithms.
- **Regularization & Shrinkage:**
  - L2 weight leaf regularization ($\lambda$ / `reg_lambda`)
  - Minimum loss reduction split penalty ($\gamma$ / `gamma`)
  - `min_child_weight` Hessian sum constraint
  - Step size shrinkage (`learning_rate`)
- **Subsampling:** Row subsampling (`subsample`) and column subsampling by node (`colsample_bynode`).
- **Pluggable Objective Interface:** Extensible loss architecture supporting custom first-order gradients and second-order Hessians (includes `SquaredErrorObjective`).
- **Bayesian Optimization:** Automated hyperparameter tuning powered by **Optuna**.
- **Model Serialization:** Save and load learned booster trees and weights.

### Interactive Streamlit Web App

- Modern dashboard to configure hyperparameters in real-time (`learning_rate`, `max_depth`, `subsample`, `reg_lambda`, `gamma`, `min_child_weight`, `tree_method`).
- Interactive training execution with live loss progression graphs powered by **Plotly**.
- $R^2$ score evaluation, predictions table viewer, and one-click CSV export.

---

## Project Structure

```text
Logistic_Regression/
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions CI (lint, format, test)
├── datasets/
│   ├── __init__.py
│   ├── data_preprocess.py             # MNIST parsing, binary filtering, normalization
│   └── mnist/                         # Binary MNIST dataset files
│       ├── t10k-images.idx3-ubyte
│       ├── t10k-labels.idx1-ubyte
│       ├── train-images.idx3-ubyte
│       └── train-labels.idx1-ubyte
├── scripts/
│   ├── __init__.py
│   ├── options.py                     # CLI argument parser and optimizer definitions
│   ├── others.py                      # Timing decorators, CLI input prompts, R² metric
│   ├── plot.py                        # Convergence curves and weight comparison plots
│   └── squared_error_objective.py     # Custom objective (loss, gradient, hessian)
├── src/
│   ├── find_best_parameters.py        # Optuna Bayesian hyperparameter search
│   ├── logistic_regression.py         # Core Logistic Regression & 14 optimizers
│   └── xgboost_scratch.py             # Core XGBoost & TreeBooster implementations
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py                # Tests for data loader shapes, dtypes, normalization
│   ├── test_optimizations.py          # Tests for Logistic Regression optimizers
│   ├── test_scripts.py                # Tests for helper scripts and metrics
│   └── test_xgboost.py                # Unit tests for booster splits, gradients, Hessians
├── .pre-commit-config.yaml            # Pre-commit hooks configuration
├── .python-version                    # Python version pin (3.12)
├── app.py                             # Streamlit interactive web application
├── LICENSE                            # License file
├── main.py                            # CLI entry point for training and evaluation
├── pyproject.toml                     # uv and Hatchling project specification
├── requirements.txt                   # Frozen requirements for pip compatibility
├── README.md                          # Project documentation
└── uv.lock                            # uv lockfile for deterministic builds
```

---

## Installation & Setup

### Prerequisites

- **Python**: `>= 3.12`
- **uv**: Fast Python package installer and resolver ([astral.sh/uv](https://astral.sh/uv/))

### Environment Setup with uv (Recommended)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Zephyruss1/Logistic_Regression.git
   cd Logistic_Regression
   ```

2. **Install dependencies and create virtual environment with uv:**
   ```bash
   uv sync
   ```

   To include development dependencies (pytest, ruff, pre-commit):
   ```bash
   uv sync --all-groups
   ```

3. **Activate the virtual environment (optional):**
   ```bash
   source .venv/bin/activate
   ```
   *(Or prepend any command with `uv run`, e.g., `uv run python main.py`)*

### Alternative Setup with pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Dataset

The project uses the **MNIST handwritten digit dataset** formatted as binary classification between digits `0` and `1`. Binary files are included in `datasets/mnist/`:

- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

The loader in `datasets/data_preprocess.py` parses IDX binary streams, filters selected classes, normalizes pixels to $[0, 1]$, and returns flattened matrices ready for training.

---

## Usage Guide

### 1. CLI Training (`main.py`)

Run the interactive training script:

```bash
uv run python main.py
```

You will be prompted to select:
1. **Logistic Regression** — Trains Logistic Regression using the specified optimizer, calculates Hessian eigenvalues, prints iteration errors, saves weights, and generates comparison plots.
2. **XGBoost** — Trains the from-scratch XGBoost model using default parameters or launches Optuna for hyperparameter search.
3. **Exit**

#### Customizing Logistic Regression via CLI flags:

```bash
uv run python main.py --optimizer BFGS --lr 0.1 --iteration 250 --gamma 0.1
```

**Available `--optimizer` choices:**
- `GD`, `GDArmijo`
- `ModifiedNewton`, `ModifiedNewtonArmijo`
- `ConjugateGradient`, `ConjugateGDArmijo`
- `LevenbergMarquardt`
- `BFGS`, `LBFGS`
- `Adam`, `AdamW`
- `SGD`, `SGDW`
- `NelderMead`

### 2. Interactive Web Application (`app.py`)

Launch the Streamlit web application:

```bash
uv run streamlit run app.py
```

Open your browser at `http://localhost:8501` to:
- Adjust hyperparameters on the sidebar in real time.
- Train the XGBoost model on MNIST.
- Inspect loss curves, $R^2$ scores, and sample predictions.
- Download prediction results as a CSV file.

### 3. Hyperparameter Tuning with Optuna

Tune XGBoost hyperparameters independently using Optuna:

```bash
uv run python src/find_best_parameters.py
```

### 4. Convergence & Weight Plotting

Generate convergence and weight comparison plots:

```bash
uv run python scripts/plot.py --comparison 1
```

Generated plots are saved in the `optimization_results/` directory.

---

## Testing & Quality Assurance

Run the complete unit test suite across data loading, optimization algorithms, helper utilities, and XGBoost logic:

```bash
uv run pytest
```

To run with verbose output:

```bash
uv run pytest tests/ -v
```

### Code Formatting & Linting

We use **Ruff** for high-speed linting and code formatting:

```bash
# Check code quality
uv run ruff check .

# Check formatting
uv run ruff format --check .

# Auto-format codebase
uv run ruff format .
```

---

## CI/CD Pipeline

Continuous Integration is set up via **GitHub Actions** (`.github/workflows/ci.yml`). On every push and pull request to `main`/`master`, the pipeline:
1. Provisions Python 3.12 and sets up `uv` caching via `astral-sh/setup-uv`.
2. Synchronizes dependencies with `uv sync`.
3. Runs `ruff check .` and `ruff format --check .`.
4. Executes the full `pytest` suite.

---

## Maintainers & Contributing

Contributions, issues, and feature requests are welcome!

### Development Setup & Dependency Groups

The project specifies its development tools in `pyproject.toml` using `[dependency-groups]`:

```toml
[dependency-groups]
dev = [
  "pytest>=8.3.3,<9.0.0",   # Test execution framework
  "ruff>=0.7.2,<0.8.0",     # Linter and code formatter
  "pre-commit>=4.6.2",      # Git hook management
]
```

To set up the development environment:

1. **Install dev dependencies:**
   ```bash
   uv sync --group dev
   # or install all dependency groups:
   uv sync --all-groups
   ```

2. **Set up pre-commit hooks:**
   Install the Git pre-commit hooks to automatically enforce Ruff linting and formatting on every commit:
   ```bash
   uv run pre-commit install
   ```

3. **Run hooks manually across all files:**
   ```bash
   uv run pre-commit run --all-files
   ```

### Contribution Workflow

1. Fork the repository and create a new feature branch (`git checkout -b feature/my-feature`).
2. Implement your changes, ensuring code is formatted and tested.
3. Run the linter and test suite before committing:
   ```bash
   uv run ruff check .
   uv run ruff format .
   uv run pytest
   ```
4. Commit your changes and push to your fork.
5. Submit a Pull Request targeting `main`.