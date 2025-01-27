# Logistic Regression & XGBoost (From Scratch)
This repository provides implementations of **Logistic Regression** and **XGBoost** algorithms built entirely from scratch. The intent is to offer a detailed look at the underlying mechanisms of these machine learning techniques, enabling a deeper understanding and customization for research or educational purposes.

## Features
### Logistic Regression (with multiple optimization algorithms)
- Implements various optimization methods for parameter updates, including gradient descent, Newton's method, and advanced methods like **Adam**, **AdamW**, and **BFGS**.
- Supports **Armijo step-size search**, which dynamically adjusts the learning rate for better convergence.
- Uses **CVXPY** for solving optimal weights for comparative analysis.
- Extensive support for monitoring error differences (`weight_diff` and `objective_diff`) and Hessian eigenvalues.

### XGBoost (Simplified Implementation)
- Custom implementation of the Gradient Boosting Decision Tree (GBDT) model for supervised learning tasks.
- Implements tree splitting, depth control, gradient calculation, and Hessian-based optimization.
- Supports hyperparameter tuning such as:
    - `learning_rate`
    - `max_depth`
    - `subsample`
    - `reg_lambda` (L2 regularization)
    - `gamma` (Split regularization)
    - `min_child_weight`

- Fully compatible with custom loss functions (e.g., squared error implementations).
- Ability to predict and evaluate using an **objective interface** for gradient boosting.

## Files Overview
### Core Models
- **`logistic_regression.py`:** Implements the core **Logistic Regression** model with different optimization strategies.
- **`xgboost_scratch.py`:** Contains the simplified **XGBoost** implementation, including tree-building logic (`TreeBooster`) and the boosting mechanism (`XGBoostModel`).

### Main Execution
- **`main.py`:** Main execution script to train Logistic Regression or XGBoost models, with configurable hyperparameters and interactive CLI input.

### Dataset Management
- **`data_preprocess.py`:** Processes **MNIST handwritten digit dataset** for training and testing (focuses on binary classification for selected digits).
- **Directory:** Contains MNIST binary data files required for model training.

### Testing Modules
- **`test_xgboost.py`:** Unit tests for the XGBoost implementation, including gradient, Hessian, splitting logic, and model predictions.
- **`test_optimizations.py`:** Tests to validate optimization techniques for Logistic Regression.
- **`test_dataset.py`:** Validates the loading, preprocessing, and formatting logic implemented in `data_preprocess.py`.

### Hyperparameter Tuning
- **`find_best_parameters.py`:** Utilizes **Optuna** to perform hyperparameter tuning for the XGBoost model.

### Visualization
- **`plot.py`:** Generates comparison plots for Logistic Regression, comparing weights and objectives across optimization methods (e.g., AdamW, SGD, etc.).

### Arguments/Configurations
- **`options.py`:** Handles command-line argument parsing for hyperparameter configurations for both Logistic Regression and XGBoost.

### Web Application
- **`app.py`:** Streamlit-based web application for interactive model training and prediction.

## Installation
### Prerequisites
Ensure the following dependencies are installed:
- Python 3.12 or newer
- Required libraries: `numpy`, `pandas`, `cvxpy`, `optuna`, `scikit-learn`, `xgboost`, `matplotlib`, `pytest`, `torch`, `streamlit`

Install dependencies using:
``` bash
poetry install 
```

## How to Use
### Dataset Preparation
The project uses the **MNIST dataset** (for handwritten digit classification) available in `./mnist/`. Ensure you download ⬇️ the binary MNIST files and place them correctly:
- `train-labels.idx1-ubyte`
- `train-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`

### Running Models
1. Ensure proper dependencies are installed.
2. Run `main.py` to train models.
    - The default is **XGBoost** using predefined hyperparameters.
    - Interactive mode allows switching between **Logistic Regression** and **XGBoost** during runtime.
``` bash
python main.py
```
**Example Workflow:**
1. Select Model: Logistic Regression or XGBoost.
2. Specify model hyperparameters via CLI or use defaults.
3. View metrics such as loss, weights, and testing accuracy.

### Running the Web Application
1. Ensure proper dependencies are installed.
2. Run `app.py` to start the Streamlit web application.
``` bash
streamlit run app.py
```
3. Use the web interface to interactively train and predict using the models.

### Hyperparameter Tuning with Optuna
Fine-tune hyperparameters for XGBoost using `find_best_parameters.py`.
Run:
``` bash
python find_best_parameters.py
```

### Tests
Run unit tests for all modules using `pytest`:
``` bash
pytest
```

## Key Functionalities
### Logistic Regression:
1. Supports various optimizers:
    - **Gradient Descent**: Standard gradient-based updates.
    - **Modified Newton**: Leverages the Hessian matrix for curvature adjustment.
    - **Conjugate Gradient**: Gradient direction combined with prior steps for faster convergence.
    - **Adam & AdamW**: Advanced momentum-based optimization.
    - **Stochastic Gradient Descent (SGD, SGD-W)**: Efficient stochastic updates.

2. Pre-determines the optimal solution using **CVXPY** for comparison.
3. Debug capabilities to ensure convergence properties hold theoretically (e.g., Hessian eigenvalue computations).

### XGBoost:
1. Builds decision trees using **gradient** and **Hessian-based optimizations**.
2. Implements:
    - **Regularization:** L2 regularization (`reg_lambda`) and split penalty (`gamma`).
    - **Subsampling:** Improves generalization by using random subsets.
    - **Tree depth limitations:** Prevents overfitting on training data.

3. Customizable loss functions (Squared Error provided as an example).

## Results Visualization
1. Logistic Regression:
    - Generates weight and loss comparison plots for various optimization methods.

2. XGBoost:
    - Displays training metrics (e.g., training loss reduction across boosting rounds).
    - Can integrate visualizations for hyperparameter tuning outcomes.

## Maintainers
- Built as an educational and practical implementation of core machine learning techniques.
- Feel free to contribute by creating pull requests or raising issues.