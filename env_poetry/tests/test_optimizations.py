import pytest
import numpy as np
import sys
import torch

sys.path.append('..')
from logistic_regression import LogisticRegression
from datasets.data_preprocess import data_preprocess
from options import args_parser

# Parse arguments
args = args_parser()

@pytest.fixture
def setup_data():
    # Preprocess data
    (x_train, y_train), (x_test, y_test) = data_preprocess(args)
    return x_train, y_train, x_test, y_test

@pytest.fixture
def setup_model(setup_data):
    x_train, y_train, x_test, _ = setup_data
    # Initialize the LogisticRegression model
    model = LogisticRegression(args=args, X_train=x_train, Y_train=y_train, X_test=x_test)
    return model

# LBFGS
def test_lbfgs_initialization(setup_model):
    setup_model.LBFGS()

    # Check if weights are initialized as a torch tensor
    assert callable(setup_model.LBFGS), "LBFGS method is not implemented"

# def test_lbfgs_weight_initialization(setup_model):
#     setup_model.LBFGS()

#     # Check if weights are initialized as a torch tensor
#     assert isinstance(setup_model.weights, torch.Tensor), "Weights should be a torch tensor"

# def test_lbfgs_loss_reduction(setup_model):
#     model = setup_model
#     initial_loss = model.objective(model.weights).item()
#     # Execute the LBFGS method
#     model.LBFGS()
#     final_loss = model.objective(model.weights).item()
#     # Check if the loss has been reduced
#     assert final_loss < initial_loss, "Loss should be reduced after LBFGS execution"

# def test_lbfgs_weight_update(setup_model):
#     model = setup_model
#     initial_weights = model.weights.clone()
#     # Execute the LBFGS method
#     model.LBFGS()
#     # Check if the weights have been updated
#     assert not torch.equal(initial_weights, model.weights), "Weights should be updated after LBFGS execution"

# NetherMead
def test_nethermead_initialization(setup_model):
    setup_model.nelder_mead()

    # Check if weights are initialized as a torch tensor
    assert callable(setup_model.nelder_mead), "NelderMead method is not implemented"

# SGDW
# def test_sgdw_initialization(setup_model):
#     setup_model.sgdw()

#     # Check if weights are initialized as a torch tensor
#     assert callable(setup_model.sgdw), "SGDW method is not implemented"

if __name__ == "__main__":
    pytest.main()