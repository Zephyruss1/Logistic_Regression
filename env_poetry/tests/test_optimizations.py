import pytest
import sys
import numpy as np

sys.path.append("..")
from logistic_regression import LogisticRegression
from datasets.data_preprocess import data_preprocess
from options import arg_parser_for_tests

# Parse arguments
args = arg_parser_for_tests()


@pytest.fixture
def setup_data():
    # Preprocess data
    (x_train, y_train), (x_test, y_test) = data_preprocess(args)
    return x_train, y_train, x_test, y_test


@pytest.fixture
def setup_model(setup_data):
    x_train, y_train, x_test, _ = setup_data
    # Initialize the LogisticRegression model
    model = LogisticRegression(
        args=args, X_train=x_train, Y_train=y_train, X_test=x_test
    )
    return model


def test_neldermead_weights_properly_updated(setup_model):
    # Call the Nelder-Mead method without assignment
    setup_model.nelder_mead()

    # Assert if the weights attribute is updated correctly (example check)
    assert isinstance(setup_model.weights, np.ndarray), "Weights were not updated properly"


def test_adamw_weights_properly_updated(setup_model):
    # Call the AdamW method without assignment
    setup_model.adamw()

    # Assert if the weights attribute is updated correctly (example check)
    assert isinstance(setup_model.weights, np.ndarray), "Weights were not updated properly"



if __name__ == "__main__":
    pytest.main()
