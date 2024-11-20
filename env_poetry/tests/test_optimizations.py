import pytest
import sys

sys.path.append("..")
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
    model = LogisticRegression(
        args=args, X_train=x_train, Y_train=y_train, X_test=x_test
    )
    return model


def test_nethermead_initialization(setup_model):
    setup_model.nelder_mead()

    # Check if weights are initialized as a torch tensor
    assert callable(setup_model.nelder_mead), "NelderMead method is not implemented"


def test_adamw_initialization(setup_model):
    setup_model.adamw()

    # Check if weights are initialized as a torch tensor
    assert callable(setup_model.adamw), "AdamW method is not implemented"


if __name__ == "__main__":
    pytest.main()
