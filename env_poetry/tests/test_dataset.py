import pytest
import numpy as np
import sys
sys.path.append("..")

from datasets.data_preprocess import data_preprocess
from options import args_parser

# Parse arguments with defaults
args = args_parser()

@pytest.fixture
def setup_data():
    """Fixture to preprocess and return data."""
    # Preprocess data using the default args
    (x_train, y_train), (x_test, y_test) = data_preprocess(args)
    return (x_train, y_train), (x_test, y_test)

def test_setup_data():
    """Check if dataset callable returns a dataset."""
    assert callable(data_preprocess)

def test_astype_data(setup_data):
    """Check if the datatype is `float32`."""
    (x_train, y_train), (x_test, y_test) = setup_data
    assert (x_train.dtype == np.float32 and x_test.dtype == np.float32), (
        "Data type is not `float32`"
    )
    assert (y_train.dtype == np.uint8 and y_test.dtype == np.uint8), (
        "Data type is not `uint8`"
    )

if __name__ == "__main__":
    pytest.main()
