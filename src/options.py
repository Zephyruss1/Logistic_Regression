import argparse


def args_parser(test_case=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate for each update step"
    )
    parser.add_argument(
        "--optimizer", type=str, default="BFGS", help="using GD for update"
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=250,
        help="maximum update iterations if not exit automatically",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="penalty term for logistic regression",
    )

    if test_case:
        # For tests, return only known arguments
        known_args, _ = parser.parse_known_args()
        return known_args
    return parser