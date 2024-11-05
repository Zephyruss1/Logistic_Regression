import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate for each update step"
    )
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", help="using GD for update"
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

    args = parser.parse_args()
    return args


def arg_parser_for_tests():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", type=float, default=0.1, help="learning rate for each update step"
    )
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", help="using GD for update"
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

    known_args = parser.parse_known_args()
    return known_args
