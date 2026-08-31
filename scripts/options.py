import argparse

AVAILABLE_OPTIMIZERS: list[str] = [
  "GD",
  "ModifiedNewton",
  "ModifiedNewtonArmijo",
  "ConjugateGradient",
  "ConjugateGDArmijo",
  "LevenbergMarquardt",
  "BFGS",
  "LBFGS",
  "GDArmijo",
  "Adam",
  "AdamW",
  "SGD",
  "SGDW",
  "NelderMead",
]


def args_parser(test_case=False) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--lr", type=float, default=0.1, help="learning rate for each update step")
  parser.add_argument(
    "--optimizer",
    type=str,
    default="BFGS",
    help=f"Optimization algorithm (options: {', '.join(AVAILABLE_OPTIMIZERS)})",
    choices=AVAILABLE_OPTIMIZERS,
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
  parser.add_argument(
    "--comparison",
    type=int,
    default=0,
    help="if 1, compare different optimization algorithms",
  )
  if test_case:
    # For tests, return only known arguments
    known_args, _ = parser.parse_known_args()
    return known_args
  return parser.parse_args()
