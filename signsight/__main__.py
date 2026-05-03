"""Main CLI interface."""

import sys

# Must use Python 3.12
if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"error: Python {VERSION} in use, SignSight requires Python 3.12")
    sys.exit(1)


from argparse import ArgumentParser
from pathlib import Path

from signsight.const import DATASET_ROOT_PATH
from signsight.core import evaluate_model, train_model

# Must be able to detect the dataset
if not Path(DATASET_ROOT_PATH).exists():
    print("error: extracted database not found:")
    print("\tThe ASL-HG database must be downloaded and")
    print("\textracted into the `data/` folder in this repo.")
    print("\tPlease consult the README for more information.")
    sys.exit(2)


def main() -> None:
    """Entry point for the SignSight CLI."""

    parser = ArgumentParser(prog="signsight")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train the model")
    subparsers.add_parser("eval", help="Evaluate the model")

    args = parser.parse_args()

    match args.command:
        case "train":
            train_model()
        case "eval":
            evaluate_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
