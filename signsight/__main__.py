"""
Main executable script for SignSight.
"""

import sys
from argparse import ArgumentParser

# Must use python 3.12
if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    print(f"error: python {sys.version} in use, python 3.12 is required")
    sys.exit(1)

from signsight.train import train_model


def main() -> None:
    """Entry point for the SignSight CLI."""

    parser = ArgumentParser(prog="signsight")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train the model")

    args = parser.parse_args()

    match args.command:
        case "train":
            train_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
