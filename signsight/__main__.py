"""Main executable for the SignSight program."""

import sys

# TODO: relocate version check
# Must use Python 3.12
if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"error: Python {VERSION} in use, SignSight requires Python 3.12")
    sys.exit(1)


from argparse import ArgumentParser
from pathlib import Path

from signsight.const import DATASET_ROOT_PATH
from signsight.core import evaluate_model, get_device, train_model
from signsight.inference import run_inference_loop

# TODO: relocate path assertion logic to another helper module
# TODO: make more assertions for all the necessary paths
# Must be able to detect the dataset
if not Path(DATASET_ROOT_PATH).exists():
    print("error: extracted database not found:")
    print("\tThe ASL-HG database must be downloaded and")
    print("\textracted into the `data/` folder in this repo.")
    print("\tPlease consult the README for more information.")
    sys.exit(2)


# TODO: apply Google-style formatting to all function docstrings


def main() -> None:
    """Entry point for the SignSight CLI."""

    parser = ArgumentParser(prog="signsight")

    # TODO: add more flags, e.g. epoch count, learning rate, model architecture
    parser.add_argument(
        "-b",
        "--batch-size",
        action="store",
        dest="batch_size",
        type=int,
        default=(32 if get_device().type == "cpu" else 64),
        help="Set batch size (default: 32 on CPU, 64 on CUDA)",
    )

    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train the model")
    subparsers.add_parser("eval", help="Evaluate the model")
    subparsers.add_parser("run", help="Run live inference")

    args = parser.parse_args()

    match args.command:
        case "train":
            train_model(args.batch_size)
        case "eval":
            evaluate_model(args.batch_size)
        case "run":
            run_inference_loop()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
