"""Main CLI interface."""

import sys

# Must use Python 3.12
if sys.version_info < (3, 12) or sys.version_info >= (3, 13):
    VERSION = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"error: Python {VERSION} in use, SignSight requires Python 3.12")
    sys.exit(1)


from argparse import ArgumentParser
from pathlib import Path

import cv2
from torchvision import datasets

from signsight.const import DATASET_ROOT_PATH, DATASET_TRAIN_PATH
from signsight.core import evaluate_model, train_model
from signsight.inference import (
    close_camera,
    create_hand_detector,
    detect_hand,
    draw_landmarks,
    draw_no_hand_message,
    draw_prediction,
    load_predictor,
    open_camera,
    predict,
    read_frame,
)

# Must be able to detect the dataset
if not Path(DATASET_ROOT_PATH).exists():
    print("error: extracted database not found:")
    print("\tThe ASL-HG database must be downloaded and")
    print("\textracted into the `data/` folder in this repo.")
    print("\tPlease consult the README for more information.")
    sys.exit(2)


def _run_inference() -> None:
    """Run the live inference loop."""

    model, device = load_predictor()

    # Load class names from dataset folder structure
    dataset = datasets.ImageFolder(DATASET_TRAIN_PATH)
    class_names = dataset.classes

    camera = open_camera()
    detector = create_hand_detector()

    print("Running inference. Press 'q' to quit.")

    try:
        while True:
            success, frame = read_frame(camera)

            if not success:
                print("Error: could not read frame.")
                break

            roi, landmarks = detect_hand(detector, frame)

            if roi is None or landmarks is None:
                frame = draw_no_hand_message(frame)
            else:
                predicted_class, confidence = predict(
                    model, device, roi, class_names
                )
                frame = draw_prediction(frame, predicted_class, confidence)
                frame = draw_landmarks(frame, landmarks)

            cv2.imshow("SignSight", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        close_camera(camera)


def main() -> None:
    """Entry point for the SignSight CLI."""

    parser = ArgumentParser(prog="signsight")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("train", help="Train the model")
    subparsers.add_parser("eval", help="Evaluate the model")
    subparsers.add_parser("run", help="Run live inference")

    args = parser.parse_args()

    match args.command:
        case "train":
            train_model()
        case "eval":
            evaluate_model()
        case "run":
            _run_inference()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
