"""Main executable for the SignSight program."""

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

from signsight.const import (
    DATASET_ROOT_PATH,
    DATASET_TRAIN_PATH,
    INFERENCE_INTERVAL,
)
from signsight.core import evaluate_model, get_device, train_model
from signsight.inference import (
    create_hand_detector,
    detect_hand,
    draw_landmarks,
    draw_no_hand_message,
    draw_prediction,
    load_predictor,
    open_camera_session,
    predict,
    read_frame,
)

# TODO: relocate path assertion logic to another helper module
# TODO: make more assertions for all the necessary paths
# Must be able to detect the dataset
if not Path(DATASET_ROOT_PATH).exists():
    print("error: extracted database not found:")
    print("\tThe ASL-HG database must be downloaded and")
    print("\textracted into the `data/` folder in this repo.")
    print("\tPlease consult the README for more information.")
    sys.exit(2)


# TODO: relocate camera logic to another designated module
def _run_inference_pipeline() -> None:
    """Run the live inference loop."""

    model, device = load_predictor()
    dataset = datasets.ImageFolder(DATASET_TRAIN_PATH)
    class_names = dataset.classes
    detector = create_hand_detector()

    print("Running inference. Press 'q' to quit.")

    predicted_class = ""
    confidence = 0.0
    frame_count = 0

    with open_camera_session() as camera:
        while True:
            success, frame = read_frame(camera)

            if not success:
                print("warning: could not read frame")
                break

            roi, landmarks = detect_hand(detector, frame)

            if roi is None or landmarks is None:
                frame = draw_no_hand_message(frame)

            else:
                if frame_count % INFERENCE_INTERVAL == 0:
                    predicted_class, confidence = predict(
                        model, device, roi, class_names
                    )
                frame = draw_prediction(frame, predicted_class, confidence)
                frame = draw_landmarks(frame, landmarks)

            frame_count += 1
            cv2.imshow("SignSight", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


# TODO: apply Google-style formatting to all function docstrings


def main() -> None:
    """Entry point for the SignSight CLI."""

    parser = ArgumentParser(prog="signsight")

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
            _run_inference_pipeline()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
