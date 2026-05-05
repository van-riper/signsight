"""Live inference loop for the prediction interface."""

import cv2
from torchvision import datasets

from ..const import DATASET_TRAIN_PATH, INFERENCE_INTERVAL
from .capture import open_camera_session, read_frame
from .detect import create_hand_detector, detect_hand
from .display import draw_landmarks, draw_prediction, draw_no_hand_message
from .predict import load_predictor, predict


def run_inference_loop() -> None:
    """Run the live inference loop."""

    model, device = load_predictor()
    dataset = datasets.ImageFolder(DATASET_TRAIN_PATH)
    class_names = dataset.classes
    detector = create_hand_detector()

    print("Running inference. Press 'q' to quit.")

    predicted_class = ""
    confidence = 0.0
    frame_count = 0

    cv2.namedWindow("SignSight", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("SignSight", 1280, 960)

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
