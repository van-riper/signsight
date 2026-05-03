"""Hand detection and ROI extraction using MediaPipe."""

from typing import Any

import cv2
import mediapipe as mp
from cv2.typing import MatLike
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ..const import HAND_LANDMARKER_PATH, ROI_PADDING

type CoordTuple = tuple[MatLike, tuple[int, int]]


def create_hand_detector() -> Any:
    """Create and return a MediaPipe hand landmarker detector."""

    base_options = mp_python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    return mp_vision.HandLandmarker.create_from_options(options)


def detect_hand(detector: Any, frame: MatLike):
    """Detect a hand in a frame and return the masked ROI and landmarks.

    Args:
        detector: MediaPipe hand landmarker instance.
        frame: BGR frame from the webcam.

    Returns:
        A tuple of the masked hand ROI and landmarks. Both are None if
        no hand is detected.
    """

    height, width = frame.shape[:2]

    # MediaPipe expects RGB, OpenCV provides BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    results = detector.detect(mp_image)

    if not results.hand_landmarks:
        return None, None

    landmarks = results.hand_landmarks[0]

    roi, roi_origin = _crop_roi(frame, landmarks)
    # masked_roi = _apply_hand_mask(roi, landmarks, (height, width), roi_origin)
    # return masked_roi, landmarks


def _crop_roi(frame: MatLike, landmarks: Any) -> CoordTuple:
    """Crop the hand region from the frame using landmark bounding box.

    Args:
        frame: BGR frame from the webcam.
        landmarks: MediaPipe hand landmarks.

    Returns:
        Cropped and padded region of interest containing the hand.
    """

    height, width = frame.shape[:2]

    x_coords = [lm.x * width for lm in landmarks]
    y_coords = [lm.y * height for lm in landmarks]

    # Compute bounding box with padding, clamped to frame boundaries
    x_min = max(0, int(min(x_coords)) - ROI_PADDING)
    x_max = min(width, int(max(x_coords)) + ROI_PADDING)
    y_min = max(0, int(min(y_coords)) - ROI_PADDING)
    y_max = min(height, int(max(y_coords)) + ROI_PADDING)

    return frame[y_min:y_max, x_min:x_max], (x_min, y_min)


def _apply_hand_mask():
    pass
