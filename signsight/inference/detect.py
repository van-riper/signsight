"""Hand detection and ROI extraction using MediaPipe."""

# from typing import Any

from typing import Any

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ..const import HAND_LANDMARKER_PATH


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


def detect_hand():
    pass


def _crop_roi():
    pass


def _apply_hand_mask():
    pass
