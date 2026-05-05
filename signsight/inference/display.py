"""Frame annotation utilities for the inference pipeline."""

from typing import Any

import cv2
from cv2.typing import MatLike

from ..const import BOX_PADDING

# Interface display options
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.25
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
CONNECTION_COLOR = (0, 255, 0)
LANDMARK_COLOR = (0, 0, 255)
LANDMARK_RADIUS = 4


# MediaPipe hand landmark connections
HAND_LANDMARKS = [
    [(0, 1), (1, 2), (2, 3), (3, 4)],  # thumb
    [(0, 5), (5, 6), (6, 7), (7, 8)],  # index finger
    [(0, 9), (9, 10), (10, 11), (11, 12)],  # middle finger
    [(0, 13), (13, 14), (14, 15), (15, 16)],  # ring finger
    [(0, 17), (17, 18), (18, 19), (19, 20)],  # pinky finger
    [(5, 9), (9, 13), (13, 17)],  # palm
]


def draw_prediction(
    frame: MatLike,
    predicted_class: str = "",
    confidence_score: float = 0.0,
    is_hand_detected: bool = True,
) -> MatLike:
    """Draw the predicted letter and confidence score onto the frame.

    Args:
        frame: BGR frame from the webcam.
        predicted_class: Predicted ASL letter.
        confidence: Confidence score as a percentage.
        is_hand_detected: Whether a hand is detected in the given frame or not.

    Returns:
        Frame with prediction overlay drawn on it.
    """

    # Show message when no hand is detected, otherwise show prediction
    if not is_hand_detected:
        label = "No hand detected"
    else:
        label = f"{predicted_class} ({confidence_score:.1f}%)"

    # Measure text size to draw a background rectangle behind it
    (text_width, text_height), baseline = cv2.getTextSize(
        label, FONT, FONT_SCALE, FONT_THICKNESS
    )

    # Draw background rectangle in top left corner
    cv2.rectangle(
        frame,
        (0, 0),
        (
            text_width + BOX_PADDING * 2,
            text_height + baseline + BOX_PADDING * 2,
        ),
        BACKGROUND_COLOR,
        thickness=cv2.FILLED,
    )

    # Draw prediction text over the background rectangle
    cv2.putText(
        frame,
        label,
        (BOX_PADDING, text_height + BOX_PADDING),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
    )

    return frame


def draw_landmarks(
    frame: MatLike,
    landmarks: Any,
    is_hand_detected: bool = True,
) -> MatLike:
    """Draw hand landmarks and connections onto the frame.

    Args:
        frame: BGR frame from the webcam.
        landmarks: MediaPipe hand landmarks.

    Returns:
        Frame with landmarks and connections drawn on it.
    """

    if not is_hand_detected:
        return frame

    height, width = frame.shape[:2]

    # Convert normalized landmark coordinates to pixel coordinates
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]

    # Draw connections between landmarks
    for start, end in [point for region in HAND_LANDMARKS for point in region]:
        cv2.line(frame, points[start], points[end], CONNECTION_COLOR, 2)

    # Draw landmark points on top of connections
    for point in points:
        cv2.circle(frame, point, LANDMARK_RADIUS, LANDMARK_COLOR, cv2.FILLED)

    return frame
