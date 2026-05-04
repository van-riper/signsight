"""Frame annotation utilities for the inference pipeline."""

import cv2

from ..const import ROI_PADDING

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.5
FONT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
BACKGROUND_COLOR = (0, 0, 0)
CONNECTION_COLOR = (0, 255, 0)
LANDMARK_COLOR = (0, 0, 255)
LANDMARK_RADIUS = 4


def draw_prediction(
    frame: cv2.typing.MatLike,
    predicted_class: str,
    confidence: float,
) -> cv2.typing.MatLike:
    """Draw the predicted letter and confidence score onto the frame.

    Args:
        frame: BGR frame from the webcam.
        predicted_class: Predicted ASL letter.
        confidence: Confidence score as a percentage.

    Returns:
        Frame with prediction overlay drawn on it.
    """

    label = f"{predicted_class} ({confidence:.1f}%)"

    # Measure text size to draw a background rectangle behind it
    (text_width, text_height), baseline = cv2.getTextSize(
        label, FONT, FONT_SCALE, FONT_THICKNESS
    )

    # Draw background rectangle in top left corner
    cv2.rectangle(
        frame,
        (0, 0),
        (
            text_width + ROI_PADDING * 2,
            text_height + baseline + ROI_PADDING * 2,
        ),
        BACKGROUND_COLOR,
        thickness=cv2.FILLED,
    )

    # Draw prediction text over the background rectangle
    cv2.putText(
        frame,
        label,
        (ROI_PADDING, text_height + ROI_PADDING),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
    )

    return frame


def draw_landmarks():
    pass


def show_no_hand_message():
    pass
