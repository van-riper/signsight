"""Webcam feed and frame grabbing."""

from cv2 import VideoCapture, destroyAllWindows
from cv2.typing import MatLike

# Camera index for the built-in webcam
CAMERA_INDEX: int = 0


def open_camera() -> VideoCapture:
    """Open the default webcam and return the capture object."""

    camera = VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera at index {CAMERA_INDEX}.")

    return camera


def read_frame(camera: VideoCapture) -> tuple[bool, MatLike]:
    """Read a single frame from the camera.

    Returns:
        A tuple of a success flag and the frame. If the flag is False,
        the frame should be discarded.
    """

    return camera.read()


def close_camera(camera: VideoCapture) -> None:
    """Release the camera and close any OpenCV windows."""

    camera.release()

    destroyAllWindows()
