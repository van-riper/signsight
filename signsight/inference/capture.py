"""Webcam feed and frame grabbing."""

from contextlib import contextmanager
from typing import Generator

from cv2 import VideoCapture, destroyAllWindows

type CameraSession = Generator[VideoCapture, None, None]


# Camera index for the built-in webcam
CAMERA_INDEX: int = 0


@contextmanager
def open_camera_session(index: int = CAMERA_INDEX) -> CameraSession:
    """Open the camera and guarantee cleanup on exit."""

    camera = VideoCapture(index)

    if not camera.isOpened():
        raise RuntimeError(f"error: camera at index {index} not found")

    try:
        yield camera
    finally:
        camera.release()
        destroyAllWindows()
