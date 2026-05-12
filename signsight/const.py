"""Global constants used by the program."""

# Core module

# 36 dataset classes (A-Z plus 0-9)
CLASS_COUNT: int = 36

# Images are scaled down to 128x128 pixels
IMAGE_SIZE: int = 128


# Inference module

# Set prediction box padding
BOX_PADDING: int = 20

# Run model inference once every N frames
INFERENCE_INTERVAL: int = 5
