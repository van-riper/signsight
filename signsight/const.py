"""Global constants used by the program."""

# Core module

# 36 dataset classes (A-Z)
CLASS_COUNT: int = 26

# Exclude the number classes from training
EXCLUDED_CLASSES = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

# Images are scaled down to 128x128 pixels
IMAGE_SIZE: int = 128


# Inference module

# Set prediction box padding
BOX_PADDING: int = 20

# Run model inference once every N frames
INFERENCE_INTERVAL: int = 5
