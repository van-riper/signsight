"""
Constants used by the training and interface pipelines.
"""

from pathlib import Path

# Paths to the dataset and weights
DATASET_PATH: str = str(
    Path("data/archive/asl_alphabet_train/asl_alphabet_train").resolve()
)
MODEL_PATH: str = str(Path("models/signsight.pth").resolve())

# Training parameters
BATCH_SIZE: int = 32
EPOCH_COUNT: int = 10
VAL_SPLIT: float = 0.2

# Total comes from the 26-letter alphabet plus SPACE, NOTHING, and DELETE
CLASS_COUNT: int = 29

# Images are scaled down to 64x64 pixels
IMAGE_SIZE: int = 64
