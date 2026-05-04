"""Global constants used by the program."""

from pathlib import Path

# Resolved paths set as strings

# Paths to the dataset subfolders
DATASET_ROOT_PATH: Path = Path("data/ASL_HG_36000").resolve()
DATASET_RAW_PATH: str = str(Path(DATASET_ROOT_PATH / "asl_dataset"))
DATASET_TRAIN_PATH: str = str(Path(DATASET_ROOT_PATH / "asl_processed/train"))
DATASET_TEST_PATH: str = str(Path(DATASET_ROOT_PATH / "asl_processed/test"))

# Paths to the weights and landmark models
MODEL_PATH: str = str(Path("models/signsight.pth").resolve())
HAND_LANDMARKER_PATH: str = str(Path("models/hand_landmarker.task").resolve())

# Path to the confusion matrix image file
FIGURE_PATH: str = str(Path("confusion_matrix.png").resolve())


# Core module

# 36 dataset classes (A-Z plus 0-9)
CLASS_COUNT: int = 36

# Images are scaled down to 64x64 pixels
IMAGE_SIZE: int = 128


# Inference module

# Inference ROI box padding
ROI_PADDING: int = 20

# Run model inference once every N frames
INFERENCE_INTERVAL: int = 10
