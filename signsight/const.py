"""Global constants used by the program."""

from pathlib import Path

# Resolved paths set as strings

# Paths to the dataset subfolders
DATASET_ROOT_PATH: Path = Path("data/ASL_HG_36000").resolve()
DATASET_RAW_PATH: str = str(Path(DATASET_ROOT_PATH / "asl_dataset"))
DATASET_TRAIN_PATH: str = str(Path(DATASET_ROOT_PATH / "asl_processed/train"))
DATASET_TEST_PATH: str = str(Path(DATASET_ROOT_PATH / "asl_processed/test"))

# Paths to the weights and confusion matrix figure
MODEL_PATH: str = str(Path("models/signsight.pth").resolve())
FIGURE_PATH: str = str(Path("confusion_matrix.png").resolve())

# Path to the land landmarker task file
HAND_LANDMARKER_PATH = "models/hand_landmarker.task"


# Training loop lasts 10 epochs
EPOCH_COUNT: int = 10

# Each batch contains 32 image-label pairs
BATCH_SIZE: int = 32

# Reserve 20% of the dataset for VALidation
VAL_SPLIT: float = 0.2

# Total comes from the 26-letter alphabet plus SPACE, NOTHING, and DELETE
CLASS_COUNT: int = 29

# Images are scaled down to 64x64 pixels
IMAGE_SIZE: int = 128

# Camera index for the built-in webcam
CAMERA_INDEX: int = 0

# Inference box padding
BOX_PADDING: int = 20
