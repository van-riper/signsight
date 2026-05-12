"""Path objects and existence checks."""

from pathlib import Path

# Resolved paths as Path objects

# Paths to the dataset subfolders
DATASET_ROOT_PATH: Path = Path("data/ASL_HG_36000").resolve()
DATASET_RAW_PATH: Path = DATASET_ROOT_PATH / "asl_dataset"
DATASET_TRAIN_PATH: Path = DATASET_ROOT_PATH / "asl_processed/train"
DATASET_TEST_PATH: Path = DATASET_ROOT_PATH / "asl_processed/test"

# TODO: acommodate multiple model weight files
# Paths to the weights and landmark models
MODEL_PATH: Path = Path("models/signsight.pth").resolve()
HAND_LANDMARKER_PATH: Path = Path("models/hand_landmarker.task").resolve()

# Path to the confusion matrix image file
FIGURE_PATH: Path = Path("confusion_matrix.png").resolve()


def assert_paths() -> None:
    """Check if all necessary path objects exist, throw error if not."""

    necessary_paths = (
        DATASET_ROOT_PATH,
        DATASET_RAW_PATH,
        DATASET_TRAIN_PATH,
        DATASET_TEST_PATH,
        HAND_LANDMARKER_PATH,
    )

    for path in necessary_paths:
        if not path.exists():
            raise FileNotFoundError(path)
