"""Functions for model building, image transforms, and batch progress."""

from datetime import timedelta

import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets, models, transforms

from ..const import CLASS_COUNT, DATASET_RAW_PATH, IMAGE_SIZE

# Reserve 20% of the dataset for validation testing
VAL_SPLIT: float = 0.2


def get_device() -> torch.device:
    """Detect CUDA device if available, otherwise use CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform(training: bool) -> transforms.Compose:
    """Build the image preprocessing pipeline.

    Note:
        Augmentation is only applied during training, not inference.
    """

    base = [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
    ]

    augmentation = []

    if training:
        augmentation = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
        ]

    common = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]

    return transforms.Compose(base + augmentation + common)


def build_model(pretrained: bool) -> torch.nn.Module:
    """Build the model weights."""

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)

    model.fc = torch.nn.Linear(model.fc.in_features, CLASS_COUNT)

    return model


def load_model(path: str, device: torch.device) -> torch.nn.Module:
    """Load saved model weights from disk."""

    model = build_model(pretrained=False)

    # Ensure GPU-trained weights load correctly on CPU with map_location
    model.load_state_dict(torch.load(path, map_location=device))

    # Move the model to the right device
    model.to(device)

    model.eval()

    return model


def split_dataset() -> list[Subset]:
    """Split the dataset into training and validation subsets."""

    transform = get_transform(training=True)

    dataset = datasets.ImageFolder(DATASET_RAW_PATH, transform)

    dataset_val_size = int(len(dataset) * VAL_SPLIT)

    return random_split(
        dataset, [len(dataset) - dataset_val_size, dataset_val_size]
    )


def print_batch_progress(batch_counter: int, batch_total: int) -> None:
    """Print batch training/evaluation progress."""

    # Whitespace padding in progress quotient and percentage
    count_str = str(batch_counter).rjust(len(str(batch_total)))
    ratio_str = f"{count_str}/{batch_total}".rjust(9)
    progress_percent = f"({(batch_counter / batch_total) * 100:.1f}%)".rjust(8)
    batch_message = f"Batch progress: {ratio_str} {progress_percent}"

    # Clear the previous line and print over it
    print(batch_message.ljust(40), end="\r", flush=True)


def print_time_elapsed(start_seconds: float, stop_seconds: float) -> None:
    """Print the formatted time elapsed between `start` and `stop`."""

    print(f"Elapsed time: {timedelta(seconds=stop_seconds-start_seconds)}")
