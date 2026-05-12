"""Functions for model building, image transforms, and batch progress."""

from datetime import timedelta
from os import listdir
from pathlib import Path

import torch
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from ..const import CLASS_COUNT, EXCLUDED_CLASSES, IMAGE_SIZE


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


# TODO: restore split_dataset for the raw dataset pending model performance


def get_class_names(dataset_path: Path) -> list[str]:
    """Return sorted class names excluding digit classes."""

    return [
        cls
        for cls in sorted(listdir(dataset_path))
        if cls not in EXCLUDED_CLASSES and (dataset_path / cls).is_dir()
    ]


def load_dataset(path: Path, transform: transforms.Compose) -> ImageFolder:
    """Load dataset excluding digit classes."""

    dataset = ImageFolder(str(path), transform=transform)

    # Filter out digit classes
    filtered_classes = [
        cls for cls in dataset.classes if cls not in EXCLUDED_CLASSES
    ]

    # Remap class indices
    dataset.classes = filtered_classes
    dataset.class_to_idx = {
        cls: idx for idx, cls in enumerate(filtered_classes)
    }
    dataset.samples = [
        (path, dataset.class_to_idx[dataset.classes[label]])
        for path, label in dataset.samples
        if dataset.classes[label] not in EXCLUDED_CLASSES
    ]
    dataset.targets = [label for _, label in dataset.samples]

    return dataset


# TODO: support multiple model architectures: resnet, mobilenet, efficientnet
def build_model(pretrained: bool) -> torch.nn.Module:
    """Build the model weights."""

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)

    model.fc = torch.nn.Linear(model.fc.in_features, CLASS_COUNT)

    return model


# TODO: set path argument as a Path object
def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    """Load saved model weights from disk."""

    model = build_model(pretrained=False)

    # Ensure GPU-trained weights load correctly on CPU with map_location
    model.load_state_dict(torch.load(path, map_location=device))

    # Move the model to the right device
    model.to(device)

    model.eval()

    return model


# TODO: also print total progress across all epochs
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

    elapsed_seconds = round(stop_seconds - start_seconds)
    print(f"Elapsed time: {timedelta(seconds=elapsed_seconds)}")
