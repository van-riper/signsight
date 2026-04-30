"""
Basic model definitions using ResNet18 architecture.
"""

import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets, models

from signsight.const import CLASS_COUNT, DATASET_PATH, VAL_SPLIT
from signsight.preprocess import get_transform


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

    transform = get_transform()

    dataset = datasets.ImageFolder(DATASET_PATH, transform)

    val_size = int(VAL_SPLIT * len(dataset))

    return random_split(dataset, [len(dataset) - val_size, val_size])


def get_device() -> torch.device:
    """Detect CUDA device if available, otherwise use CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
