"""Functions for model building, image transforms, and batch progress."""

import torch
from torch.utils.data import Subset, random_split
from torchvision import datasets, models, transforms

from ..const import CLASS_COUNT, DATASET_PATH, IMAGE_SIZE, VAL_SPLIT


def get_device() -> torch.device:
    """Detect CUDA device if available, otherwise use CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform() -> transforms.Compose:
    """Build the image preprocessing pipeline for training and inference."""

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


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


def print_batch_progress(batch_counter: int, batch_total: int) -> None:
    """Print batch training/evaluation progress."""

    # Zero padding in numerator that aligns with the denominator
    batch_counter_str = str(batch_counter).zfill(len(str(batch_total)))
    batch_message = f"Batch progress: {batch_counter_str}/{batch_total}"

    # Clear the previous line and print over it
    print(batch_message.ljust(40), end="\r", flush=True)
