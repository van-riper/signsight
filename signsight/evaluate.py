"""
Model performance and accuracy testing.
"""

from torch.utils.data import DataLoader
from torchvision import datasets

from signsight.const import BATCH_SIZE, DATASET_PATH, MODEL_PATH
from signsight.model import get_device, load_model
from signsight.preprocess import get_transform


def evaluate_model() -> None:
    """Load saved model and print accuracy and confusion matrix."""

    device = get_device()
    model = load_model(MODEL_PATH, device)

    dataset = datasets.ImageFolder(DATASET_PATH, transform=get_transform())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    class_names = dataset.classes
