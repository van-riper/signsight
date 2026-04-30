"""
Model training and validation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from signsight.const import (
    BATCH_SIZE,
    DATASET_PATH,
    EPOCH_COUNT,
    MODEL_PATH,
    VAL_SPLIT,
)
from signsight.model import build_model, get_device
from signsight.preprocess import get_transform


def train_model() -> None:
    """Run the full training loop and save weights to disk."""

    device = get_device()
    transform = get_transform()

    # Load dataset and split into train/val
    dataset = datasets.ImageFolder(DATASET_PATH, transform)
    val_size = int(VAL_SPLIT * len(dataset))
    train_set, val_set = random_split(
        dataset, [len(dataset) - val_size, val_size]
    )

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)

    model = build_model(pretrained=True).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_loss: float

    for epoch in range(EPOCH_COUNT):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            loss = criterion(model(images), labels)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        val_loss, val_accuracy = _validate(
            model, val_loader, criterion, device, len(val_set)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCH_COUNT} "
            f"| Train Loss: {train_loss / len(train_loader):.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val Accuracy: {val_accuracy:.2f}%"
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def _validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
    dataset_size: int,
) -> tuple[float, float]:
    """Run validation loop and return average loss and accuracy."""

    loss_total: float = 0.0
    correct_count: int = 0

    model.eval()

    # Disable gradient calculation since this is just calculating the losses
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss_total += criterion(outputs, labels).item()
            correct_count += (outputs.argmax(1) == labels).sum().item()

    loss_average: float = loss_total / len(loader)
    accuracy: float = 100 * correct_count / dataset_size

    return loss_average, accuracy
