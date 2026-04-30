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

        print(
            f"Epoch {epoch + 1}/{EPOCH_COUNT} "
            f"| Train Loss: {train_loss / len(train_loader):.4f} "
        )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
