"""Model training and validation."""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from ..const import (
    BATCH_SIZE,
    DATASET_TEST_PATH,
    DATASET_TRAIN_PATH,
    EPOCH_COUNT,
    MODEL_PATH,
)
from .utils import (
    build_model,
    get_device,
    get_transform,
    print_batch_progress,
)

# from .utils import split_dataset


def train_model() -> None:
    """Run the full training loop and save weights to disk."""

    # Accomodate CUDA devices
    device = get_device()

    dataset_train = ImageFolder(
        DATASET_TRAIN_PATH, transform=get_transform(training=True)
    )
    dataset_val = ImageFolder(
        DATASET_TEST_PATH, transform=get_transform(training=False)
    )

    # dataset_train, dataset_val = split_dataset()

    # Wraps data with proper batch size for training/validation
    # and randomize image order when grouping batches
    dataloader_train = DataLoader(dataset_train, BATCH_SIZE, shuffle=True)
    dataloader_val = DataLoader(dataset_val, BATCH_SIZE)

    # NOTE: `_val` is an abbreviation of `_validation`

    # Load weights to device and replace the final layer for the 29 classes
    model = build_model(pretrained=True).to(device)

    # Loss function which measures how wrong the model's predictions are
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer which reduces losses by carefully adjusting weights
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # NOTE: learning rate (lr) controls how large each adjustment is

    # Track the total amount of incorrect predictions across all epochs
    loss_total_train: float

    print(f"Beginning model training loop with {EPOCH_COUNT} epochs...")

    for epoch in range(EPOCH_COUNT):

        # "Training mode" enables dropout and batch normalization updates
        model.train()

        # Reset loss totals and batch count for this new epoch
        loss_total_train = 0.0

        # Get the respective images and labels together for each batch
        for batch, (images, labels) in enumerate(dataloader_train):

            # Send this batch's images and labels to the computing device
            images, labels = images.to(device), labels.to(device)

            # Clear gradients from previous batch
            optimizer.zero_grad()

            # Compute losses for this batch
            loss = criterion(model(images), labels)

            # Calculate gradients for this batch
            loss.backward()

            # Update the weights
            optimizer.step()

            # NOTE: this is where the weights in the .pth file come from

            # Accumulate total losses across all batches for display
            loss_total_train += loss.item()

            print_batch_progress(batch + 1, len(dataloader_train))

        # Run model on the validation subset without updating the weights
        loss_total_val, accuracy_val = _validate(
            model, dataloader_val, criterion, device, len(dataset_val)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCH_COUNT} "
            f"| Train Loss: {loss_total_train / len(dataloader_train):.4f} "
            f"| Validation Loss: {loss_total_val:.4f} "
            f"| Validation Accuracy: {accuracy_val*100:.2f}%"
        )

    # Save weights to disk as a .pth file
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


def _validate(
    model: torch.nn.Module,
    dataloader_val: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
    dataset_val_size: int,
) -> tuple[float, float]:
    """Run validation loop and return average loss and accuracy."""

    # Track the total amount of losses and correct predictions
    loss_total_val: float = 0.0
    correct_predictions_count: int = 0

    # "Evaluation mode" disables dropout and freezes batch normalization
    model.eval()

    # NOTE: this saves memory and speeds up the training pipeline

    # Disable gradient tracking since weights are not being updated here
    with torch.no_grad():

        # Same control flow for images and lables as in train_model()
        for images, labels in dataloader_val:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_total_val += criterion(outputs, labels).item()

            # Get the index of the highest confidence prediction for each image
            predictions = outputs.argmax(dim=1)

            # Compare predictions to true labels, producing a boolean tensor
            correct_mask = predictions == labels.to(device)

            # Count the number of truthy values and add it to the total count
            correct_predictions_count += correct_mask.sum().item()

    # Average out all the losses
    val_loss_average: float = loss_total_val / len(dataloader_val)

    # Get the ratio of accurate predictions
    val_accuracy: float = correct_predictions_count / dataset_val_size

    return val_loss_average, val_accuracy
