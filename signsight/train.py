"""
Model training and validation.
"""

import torch
from torch.utils.data import DataLoader

from signsight.const import BATCH_SIZE, EPOCH_COUNT, MODEL_PATH
from signsight.model import build_model, get_device, split_dataset


def train_model() -> None:
    """Run the full training loop and save weights to disk."""

    # Accomodate CUDA devices
    device = get_device()

    # Split dataset into training (80%) and validation (20%) subsets
    train_set, val_set = split_dataset()

    # Wraps data with proper batch size for training/validation and randomize
    # image order when grouping batches
    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, BATCH_SIZE)

    # NOTE: `val_` is an abbreviation of `validation_`

    # Load weights to device and replace the final layer for the 29 classes
    model = build_model(pretrained=True).to(device)

    # Loss function which measures how wrong the model's predictions are
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer which reduces losses by carefully adjusting weights
    # NOTE: learning rate (lr) controls how large each adjustment is
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Track the total amount of incorrect predictions across all epochs
    train_loss_total: float

    print(f"Beginning model training loop with {EPOCH_COUNT} epochs...")

    for epoch in range(EPOCH_COUNT):

        # "Training mode" enables dropout and batch normalization updates
        model.train()

        # Reset loss totals and batch count for this new epoch
        train_loss_total = 0.0

        # Get the respective images and labels together for each batch
        for batch, (images, labels) in enumerate(train_loader):

            # Send this batch's images and labels to the computing device
            images, labels = images.to(device), labels.to(device)

            # Clear gradients from previous batch
            optimizer.zero_grad()

            # Compute losses for this batch
            loss = criterion(model(images), labels)

            # Calculate gradients for this batch
            loss.backward()

            # Update the weights
            # NOTE: this is where the weights in the .pth file come from
            optimizer.step()

            # Accumulate total losses across all batches for display
            train_loss_total += loss.item()

            _print_batch_progress(batch + 1, len(train_loader))

        # Run model on the validation subset without updating the weights
        val_loss_total, val_accuracy = _validate(
            model, val_loader, criterion, device, len(val_set)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCH_COUNT} "
            f"| Train Loss: {train_loss_total / len(train_loader):.4f} "
            f"| Validation Loss: {val_loss_total:.4f} "
            f"| Validation Accuracy: {val_accuracy*100:.2f}%"
        )

    # Save weights to disk as a .pth file
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


def _validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
    dataset_size: int,
) -> tuple[float, float]:
    """Run validation loop and return average loss and accuracy."""

    # Track the total amount of losses and correct predictions
    val_loss_total: float = 0.0
    correct_predictions_count: int = 0

    # "Evaluation mode" disables dropout and freezes batch normalization
    # NOTE: this saves memory and speeds up the training pipeline
    model.eval()

    # Disable gradient tracking since weights are not being updated here
    with torch.no_grad():

        # Same control flow for images and lables as in train_model()
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss_total += criterion(outputs, labels).item()

            # Get the index of the highest confidence prediction for each image
            predictions = outputs.argmax(dim=1)

            # Compare predictions to true labels, producing a boolean tensor
            correct_mask = predictions == labels.to(device)

            # Count the number of truthy values and add it to the total count
            correct_predictions_count += correct_mask.sum().item()

    # Average out all the losses
    val_loss_average: float = val_loss_total / len(val_loader)

    # Get the ratio of accurate predictions
    val_accuracy: float = correct_predictions_count / dataset_size

    return val_loss_average, val_accuracy


def _print_batch_progress(batch_counter: int, batch_total: int) -> None:
    """Print epoch batch training progress."""

    # Zero padding in numerator that aligns with the denominator
    batch_counter_str = str(batch_counter).zfill(len(str(batch_total)))
    batch_message = f"Batch progress: {batch_counter_str}/{batch_total}"

    # Clear the previous line and print over it
    print(batch_message.ljust(40), end="\r", flush=True)
