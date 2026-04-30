"""
Model performance and accuracy testing.
"""

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix as sklearn_cm
from torch.utils.data import DataLoader
from torchvision import datasets

from signsight.const import BATCH_SIZE, DATASET_PATH, FIGURE_PATH, MODEL_PATH
from signsight.model import get_device, load_model
from signsight.preprocess import get_transform
from signsight.train import _print_batch_progress


def evaluate_model() -> None:
    """Load saved model and print accuracy and confusion matrix."""

    # Accomodate CUDA devices
    device = get_device()

    # Load the entire dataset (unlike for training, this doesn't split it)
    dataset_full = datasets.ImageFolder(DATASET_PATH, transform=get_transform())

    # Wrap data and set the size of each batch
    loader_eval = DataLoader(dataset_full, batch_size=BATCH_SIZE)

    # Load trained weights from disk
    model_trained = load_model(MODEL_PATH, device)

    # Get the predictions
    predictions, labels = _collect_predictions(
        model_trained, loader_eval, device
    )

    # Creat a boolean tensor of predictions onto their correct labels
    correct_mask = predictions == labels

    # Get the sum of all correct predictions
    correct_predictions_count = correct_mask.sum().item()

    # Get the ratio of correct predictions
    eval_accuracy: float = correct_predictions_count / len(labels)

    print(f"Evaluation accuracy: {eval_accuracy*100:.2f}%")

    _plot_confusion_matrix(predictions, labels, dataset_full.classes)


def _collect_predictions(
    model_trained: torch.nn.Module,
    loader_eval: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run model over dataset and return all predictions and labels."""

    predictions_superset = []
    labels_superset = []
    loader_eval_size = len(loader_eval)

    print("Collecting predictions of trained model...")

    # Disable gradient tracking since weights are not being updated here
    with torch.no_grad():
        for batch, (images, labels) in enumerate(loader_eval):
            images = images.to(device)

            # Get confidence scores for each class
            outputs = model_trained(images)

            # Pick the class index with the highest confidence
            predictions = outputs.argmax(dim=1).cpu()

            predictions_superset.append(predictions)
            labels_superset.append(labels)

            _print_batch_progress(batch + 1, loader_eval_size)

    # Concatenate predictions and labels from all batches into single tensors
    return torch.cat(predictions_superset), torch.cat(labels_superset)


def _plot_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    dataset_class_names: list[str],
) -> None:
    """Plot and display a confusion matrix using matplotlib."""

    # Convert tensors to numpy arrays for scikit-learn
    confusion_matrix = sklearn_cm(labels.numpy(), predictions.numpy())

    _, axes = plt.subplots(figsize=(18, 16))

    # Create a visualization the confusion matrix
    confusion_matrix_display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=dataset_class_names
    )

    # Plot the display figure
    confusion_matrix_display.plot(
        ax=axes,
        xticks_rotation=45,  # Rotate x-axis labels to prevent overlap
        colorbar=True,
        values_format="",  # Hides cell numbers
    )

    axes.set_title("SignSight Confusion Matrix", fontsize=18, pad=20)
    axes.tick_params(axis="both", labelsize=11)
    axes.set_xlabel("Predicted label", fontsize=13)
    axes.set_ylabel("True label", fontsize=13)
    plt.tight_layout()

    # Save the confusion matrix to disk
    plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print("Confusion matrix figure saved to: confusion_matrix.png")

    # Attempt to show interactively if a display is available
    try:
        import matplotlib

        if matplotlib.get_backend() != "agg":
            plt.show()

    except UserWarning:
        print("warning: cannot display figure, open the image file instead")
