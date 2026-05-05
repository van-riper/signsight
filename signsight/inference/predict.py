"""Model inference for the SignSight pipeline."""

import cv2
import torch
from PIL import Image

from ..const import MODEL_PATH
from ..core import get_device, get_transform, load_model


def load_predictor() -> tuple[torch.nn.Module, torch.device]:
    """Load the trained model and return it with its device."""

    device = get_device()
    model = load_model(MODEL_PATH, device)

    return model, device


def predict(
    model: torch.nn.Module,
    device: torch.device,
    roi: cv2.typing.MatLike,
    class_names: list[str],
) -> tuple[str, float]:
    """Run inference on a cropped hand ROI and return the predicted letter.

    Args:
        model: Trained SignSight model.
        device: Device the model is loaded on.
        roi: Cropped hand region of interest from detect.py.
        class_names: Ordered list of class names from the dataset.

    Returns:
        A tuple of the predicted class name and its confidence score.
    """

    transform = get_transform(training=False)

    # Convert BGR numpy array to RGB PIL image for torchvision transforms
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)

    # Preprocess and add batch dimension
    tensor: torch.Tensor = transform(pil_image)  # type: ignore[assignment]
    tensor = tensor.clone().detach().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)

        # Convert raw scores to probabilities
        probabilities = torch.softmax(outputs, dim=1)

        # Get the highest confidence prediction
        confidence, predicted_index = probabilities.max(dim=1)

    predicted_class = class_names[int(predicted_index.item())]
    confidence_score = confidence.item() * 100

    return predicted_class, confidence_score
