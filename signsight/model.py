"""
Basic model definitions using ResNet18 architecture.
"""

import torch
from torch import nn
from torchvision import models

# 26 letters plus 3 control signs
NUM_CLASSES: int = 29


def build_model(pretrained: bool = True) -> nn.Module:
    """Build the model weights."""

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None

    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    return model


def load_model(path: str, device: torch.device) -> nn.Module:
    """Load saved model weights from disk."""

    model = build_model(pretrained=False)

    # map_location ensures GPU-trained weights load correctly on CPU
    model.load_state_dict(torch.load(f=path, map_location=device))

    model.eval()
    return model


def get_device() -> torch.device:
    """Detect CUDA device if available, otherwise use CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
