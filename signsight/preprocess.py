"""
Transforms pipeline for preprocessing input images before model inference.
"""

from torchvision import transforms

IMAGE_SIZE: int = 64


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
