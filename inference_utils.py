"""Image preparation and binary-probability interpretation helpers."""

from __future__ import annotations

import numpy as np
from PIL import Image


def prepare_image(image: Image.Image, image_size: tuple[int, int]) -> np.ndarray:
    """Convert an uploaded image into the RGB batch expected by the model."""
    image = image.convert("RGB")
    image = image.resize(image_size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def interpret_probability(
    rotten_probability: float,
    class_names: list[str],
    threshold: float,
) -> tuple[str, float]:
    """Return the selected label and its binary-class confidence score."""
    rotten_probability = float(np.clip(rotten_probability, 0.0, 1.0))
    if rotten_probability >= threshold:
        return class_names[1], rotten_probability
    return class_names[0], 1.0 - rotten_probability


def display_label(label: str) -> str:
    """Convert a stored class name into a user-facing label."""
    return label.replace("_", " ").title()
