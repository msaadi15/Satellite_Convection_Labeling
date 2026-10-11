"""Pure-numpy/PIL image helpers (no TensorFlow dependency, easy to test)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def preprocess_image(image: Image.Image, size=(160, 160)) -> np.ndarray:
    """RGB image -> float32 array in [0, 1] of shape (H, W, 3), as used in training."""
    h, w = size
    img = image.convert("RGB").resize((w, h), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def postprocess_mask(proba: np.ndarray, out_size: tuple[int, int], threshold=0.5) -> np.ndarray:
    """Probability map -> 0/255 uint8 mask resized (nearest) to out_size=(W, H)."""
    binary = (proba > threshold).astype(np.uint8) * 255
    return np.asarray(Image.fromarray(binary).resize(out_size, Image.NEAREST))


def overlay_mask(image: Image.Image, mask: np.ndarray, color=(255, 255, 0), alpha=0.45) -> Image.Image:
    """Draw the convection mask over the image as a translucent colour layer."""
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    on = (mask > 0)[..., None]
    tint = np.array(color, dtype=np.float32)
    out = np.where(on, (1 - alpha) * base + alpha * tint, base)
    return Image.fromarray(out.astype(np.uint8))


def coverage(mask: np.ndarray) -> float:
    """Fraction of pixels flagged as convection."""
    return float((mask > 0).mean())
