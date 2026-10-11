"""Classical baseline: HSV colour thresholds on the convection RGB (yellow, orange, red).

Same thresholds as the original `steps/opencv_2024.py` script. No training involved, so it is the
reference the U-Net is compared against.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# (lower, upper) in OpenCV HSV (H in 0-180). Red wraps around 0/180.
HSV_RANGES = {
    "yellow": ((20, 100, 100), (30, 255, 255)),   # thick ice cloud, small crystals
    "orange": ((10, 100, 100), (20, 255, 255)),   # intermediate clouds
    "red_low": ((0, 100, 100), (10, 255, 255)),   # thick ice cloud, large crystals
    "red_high": ((160, 100, 100), (180, 255, 255)),
}


def segment_hsv(image: Image.Image) -> np.ndarray:
    """RGB PIL image -> uint8 0/255 convection mask."""
    import cv2

    hsv = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES.values():
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    return mask


def process_folder(input_folder: str, output_folder: str) -> int:
    """Write `mask_<name>` for every image in input_folder; returns the number processed."""
    src = Path(input_folder)
    if not src.is_dir():
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist")
    out = Path(output_folder)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            Image.fromarray(segment_hsv(Image.open(p))).save(out / f"mask_{p.name}")
            n += 1
    return n
