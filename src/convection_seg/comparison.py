"""Side-by-side comparison of U-Net vs OpenCV (and optionally expert) segmentations."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .evaluate import binary_metrics
from .preprocessing import overlay_mask

AGREE, ONLY_UNET, ONLY_CV = (0, 200, 0), (230, 40, 40), (40, 90, 240)


def agreement_map(image: Image.Image, unet: np.ndarray, cv: np.ndarray) -> Image.Image:
    """Green = both methods, red = U-Net only, blue = OpenCV only."""
    out = np.asarray(image.convert("RGB"), dtype=np.float32) * 0.5
    u, c = unet > 0, cv > 0
    for sel, col in ((u & c, AGREE), (u & ~c, ONLY_UNET), (~u & c, ONLY_CV)):
        out[sel] = col
    return Image.fromarray(out.astype(np.uint8))


def comparison_panel(image: Image.Image, unet: np.ndarray, cv: np.ndarray,
                     truth: np.ndarray | None = None) -> Image.Image:
    """Image | U-Net overlay | OpenCV overlay | agreement [| expert overlay], with titles."""
    tiles = [("RGB", image.convert("RGB")),
             ("U-Net", overlay_mask(image, unet)),
             ("OpenCV (HSV)", overlay_mask(image, cv)),
             ("Agreement: green both / red U-Net / blue OpenCV", agreement_map(image, unet, cv))]
    if truth is not None:
        tiles.insert(1, ("Expert", overlay_mask(image, truth, color=(0, 255, 255))))
    w, h = image.size
    bar = 22
    panel = Image.new("RGB", (w * len(tiles), h + bar), "white")
    draw = ImageDraw.Draw(panel)
    for i, (title, tile) in enumerate(tiles):
        panel.paste(tile, (i * w, bar))
        draw.text((i * w + 4, 5), title[:int(w / 6)], fill="black")
    return panel


def compare_stats(unet: np.ndarray, cv: np.ndarray, truth: np.ndarray | None = None) -> dict:
    """Agreement between methods, plus each method's score against the expert if provided."""
    m = binary_metrics(unet, cv)
    stats = {"unet_coverage": float((unet > 0).mean()), "opencv_coverage": float((cv > 0).mean()),
             "iou_unet_vs_opencv": m["iou"], "dice_unet_vs_opencv": m["dice"]}
    if truth is not None:
        for name, mask in (("unet", unet), ("opencv", cv)):
            for k, v in binary_metrics(mask, truth).items():
                stats[f"{k}_{name}_vs_expert"] = v
    return stats
