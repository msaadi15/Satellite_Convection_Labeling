"""Offline evaluation against expert masks, with an optional quality gate for CI/CD."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .model import Segmenter


def binary_metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    """IoU / Dice / pixel accuracy for two 0/255 masks (empty-vs-empty counts as perfect)."""
    p, t = pred > 0, truth > 0
    inter = float((p & t).sum())
    union = float((p | t).sum())
    total = float(p.sum() + t.sum())
    return {
        "iou": inter / union if union else 1.0,
        "dice": 2 * inter / total if total else 1.0,
        "acc": float((p == t).mean()),
    }


def pair_files(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Match `prod_X.png` with `mask_prod_X.png`."""
    pairs = []
    for img in sorted(image_dir.glob("*.png")):
        m = mask_dir / f"mask_{img.name}"
        if m.exists():
            pairs.append((img, m))
    return pairs


def evaluate(cfg: dict, image_dir: str, mask_dir: str, limit: int | None = None,
             out_csv: str | None = None) -> dict:
    import pandas as pd

    seg = Segmenter.from_config(cfg)
    pairs = pair_files(Path(image_dir), Path(mask_dir))[:limit]
    if not pairs:
        raise SystemExit(f"No image/mask pairs found in {image_dir} / {mask_dir}")
    rows = []
    for img_path, mask_path in pairs:
        image = Image.open(img_path).convert("RGB")
        pred, _ = seg.predict(image)
        truth = np.asarray(Image.open(mask_path).convert("L").resize(image.size, Image.NEAREST))
        rows.append({"filename": img_path.stem, **binary_metrics(pred, truth)})
    df = pd.DataFrame(rows)
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    summary = {"n_images": len(df), **{f"mean_{c}": float(df[c].mean()) for c in ("iou", "dice", "acc")}}
    if cfg["monitoring"].get("mlflow_enabled"):
        import mlflow

        mlflow.set_experiment(cfg["monitoring"]["mlflow_experiment"])
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_params({"model_path": cfg["model"]["path"], "threshold": seg.threshold})
            mlflow.log_metrics({k: v for k, v in summary.items() if k != "n_images"})
    return summary


def quality_gate(summary: dict, min_dice: float) -> bool:
    return summary["mean_dice"] >= min_dice


def dump(summary: dict) -> str:
    return json.dumps(summary, indent=2)
