"""Prediction monitoring: JSONL log, drift checks vs a reference, optional MLflow."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)
DRIFT_FEATURES = ("coverage", "mean_brightness", "mean_proba")


def image_stats(image: Image.Image, mask: np.ndarray, proba: np.ndarray) -> dict:
    """Cheap per-prediction features used for monitoring."""
    return {
        "coverage": float((mask > 0).mean()),
        "mean_brightness": float(np.asarray(image.convert("L"), dtype=np.float32).mean() / 255),
        "mean_proba": float(proba.mean()),
        "max_proba": float(proba.max()),
    }


def build_reference(records: list[dict]) -> dict:
    """Mean/std per drift feature from a list of historical stat records."""
    ref = {}
    for f in DRIFT_FEATURES:
        vals = np.array([r[f] for r in records], dtype=float)
        ref[f] = {"mean": float(vals.mean()), "std": float(vals.std() or 1e-6)}
    return ref


def drift_alerts(stats: dict, reference: dict, z_threshold: float = 3.0) -> dict[str, float]:
    """Features whose z-score against the reference exceeds the threshold."""
    alerts = {}
    for f in DRIFT_FEATURES:
        if f in reference and f in stats:
            z = (stats[f] - reference[f]["mean"]) / max(reference[f]["std"], 1e-6)
            if abs(z) > z_threshold:
                alerts[f] = round(float(z), 2)
    return alerts


def load_reference(path: str | Path) -> dict | None:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def log_prediction(cfg: dict, record: dict) -> dict:
    """Append the record to the JSONL log, run drift checks, optionally log to MLflow."""
    mon = cfg["monitoring"]
    ref = load_reference(mon["reference_file"])
    if ref:
        record["drift_alerts"] = drift_alerts(record, ref, mon["drift_z_threshold"])
        if record["drift_alerts"]:
            log.warning("Drift alert for %s: %s", record.get("scan_time"), record["drift_alerts"])
    path = Path(mon["log_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    if mon.get("mlflow_enabled"):
        import mlflow

        mlflow.set_experiment(mon["mlflow_experiment"])
        with mlflow.start_run(run_name=str(record.get("scan_time"))):
            mlflow.log_metrics({k: record[k] for k in (*DRIFT_FEATURES, "max_proba")})
    return record


def read_log(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
