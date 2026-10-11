"""Real-time loop: fetch newest SEVIRI scan -> RGB -> segment -> save + monitor."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from PIL import Image

from .comparison import compare_stats, comparison_panel
from .model import Segmenter
from .monitoring import image_stats, log_prediction
from .opencv_baseline import segment_hsv
from .preprocessing import overlay_mask
from .satellite import fetch_latest_nat, nat_to_rgb, scan_time_from_name

log = logging.getLogger(__name__)


def segment_and_save(segmenter: Segmenter, image: Image.Image, scan_time: str, cfg: dict) -> dict:
    """Run the model on one image, write rgb/mask/overlay + `latest_*` copies, log stats."""
    out = Path(cfg["realtime"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    mask, proba = segmenter.predict(image)
    overlay = overlay_mask(image, mask)
    cv_mask = segment_hsv(image)
    panel = comparison_panel(image, mask, cv_mask)
    products = (("rgb", image), ("mask", Image.fromarray(mask)), ("overlay", overlay),
                ("opencv_mask", Image.fromarray(cv_mask)), ("compare", panel))
    for name, img in products:
        img.save(out / f"{scan_time}_{name}.png")
        img.save(out / f"latest_{name}.png")
    cmp = compare_stats(mask, cv_mask)
    record = {"scan_time": scan_time, **image_stats(image, mask, proba),
              "opencv_coverage": cmp["opencv_coverage"], "iou_unet_vs_opencv": cmp["iou_unet_vs_opencv"]}
    return log_prediction(cfg, record)


def run_once(segmenter: Segmenter, cfg: dict) -> dict | None:
    workdir = Path(cfg["realtime"]["output_dir"]) / "nat"
    nat = fetch_latest_nat(cfg, workdir)
    if nat is None:
        return None
    try:
        image = nat_to_rgb(nat, cfg["region"])
        return segment_and_save(segmenter, image, scan_time_from_name(nat.name), cfg)
    finally:
        if not cfg["realtime"]["keep_nat_files"]:
            nat.unlink(missing_ok=True)


def run_loop(cfg: dict, once: bool = False) -> None:
    segmenter = Segmenter.from_config(cfg)
    last = None
    while True:
        try:
            rec = run_once(segmenter, cfg)
            if rec and rec["scan_time"] != last:
                last = rec["scan_time"]
                log.info("scan %s coverage=%.3f", rec["scan_time"], rec["coverage"])
        except Exception:  # keep the service alive across transient network/data errors
            log.exception("Real-time cycle failed")
        if once:
            return
        time.sleep(cfg["realtime"]["interval_seconds"])
