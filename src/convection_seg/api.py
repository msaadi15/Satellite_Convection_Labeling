"""FastAPI service: on-demand segmentation + access to the latest real-time product."""
from __future__ import annotations

import copy
import io
import tempfile
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from . import __version__
from .comparison import comparison_panel
from .config import load_config
from .model import Segmenter
from .monitoring import read_log
from .opencv_baseline import segment_hsv
from .preprocessing import overlay_mask
from .realtime import run_once

app = FastAPI(title="Morocco convection segmentation", version=__version__)


@lru_cache
def get_cfg() -> dict:
    return load_config()


@lru_cache
def get_segmenter() -> Segmenter:
    return Segmenter.from_config(get_cfg())


def _png(img: Image.Image) -> Response:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "model_found": Path(get_cfg()["model"]["path"]).exists()}


@app.post("/predict")
async def predict(file: UploadFile = File(...), output: str = "overlay"):
    """Segment an uploaded convection RGB image. output = overlay | mask | opencv | compare."""
    if output not in {"overlay", "mask", "opencv", "compare"}:
        raise HTTPException(400, "output must be overlay, mask, opencv or compare")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file as an image")
    mask, _ = get_segmenter().predict(image)
    cv_mask = segment_hsv(image) if output in {"opencv", "compare"} else None
    result = {
        "overlay": lambda: overlay_mask(image, mask),
        "mask": lambda: Image.fromarray(mask),
        "opencv": lambda: overlay_mask(image, cv_mask),
        "compare": lambda: comparison_panel(image, mask, cv_mask),
    }[output]()
    return _png(result)


@app.get("/latest")
def latest(kind: str = "overlay"):
    """Most recent real-time product (written by `convseg realtime`)."""
    if kind not in {"rgb", "mask", "overlay", "opencv_mask", "compare"}:
        raise HTTPException(400, "kind must be rgb, mask, overlay, opencv_mask or compare")
    path = Path(get_cfg()["realtime"]["output_dir"]) / f"latest_{kind}.png"
    if not path.exists():
        raise HTTPException(404, "No real-time product yet")
    return FileResponse(path, media_type="image/png")


@app.get("/stats")
def stats(n: int = 50):
    """Last n monitoring records (coverage, brightness, drift alerts)."""
    return read_log(get_cfg()["monitoring"]["log_file"])[-n:]


@app.post("/realtime")
def realtime(
    kind: str = "compare",
    x_eumetsat_key: str | None = Header(None),
    x_eumetsat_secret: str | None = Header(None),
):
    """Segment the newest SEVIRI scan using the CALLER's own EUMETSAT credentials.

    Send your key/secret from https://api.eumetsat.int/api-key as headers `X-EUMETSAT-Key` and
    `X-EUMETSAT-Secret`. They are used for this request only: never stored, logged, or shared, and the
    server's own credentials are never used. Results are written to a throw-away directory.
    """
    if kind not in {"rgb", "mask", "overlay", "opencv_mask", "compare"}:
        raise HTTPException(400, "kind must be rgb, mask, overlay, opencv_mask or compare")
    if not (x_eumetsat_key and x_eumetsat_secret):
        raise HTTPException(
            401, "Provide your own EUMETSAT credentials in the X-EUMETSAT-Key and X-EUMETSAT-Secret headers "
                 "(get them at https://api.eumetsat.int/api-key)")
    with tempfile.TemporaryDirectory(prefix="convseg_") as tmp:
        cfg = copy.deepcopy(get_cfg())
        cfg["realtime"]["output_dir"] = tmp
        cfg["realtime"]["keep_nat_files"] = False
        try:
            record = run_once(get_segmenter(), cfg, credentials=(x_eumetsat_key, x_eumetsat_secret))
        except PermissionError as exc:
            raise HTTPException(401, str(exc))
        if record is None:
            raise HTTPException(404, "No SEVIRI scan available in the lookback window")
        png = (Path(tmp) / f"latest_{kind}.png").read_bytes()
    return Response(png, media_type="image/png", headers={
        "X-Scan-Time": str(record["scan_time"]), "X-Coverage": f"{record['coverage']:.4f}"})
