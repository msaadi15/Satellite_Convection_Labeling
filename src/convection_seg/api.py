"""FastAPI service: on-demand segmentation + access to the latest real-time product."""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image

from . import __version__
from .config import load_config
from .model import Segmenter
from .monitoring import read_log
from .preprocessing import overlay_mask

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
    """Segment an uploaded convection RGB image. output = overlay | mask."""
    if output not in {"overlay", "mask"}:
        raise HTTPException(400, "output must be 'overlay' or 'mask'")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file as an image")
    mask, _ = get_segmenter().predict(image)
    return _png(overlay_mask(image, mask) if output == "overlay" else Image.fromarray(mask))


@app.get("/latest")
def latest(kind: str = "overlay"):
    """Most recent real-time product (written by `convseg realtime`)."""
    if kind not in {"rgb", "mask", "overlay"}:
        raise HTTPException(400, "kind must be rgb, mask or overlay")
    path = Path(get_cfg()["realtime"]["output_dir"]) / f"latest_{kind}.png"
    if not path.exists():
        raise HTTPException(404, "No real-time product yet")
    return FileResponse(path, media_type="image/png")


@app.get("/stats")
def stats(n: int = 50):
    """Last n monitoring records (coverage, brightness, drift alerts)."""
    return read_log(get_cfg()["monitoring"]["log_file"])[-n:]
