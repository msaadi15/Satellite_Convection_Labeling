"""EUMETSAT access and SEVIRI .nat -> convection RGB conversion."""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

CHANNELS = ["IR_108", "IR_039", "IR_016", "VIS006", "WV_062", "WV_073"]
CREDENTIALS_FILE = Path.home() / ".eumetsat_api_key"


def load_credentials() -> tuple[str, str]:
    """Read EUMETSAT key/secret from env (EUMETSAT_KEY/SECRET) or ~/.eumetsat_api_key."""
    key, secret = os.environ.get("EUMETSAT_KEY"), os.environ.get("EUMETSAT_SECRET")
    if key and secret:
        return key, secret
    if CREDENTIALS_FILE.exists():
        creds = json.loads(CREDENTIALS_FILE.read_text())
        return creds["consumer_key"], creds["consumer_secret"]
    raise RuntimeError(
        "No EUMETSAT credentials: set EUMETSAT_KEY and EUMETSAT_SECRET or create "
        f"{CREDENTIALS_FILE}."
    )


def fetch_latest_nat(cfg: dict, workdir: Path, now: dt.datetime | None = None,
                     credentials: tuple[str, str] | None = None) -> Path | None:
    """Download the most recent SEVIRI scan in the lookback window; return the .nat path.

    `credentials` = (key, secret). If None, falls back to env/file (CLI use only; the API always
    passes the caller's own credentials). Raises PermissionError if EUMETSAT rejects them.
    """
    import eumdac

    now = now or dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    token = eumdac.AccessToken(credentials or load_credentials())
    try:
        str(token)  # forces the token request so bad credentials fail here
    except Exception as exc:
        raise PermissionError("EUMETSAT rejected the supplied key/secret") from exc
    collection = eumdac.DataStore(token).get_collection(cfg["realtime"]["collection"])
    start = now - dt.timedelta(minutes=cfg["realtime"]["lookback_minutes"])
    products = list(collection.search(dtstart=start, dtend=now, sort="start,time,1"))
    if not products:
        log.warning("No SEVIRI product between %s and %s", start, now)
        return None
    product = products[0]  # newest first
    workdir.mkdir(parents=True, exist_ok=True)
    with product.open() as src:
        target = workdir / src.name
        with open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
    if target.suffix == ".zip":
        with zipfile.ZipFile(target) as z:
            nats = [n for n in z.namelist() if n.endswith(".nat")]
            z.extractall(workdir)
        target.unlink()
        return workdir / nats[0]
    return target


def nat_to_rgb(nat_path: str | Path, region: dict) -> Image.Image:
    """Build the EUMETSAT 'convection' RGB, cropped to the region, as a PIL image.

    Same composite used to create the training images
    (R = WV6.2-WV7.3, G = IR3.9-IR10.8, B = NIR1.6-VIS0.6).
    """
    from satpy import Scene

    scn = Scene(filenames=[str(nat_path)], reader="seviri_l1b_native")
    scn.load(["convection"])
    bbox = (region["lon_min"], region["lat_min"], region["lon_max"], region["lat_max"])
    cropped = scn.crop(ll_bbox=bbox)
    rgb = cropped["convection"].transpose("y", "x", "bands").values
    # SEVIRI is seen upside-down in native scan order; satpy 'convection' is 0-1 float.
    arr = np.nan_to_num(np.clip(rgb, 0, 1)) * 255
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def scan_time_from_name(nat_name: str) -> str:
    """'MSG4-SEVI-MSG15-0100-NA-20230101120010.2…Z-NA.nat' -> '2023-01-01T12-00-10'."""
    for part in Path(nat_name).stem.split("-"):
        if part.startswith("20") and len(part) >= 14 and part[:14].isdigit():
            return f"{part[:4]}-{part[4:6]}-{part[6:8]}T{part[8:10]}-{part[10:12]}-{part[12:14]}"
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
