import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from convection_seg import api
from convection_seg.model import Segmenter


class FakeModel:
    def predict(self, x, verbose=0):
        out = np.zeros((*x.shape[:3], 1), np.float32)
        out[:, :80] = 0.9
        return out


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(api, "get_segmenter", lambda: Segmenter(FakeModel()))
    buf = io.BytesIO()
    Image.new("RGB", (60, 52), (10, 20, 30)).save(buf, format="PNG")
    r = TestClient(api.app).post("/predict?output=mask", files={"file": ("a.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    mask = np.asarray(Image.open(io.BytesIO(r.content)))
    assert mask.shape == (52, 60) and set(np.unique(mask)) == {0, 255}


def test_bad_upload_is_400(monkeypatch):
    monkeypatch.setattr(api, "get_segmenter", lambda: Segmenter(FakeModel()))
    r = TestClient(api.app).post("/predict", files={"file": ("a.png", b"nope", "image/png")})
    assert r.status_code == 400


def test_compare_output_is_panel(monkeypatch):
    monkeypatch.setattr(api, "get_segmenter", lambda: Segmenter(FakeModel()))
    buf = io.BytesIO()
    Image.new("RGB", (60, 52), (255, 0, 0)).save(buf, format="PNG")
    r = TestClient(api.app).post("/predict?output=compare", files={"file": ("a.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size[0] == 240
