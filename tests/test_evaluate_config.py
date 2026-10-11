import numpy as np

from convection_seg.config import load_config
from convection_seg.evaluate import binary_metrics, pair_files


def test_metrics_perfect_and_disjoint():
    a = np.zeros((4, 4), np.uint8)
    a[:2] = 255
    assert binary_metrics(a, a)["dice"] == 1.0
    b = np.zeros_like(a)
    b[2:] = 255
    assert binary_metrics(a, b)["iou"] == 0.0


def test_empty_masks_are_perfect():
    z = np.zeros((4, 4), np.uint8)
    assert binary_metrics(z, z)["iou"] == 1.0


def test_pair_files(tmp_path):
    (tmp_path / "i").mkdir()
    (tmp_path / "m").mkdir()
    for n in ("prod_a.png", "prod_b.png"):
        (tmp_path / "i" / n).touch()
    (tmp_path / "m" / "mask_prod_a.png").touch()
    assert len(pair_files(tmp_path / "i", tmp_path / "m")) == 1


def test_env_override(monkeypatch):
    monkeypatch.setenv("CONVSEG__MODEL__THRESHOLD", "0.7")
    assert load_config()["model"]["threshold"] == 0.7
