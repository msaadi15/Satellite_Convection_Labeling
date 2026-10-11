import numpy as np
from PIL import Image

from convection_seg.preprocessing import coverage, overlay_mask, postprocess_mask, preprocess_image


def test_preprocess_shape_and_range():
    x = preprocess_image(Image.new("RGB", (600, 521), (255, 0, 128)), (512, 512))
    assert x.shape == (512, 512, 3) and x.dtype == np.float32
    assert x.min() >= 0 and x.max() <= 1


def test_postprocess_resizes_and_thresholds():
    proba = np.zeros((512, 512), dtype=np.float32)
    proba[:256] = 0.9
    mask = postprocess_mask(proba, (600, 521))
    assert mask.shape == (521, 600)
    assert set(np.unique(mask)) == {0, 255}
    assert 0.45 < coverage(mask) < 0.55


def test_overlay_only_changes_masked_pixels():
    img = Image.new("RGB", (10, 10), (0, 0, 0))
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5] = 255
    out = np.asarray(overlay_mask(img, mask))
    assert out[9, 0].tolist() == [0, 0, 0]
    assert out[0, 0].tolist() != [0, 0, 0]
