import numpy as np
from PIL import Image

from convection_seg.comparison import agreement_map, compare_stats, comparison_panel
from convection_seg.opencv_baseline import segment_hsv


def test_hsv_detects_red_yellow_not_blue():
    img = np.zeros((3, 30, 3), np.uint8)
    img[:, :10] = (255, 0, 0)      # red
    img[:, 10:20] = (255, 255, 0)  # yellow
    img[:, 20:] = (60, 0, 255)     # blue/purple background
    m = segment_hsv(Image.fromarray(img))
    assert (m[:, :20] == 255).all() and (m[:, 20:] == 0).all()


def test_compare_stats_and_panel():
    img = Image.new("RGB", (20, 10))
    a = np.zeros((10, 20), np.uint8)
    a[:, :10] = 255
    b = np.zeros_like(a)
    b[:, 5:15] = 255
    s = compare_stats(a, b, a)
    assert abs(s["iou_unet_vs_opencv"] - 5 / 15) < 1e-9 and s["dice_unet_vs_expert"] == 1.0
    assert comparison_panel(img, a, b, a).size == (100, 32)
    assert agreement_map(img, a, b).size == img.size
