from convection_seg.monitoring import build_reference, drift_alerts, log_prediction, read_log


def _rec(c):
    return {"coverage": c, "mean_brightness": 0.4, "mean_proba": c}


def test_drift_flags_outlier_only():
    ref = build_reference([_rec(0.10 + 0.01 * i) for i in range(10)])
    assert drift_alerts(_rec(0.14), ref) == {}
    assert "coverage" in drift_alerts(_rec(0.9), ref)


def test_log_roundtrip(tmp_path):
    cfg = {"monitoring": {"log_file": str(tmp_path / "p.jsonl"), "reference_file": str(tmp_path / "none.json"),
                          "drift_z_threshold": 3.0, "mlflow_enabled": False}}
    log_prediction(cfg, {"scan_time": "t0", **_rec(0.2), "max_proba": 0.9})
    assert read_log(cfg["monitoring"]["log_file"])[0]["scan_time"] == "t0"
