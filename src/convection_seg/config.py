"""YAML configuration with environment-variable overrides."""
from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
ENV_PREFIX = "CONVSEG__"


def _coerce(raw: str, like: Any) -> Any:
    if isinstance(like, bool):
        return raw.lower() in {"1", "true", "yes"}
    if isinstance(like, int):
        return int(raw)
    if isinstance(like, float):
        return float(raw)
    return raw


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load the YAML config, then apply CONVSEG__SECTION__KEY overrides."""
    with open(path or os.environ.get("CONVSEG_CONFIG", DEFAULT_CONFIG)) as f:
        cfg = deepcopy(yaml.safe_load(f))
    for var, raw in os.environ.items():
        if not var.startswith(ENV_PREFIX):
            continue
        section, _, key = var[len(ENV_PREFIX):].lower().partition("__")
        if section in cfg and key in cfg[section]:
            cfg[section][key] = _coerce(raw, cfg[section][key])
    return cfg
