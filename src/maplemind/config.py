"""Configuration loading with deep-merge overrides and env-var overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "settings.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def load_settings(*override_paths: str | Path, overrides: dict | None = None) -> dict:
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    for p in override_paths:
        if p:
            with open(p) as f:
                cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    # env overrides: MAPLEMIND__RETRIEVAL__TOP_K=6
    for key, val in os.environ.items():
        if key.startswith("MAPLEMIND__"):
            parts = key.lower().split("__")[1:]
            node = cfg
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = _coerce(val)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
