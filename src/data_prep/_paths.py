"""Resolve project root and data root for data_prep scripts (no hardcoded absolutes)."""
from __future__ import annotations

from pathlib import Path

import yaml

# --- Project / data root ---

_PROJECT_ROOT: Path | None = None
_DATA_ROOT: Path | None = None


def get_project_root() -> Path:
    """Return project root (parent of src/)."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    return _PROJECT_ROOT


def get_data_root() -> Path:
    """Return data root (project_root / DATA_ROOT from config or default 'data')."""
    global _DATA_ROOT
    if _DATA_ROOT is not None:
        return _DATA_ROOT
    root = get_project_root()
    config_path = root / "configs" / "paths.yml"
    data_root_str = "data"
    if config_path.is_file():
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            if isinstance(cfg, dict) and cfg.get("DATA_ROOT"):
                data_root_str = str(cfg["DATA_ROOT"]).strip()
        except (yaml.YAMLError, OSError):
            pass
    _DATA_ROOT = (root / data_root_str).resolve()
    return _DATA_ROOT
