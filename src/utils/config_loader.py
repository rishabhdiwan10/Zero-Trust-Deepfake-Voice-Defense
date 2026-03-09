"""
src.utils.config_loader
=========================
YAML configuration loader utility for the Zero-Trust Deepfake Voice Defense
System.

Provides a simple ``load_config`` function that reads a YAML file and returns
a nested Python dict, with optional dot-notation accessor support.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def load_config(
    config_path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML file. Relative paths are resolved from the
        project root (directory containing ``configs/``).
    overrides : dict | None
        Optional flat dict of dot-notation key overrides, e.g.
        ``{"training.batch_size": 64}``. Applied after loading.

    Returns
    -------
    dict
        Parsed configuration as a nested Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        # Try resolving relative to cwd, then project root
        candidates = [
            Path.cwd() / config_path,
            _project_root() / config_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = candidate
                break

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        config: Dict[str, Any] = yaml.safe_load(fh) or {}

    if overrides:
        _apply_overrides(config, overrides)

    return config


def _apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> None:
    """Apply dot-notation overrides to a nested config dict in place."""
    for key, value in overrides.items():
        keys = key.split(".")
        obj = config
        for k in keys[:-1]:
            obj = obj.setdefault(k, {})
        obj[keys[-1]] = value


def _project_root() -> Path:
    """
    Heuristically locate the project root by searching upward for
    ``configs/`` or ``setup.py``.
    """
    current = Path(__file__).resolve().parent
    for _ in range(10):  # max 10 levels up
        if (current / "configs").exists() or (current / "setup.py").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path.cwd()


def get_nested(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Retrieve a value from a nested config dict using dot notation.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    key : str
        Dot-separated key path, e.g. ``"training.batch_size"``.
    default : Any
        Value to return if the key is not found.

    Returns
    -------
    Any
        The value at the specified key path, or ``default``.
    """
    keys = key.split(".")
    obj: Any = config
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k, default)
        if obj is default:
            return default
    return obj
