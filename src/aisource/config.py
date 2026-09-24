"""Configuration helpers for the repository templates."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a benchmark YAML file and check its top-level structure."""

    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("benchmark configuration must be a YAML mapping")
    required = {"experiment", "dataset", "split", "preprocessing", "metrics", "models"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"configuration is missing: {', '.join(missing)}")
    if not isinstance(config["models"], list):
        raise ValueError("models must be a list")
    config["_config_path"] = str(path.resolve())
    return config


def find_placeholders(value: Any, prefix: str = "") -> list[str]:
    """Return dotted paths containing TODO values or unresolved variables."""

    placeholders: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "_config_path":
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            placeholders.extend(find_placeholders(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            placeholders.extend(find_placeholders(item, f"{prefix}[{index}]"))
    elif isinstance(value, str) and ("TODO" in value or "${" in value):
        expanded = os.path.expandvars(value) if isinstance(value, str) else value
        if "TODO" in value or expanded == value:
            placeholders.append(prefix)
    return placeholders
