"""Benchmark workflow template.

Fill in each stage after the dataset, preprocessing, metrics, and first model
have been agreed upon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import find_placeholders, load_config


def run_benchmark(config_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Validate a recipe and mark the future benchmark stages.

    No benchmark is executed yet because the repository intentionally contains
    no model implementations.
    """

    config = (
        load_config(config_or_path)
        if isinstance(config_or_path, str | Path)
        else dict(config_or_path)
    )
    placeholders = find_placeholders(config)
    if placeholders:
        raise ValueError(f"complete the configuration first: {', '.join(placeholders)}")
    if not config["models"]:
        raise NotImplementedError("no models are configured; add the first model later")

    # TODO: Load the declared MCPL dataset.
    # TODO: Create deterministic train/validation/test splits.
    # TODO: Fit preprocessing on the training split only.
    # TODO: Train each registered model.
    # TODO: Generate neutrons and calculate the agreed metrics.
    # TODO: Save a reproducible report and artifacts.
    raise NotImplementedError("benchmark execution has not been implemented yet")

