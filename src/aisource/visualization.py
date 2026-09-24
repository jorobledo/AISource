"""Visualization placeholders."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_diagnostics(
    reference: np.ndarray,
    generated: np.ndarray,
    output_dir: str | Path,
) -> None:
    """Save the agreed diagnostic plots.

    TODO: choose plots and visual conventions with the metric protocol.
    """

    raise NotImplementedError("diagnostic plots have not been selected yet")
