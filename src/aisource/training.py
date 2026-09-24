"""Training-loop placeholder."""

from __future__ import annotations

import numpy as np

from .models import BaseGenerator


def train_model(
    model: BaseGenerator,
    train: np.ndarray,
    validation: np.ndarray | None = None,
) -> BaseGenerator:
    """Fit one future model through the common interface."""

    model.fit(train, validation)
    return model

