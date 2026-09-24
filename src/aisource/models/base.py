"""Minimal interface that future generative models should implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseGenerator(ABC):
    """Contract shared by future model implementations."""

    @abstractmethod
    def fit(self, train: np.ndarray, validation: np.ndarray | None = None) -> None:
        """Fit the model on the training split only."""

    @abstractmethod
    def sample(self, n: int, seed: int) -> np.ndarray:
        """Generate ``n`` rows in the same feature order as the training data."""
