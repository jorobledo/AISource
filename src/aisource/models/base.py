"""Minimal interface that future generative models should implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Particle:
    """Describe the phase-space features associated with a particle type."""

    def __init__(self, phase_space_dict: dict[str, object], particle_type: str) -> None:
        self.phase_space_dict = phase_space_dict
        self.training_phase_space = list(phase_space_dict.keys())
        self.ndim = len(self.training_phase_space)
        self.particle_type = particle_type


class BaseGenerator(ABC):
    """Contract shared by future model implementations."""

    def __init__(self, particle: Particle) -> None:
        self.ndim = particle.ndim
        self.Particle = particle

    @abstractmethod
    def fit(self, train: np.ndarray, validation: np.ndarray | None = None) -> None:
        """Fit the model on the training split only."""

    @abstractmethod
    def sample(self, n: int, seed: int) -> np.ndarray:
        """Generate ``n`` rows in the same feature order as the training data."""

    def _check_input_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.ndim:
            raise ValueError(f"expected an array with shape (n, {self.ndim}), instead got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("data must contain only finite values")
        return values
