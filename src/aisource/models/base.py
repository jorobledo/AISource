"""Minimal interface that future generative models should implement."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class Particle():
    
    def __init__(self, phase_space_dict, particle_type):
        self.phase_space_dict = phase_space_dict
        self.training_phase_space = list(phase_space_dict.keys())
        self.ndim = len(self.training_phase_space)
        self.particle_type = particle_type


class BaseGenerator(ABC):
    """Contract shared by future model implementations."""

    def __init__(self, Particle):
        self.ndim = Particle.ndim
        self.Particle = Particle
        
    @abstractmethod
    def fit(self, train: np.ndarray, validation: np.ndarray | None = None) -> None:
        """Fit the model on the training split only."""

    @abstractmethod
    def sample(self, n: int, seed: int) -> np.ndarray:
        """Generate ``n`` rows in the same feature order as the training data."""
