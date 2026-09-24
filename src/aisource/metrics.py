"""Metric placeholders.

Decide the benchmark metrics and their parameters before implementing them.
Candidates include MMD, Wasserstein distance, marginal KS statistics,
correlation errors, and physical-validity checks.
"""

from __future__ import annotations

import numpy as np


def evaluate(reference: np.ndarray, generated: np.ndarray) -> dict[str, float]:
    """Compare reference and generated particles.

    TODO: implement only after the metric protocol is agreed upon.
    """

    raise NotImplementedError("benchmark metrics have not been selected yet")

