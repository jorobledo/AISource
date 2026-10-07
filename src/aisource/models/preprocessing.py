import numpy as np
from __future__ import annotations
from scipy.special import ndtr, ndtri

class GaussRankTransform:
    """Per-feature rank transform to a standard normal with an interpolated inverse."""

    def fit(self, data: np.ndarray) -> GaussRankTransform:
        self.sorted_columns = np.sort(data, axis=0)
        count = len(data)
        self.grid = (np.arange(count) + 0.5) / count
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        count = len(self.sorted_columns)
        output = np.empty(data.shape, dtype=np.float64)
        for j in range(data.shape[1]):
            below = np.searchsorted(self.sorted_columns[:, j], data[:, j], side="left")
            at_or_below = np.searchsorted(self.sorted_columns[:, j], data[:, j], side="right")
            uniform = 0.5 * (below + at_or_below) / count
            output[:, j] = ndtri(np.clip(uniform, 0.5 / count, 1.0 - 0.5 / count))
        return output

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        uniform = ndtr(np.asarray(data, dtype=np.float64))
        output = np.empty(uniform.shape, dtype=np.float64)
        for j in range(uniform.shape[1]):
            output[:, j] = np.interp(uniform[:, j], self.grid, self.sorted_columns[:, j])
        return output


