"""Metrics for comparing two multivariate sample distributions.

Distance-based metrics use a pooled standardization by default so that MCPL
features with different units do not dominate solely because of their scale.

All samples are arrays shaped ``(particles, parameters)``: rows are particles
and columns are parameters. Reference and generated samples may contain
different numbers of particles but must have the same parameters.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import ndtri
from scipy.stats import binomtest

_METRIC_ALIASES = {
    "mmd": "mmd_rbf",
    "mmd_rbf": "mmd_rbf",
    "mmd_rbf_squared": "mmd_rbf",
    "c2st": "c2st",
    "energy": "energy_distance",
    "energy_distance": "energy_distance",
    "kld": "kl_divergence",
    "kl_divergence": "kl_divergence",
    "all": "all",
}


def _validate_samples(
    reference: np.ndarray,
    generated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray(reference, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if reference.ndim != 2 or generated.ndim != 2:
        raise ValueError("reference and generated samples must be two-dimensional")
    if reference.shape[1] != generated.shape[1]:
        raise ValueError("reference and generated samples must have the same features")
    if len(reference) < 2 or len(generated) < 2:
        raise ValueError("each sample must contain at least two rows")
    if not np.isfinite(reference).all() or not np.isfinite(generated).all():
        raise ValueError("samples must contain only finite values")
    return reference, generated


def _subsample(
    values: np.ndarray,
    maximum: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if maximum is None or len(values) <= maximum:
        return values
    if maximum < 2:
        raise ValueError("max_samples must be at least two")
    return values[rng.choice(len(values), maximum, replace=False)]


def _prepare_pair(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    max_samples: int | None,
    seed: int,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    reference, generated = _validate_samples(reference, generated)
    rng = np.random.default_rng(seed)
    reference = _subsample(reference, max_samples, rng)
    generated = _subsample(generated, max_samples, rng)
    if standardize:
        pooled = np.concatenate((reference, generated))
        center = pooled.mean(axis=0)
        scale = pooled.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        reference = (reference - center) / scale
        generated = (generated - center) / scale
    return reference, generated


def _median_bandwidth(values: np.ndarray, seed: int, max_pairs: int = 20_000) -> float:
    rng = np.random.default_rng(seed)
    left = rng.integers(0, len(values), size=max_pairs)
    right = rng.integers(0, len(values), size=max_pairs)
    squared = np.square(values[left] - values[right]).sum(axis=1)
    squared = squared[squared > 0]
    if not len(squared):
        return 1.0
    return float(np.sqrt(np.median(squared)))


def _rbf_kernel_sum(left: np.ndarray, right: np.ndarray, bandwidth: float) -> float:
    total = 0.0
    denominator = 2.0 * bandwidth**2
    for start in range(0, len(left), 512):
        squared = cdist(left[start : start + 512], right, metric="sqeuclidean")
        total += np.exp(-squared / denominator).sum()
    return float(total)


def mmd_rbf(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    bandwidth: float | None = None,
    estimator: Literal["unbiased", "biased"] = "unbiased",
    max_samples: int | None = 2_000,
    seed: int = 17,
    standardize: bool = True,
) -> float:
    """Return squared maximum mean discrepancy with an RBF kernel.

    If ``bandwidth`` is omitted, the median pairwise-distance heuristic is used.
    The unbiased estimate can be slightly negative at finite sample size.
    """

    reference, generated = _prepare_pair(
        reference,
        generated,
        max_samples=max_samples,
        seed=seed,
        standardize=standardize,
    )
    if estimator not in {"unbiased", "biased"}:
        raise ValueError("estimator must be 'unbiased' or 'biased'")
    if bandwidth is None:
        bandwidth = _median_bandwidth(np.concatenate((reference, generated)), seed)
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and positive")

    xx = _rbf_kernel_sum(reference, reference, bandwidth)
    yy = _rbf_kernel_sum(generated, generated, bandwidth)
    xy = _rbf_kernel_sum(reference, generated, bandwidth)
    n, m = len(reference), len(generated)
    if estimator == "unbiased":
        xx = (xx - n) / (n * (n - 1))
        yy = (yy - m) / (m * (m - 1))
    else:
        xx /= n**2
        yy /= m**2
    return float(xx + yy - 2.0 * xy / (n * m))


def c2st(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    classifier: Literal["logistic", "random_forest"] = "logistic",
    folds: int = 5,
    max_samples: int | None = 10_000,
    seed: int = 17,
) -> dict[str, float]:
    """Run a cross-validated classifier two-sample test.

    Samples are balanced before fitting. Accuracy and ROC AUC near 0.5 indicate
    that the selected classifier cannot distinguish the two samples. The result
    is classifier-dependent and is not proof that the distributions are equal.
    """

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    reference, generated = _validate_samples(reference, generated)
    rng = np.random.default_rng(seed)
    count = min(len(reference), len(generated))
    if max_samples is not None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least two")
        count = min(count, max_samples)
    reference = _subsample(reference, count, rng)
    generated = _subsample(generated, count, rng)
    values = np.concatenate((reference, generated))
    labels = np.concatenate((np.zeros(count, dtype=int), np.ones(count, dtype=int)))
    if folds < 2 or folds > count:
        raise ValueError("folds must be between two and the per-distribution sample count")

    if classifier == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2_000, random_state=seed),
        )
    elif classifier == "random_forest":
        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=seed,
        )
    else:
        raise ValueError("classifier must be 'logistic' or 'random_forest'")
    cross_validation = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        # Some Accelerate-backed NumPy builds emit spurious warnings for finite
        # matrix products. Do not suppress convergence or other model warnings.
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in matmul",
            category=RuntimeWarning,
        )
        probability = cross_val_predict(
            model,
            values,
            labels,
            cv=cross_validation,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
    prediction = (probability >= 0.5).astype(int)
    correct = int((prediction == labels).sum())
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pvalue": float(binomtest(correct, len(labels), 0.5, alternative="greater").pvalue),
    }


def energy_distance(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    max_samples: int | None = 2_000,
    seed: int = 17,
    standardize: bool = True,
) -> float:
    """Return the multivariate sample energy distance."""

    reference, generated = _prepare_pair(
        reference,
        generated,
        max_samples=max_samples,
        seed=seed,
        standardize=standardize,
    )
    cross = cdist(reference, generated).mean()
    within_reference = cdist(reference, reference).mean()
    within_generated = cdist(generated, generated).mean()
    squared = max(2.0 * cross - within_reference - within_generated, 0.0)
    return float(np.sqrt(squared))


def gauss_rank_transform(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map every feature of ``values`` to a standard normal using ``reference``.

    The empirical CDF of each reference feature gives uniform scores (mid-ranks
    for ties, kept strictly inside (0, 1)), which the inverse normal CDF maps to
    Gaussian values.
    """

    count = len(reference)
    transformed = np.empty_like(values, dtype=np.float64)
    for column in range(values.shape[1]):
        ordered = np.sort(reference[:, column])
        below = np.searchsorted(ordered, values[:, column], side="left")
        at_or_below = np.searchsorted(ordered, values[:, column], side="right")
        uniform = 0.5 * (below + at_or_below) / count
        transformed[:, column] = ndtri(np.clip(uniform, 0.5 / count, 1.0 - 0.5 / count))
    return transformed


def _gaussian_fit(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    centered = values - mean
    return mean, centered.T @ centered / (len(values) - 1)


def kl_divergence(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    max_samples: int | None = None,
    seed: int = 17,
) -> float:
    """Return KL(reference || generated) in nats after Gauss rank transformation.

    Both samples are transformed with the empirical CDFs of the reference, a
    multivariate Gaussian is fitted to each, and the closed-form Gaussian KL
    divergence is evaluated. The covariance matrices are ``(parameters,
    parameters)``, so each sample needs more particles than parameters.
    """

    reference, generated = _prepare_pair(
        reference,
        generated,
        max_samples=max_samples,
        seed=seed,
        standardize=False,
    )
    transformed_reference = gauss_rank_transform(reference, reference)
    transformed_generated = gauss_rank_transform(reference, generated)

    dimension = reference.shape[1]
    mean_p, cov_p = _gaussian_fit(transformed_reference)
    mean_q, cov_q = _gaussian_fit(transformed_generated)
    if np.linalg.matrix_rank(cov_p) < dimension or np.linalg.matrix_rank(cov_q) < dimension:
        raise ValueError(
            "covariance matrix is singular; use more particles than parameters "
            "and avoid perfectly collinear parameters"
        )
    _, logdet_p = np.linalg.slogdet(cov_p)
    _, logdet_q = np.linalg.slogdet(cov_q)

    delta = mean_p - mean_q
    trace = np.trace(np.linalg.solve(cov_q, cov_p))
    mahalanobis = delta @ np.linalg.solve(cov_q, delta)
    return float(0.5 * (trace + mahalanobis - dimension + logdet_q - logdet_p))


def evaluate(
    reference: np.ndarray,
    generated: np.ndarray,
    *,
    seed: int = 17,
    c2st_classifier: Literal["logistic", "random_forest"] = "logistic",
    max_samples: int = 2_000,
) -> dict[str, Any]:
    """Calculate metrics on the complete multivariate sample vectors."""

    reference, generated = _validate_samples(reference, generated)
    classifier_result = c2st(
        reference,
        generated,
        classifier=c2st_classifier,
        max_samples=max_samples,
        seed=seed,
    )
    return {
        "mmd_rbf_squared": mmd_rbf(reference, generated, max_samples=max_samples, seed=seed),
        "c2st_accuracy": classifier_result["accuracy"],
        "c2st_roc_auc": classifier_result["roc_auc"],
        "c2st_pvalue": classifier_result["pvalue"],
        "energy_distance": energy_distance(
            reference, generated, max_samples=max_samples, seed=seed
        ),
        "kl_divergence": kl_divergence(reference, generated, max_samples=max_samples, seed=seed),
    }


class Metrics:
    """Dispatch distribution metrics for a pair of multivariate samples.

    Samples can either be bound to the instance::

        metrics = Metrics(reference, generated)
        result = metrics.evaluate("mmd")

    or supplied for each evaluation::

        metrics = Metrics()
        result = metrics.evaluate("mmd", reference, generated)

    Parameters passed to :meth:`evaluate` are forwarded to the selected metric.
    ``seed`` and ``max_samples`` provide consistent defaults across calls, and
    can also be overridden for an individual evaluation.
    """

    available_metrics = ("mmd_rbf", "c2st", "energy_distance", "kl_divergence", "all")

    def __init__(
        self,
        reference: np.ndarray | None = None,
        generated: np.ndarray | None = None,
        *,
        seed: int = 17,
        max_samples: int | None = 2_000,
    ) -> None:
        if (reference is None) != (generated is None):
            raise ValueError("reference and generated must be provided together")
        if reference is not None and generated is not None:
            reference, generated = _validate_samples(reference, generated)
        self.reference = reference
        self.generated = generated
        self.seed = seed
        self.max_samples = max_samples

    def evaluate(
        self,
        metric: str,
        reference: np.ndarray | None = None,
        generated: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float | dict[str, float] | dict[str, Any]:
        """Evaluate ``metric`` for the supplied or instance-bound samples.

        Supported names are ``"mmd"``/``"mmd_rbf"``, ``"c2st"``,
        ``"energy"``/``"energy_distance"``, ``"kld"``/``"kl_divergence"``, and
        ``"all"``.
        """

        reference = self.reference if reference is None else reference
        generated = self.generated if generated is None else generated
        if reference is None or generated is None:
            raise ValueError("reference and generated must be supplied to Metrics or evaluate")
        if not isinstance(metric, str):
            raise TypeError("metric must be a string")
        normalized = metric.lower().replace("-", "_")
        try:
            selected = _METRIC_ALIASES[normalized]
        except KeyError as error:
            choices = ", ".join(self.available_metrics)
            raise ValueError(f"unknown metric {metric!r}; choose one of: {choices}") from error

        kwargs.setdefault("seed", self.seed)
        kwargs.setdefault("max_samples", self.max_samples)
        if selected == "mmd_rbf":
            return mmd_rbf(reference, generated, **kwargs)
        if selected == "c2st":
            return c2st(reference, generated, **kwargs)
        if selected == "energy_distance":
            return energy_distance(reference, generated, **kwargs)
        if selected == "kl_divergence":
            return kl_divergence(reference, generated, **kwargs)
        return evaluate(reference, generated, **kwargs)

    def evaluate_all(
        self,
        reference: np.ndarray | None = None,
        generated: np.ndarray | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate the complete metric suite."""

        result = self.evaluate("all", reference, generated, **kwargs)
        assert isinstance(result, dict)
        return result
