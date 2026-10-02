import numpy as np
import pytest

from aisource.metrics import (
    Metrics,
    c2st,
    energy_distance,
    evaluate,
    gauss_rank_transform,
    kl_divergence,
    mmd_rbf,
)


@pytest.fixture
def samples():
    rng = np.random.default_rng(8)
    reference = rng.normal(size=(600, 6))
    matching = rng.normal(size=(600, 6))
    shifted = rng.normal(loc=1.2, size=(600, 6))
    return reference, matching, shifted


def test_joint_metrics_detect_a_mean_shift(samples):
    reference, matching, shifted = samples
    assert mmd_rbf(reference, matching, seed=2) < mmd_rbf(reference, shifted, seed=2)
    assert energy_distance(reference, matching, seed=2) < energy_distance(
        reference, shifted, seed=2
    )


def test_c2st_detects_multivariate_shift(samples):
    reference, matching, shifted = samples
    matching_c2st = c2st(reference, matching, seed=4)
    shifted_c2st = c2st(reference, shifted, seed=4)
    assert 0.4 < matching_c2st["accuracy"] < 0.6
    assert shifted_c2st["accuracy"] > 0.75
    assert shifted_c2st["pvalue"] < 0.01


def test_dependence_metric_uses_joint_structure():
    rng = np.random.default_rng(10)
    reference = rng.normal(size=(800, 6))
    dependent = rng.normal(size=(800, 6))
    dependent[:, 1] = 0.9 * dependent[:, 0] + np.sqrt(1 - 0.9**2) * dependent[:, 1]
    assert mmd_rbf(reference, dependent, seed=3) > mmd_rbf(
        reference, rng.normal(size=(800, 6)), seed=3
    )


def test_evaluate_returns_documented_suite(samples):
    reference, matching, _ = samples
    result = evaluate(
        reference,
        matching,
        max_samples=300,
        seed=5,
    )
    assert set(result) == {
        "mmd_rbf_squared",
        "c2st_accuracy",
        "c2st_roc_auc",
        "c2st_pvalue",
        "energy_distance",
    }


def test_metrics_reject_invalid_shapes():
    with pytest.raises(ValueError, match="two-dimensional"):
        mmd_rbf(np.ones(10), np.ones((10, 1)))
    with pytest.raises(ValueError, match="same features"):
        mmd_rbf(np.ones((10, 2)), np.ones((10, 3)))
    with pytest.raises(ValueError, match="finite"):
        energy_distance(np.array([[0.0], [np.nan]]), np.ones((2, 1)))


def test_metrics_class_supports_bound_and_per_call_samples(samples):
    reference, matching, shifted = samples
    metrics = Metrics(reference, matching, seed=2, max_samples=300)

    assert metrics.evaluate("mmd") == mmd_rbf(reference, matching, seed=2, max_samples=300)
    assert metrics.evaluate("energy", reference, shifted) == energy_distance(
        reference, shifted, seed=2, max_samples=300
    )

    unbound = Metrics(seed=4, max_samples=200)
    assert unbound.evaluate("c2st", reference, shifted)["accuracy"] > 0.75
    assert set(unbound.evaluate_all(reference, matching)) == {
        "mmd_rbf_squared",
        "c2st_accuracy",
        "c2st_roc_auc",
        "c2st_pvalue",
        "energy_distance",
    }


def test_metrics_class_reports_missing_samples_and_unknown_metrics(samples):
    reference, matching, _ = samples
    with pytest.raises(ValueError, match="provided together"):
        Metrics(reference)
    with pytest.raises(ValueError, match="must be supplied"):
        Metrics().evaluate("mmd")
    with pytest.raises(ValueError, match="unknown metric"):
        Metrics(reference, matching).evaluate("not-a-metric")


def test_gauss_rank_transform_gives_standard_normal_marginals():
    rng = np.random.default_rng(1)
    values = rng.exponential(size=(2000, 3))
    transformed = gauss_rank_transform(values, values)
    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-2)
    assert np.allclose(transformed.std(axis=0), 1, atol=2e-2)


def test_kl_divergence_detects_shift_and_dependence(samples):
    reference, matching, shifted = samples
    matched = kl_divergence(reference, matching)
    assert 0 <= matched < 0.05
    assert kl_divergence(reference, shifted) > 10 * matched

    rng = np.random.default_rng(10)
    dependent = rng.normal(size=(600, 6))
    dependent[:, 1] = 0.9 * dependent[:, 0] + np.sqrt(1 - 0.9**2) * dependent[:, 1]
    assert kl_divergence(reference, dependent) > 10 * matched


def test_kl_divergence_is_invariant_to_monotonic_feature_maps(samples):
    reference, matching, _ = samples
    assert kl_divergence(np.exp(reference), np.exp(matching)) == pytest.approx(
        kl_divergence(reference, matching)
    )


def test_metrics_class_dispatches_kl_divergence(samples):
    reference, matching, shifted = samples
    metrics = Metrics(reference, matching)
    assert metrics.evaluate("kld") == kl_divergence(reference, matching)
    assert metrics.evaluate("kl_divergence", reference, shifted) == kl_divergence(
        reference, shifted
    )
    with pytest.raises(ValueError, match="regularization"):
        kl_divergence(reference, matching, regularization=-1.0)
