import numpy as np

from aisource.models.preprocessing import GaussRankTransform


def test_gauss_rank_transform_gives_standard_normal_marginals():
    rng = np.random.default_rng(0)
    data = rng.exponential(scale=2.0, size=(2_000, 3))
    gaussian = GaussRankTransform().fit(data).transform(data)
    assert np.allclose(gaussian.mean(axis=0), 0, atol=1e-2)
    assert np.allclose(gaussian.std(axis=0), 1, atol=2e-2)


def test_gauss_rank_transform_round_trips():
    rng = np.random.default_rng(1)
    data = rng.exponential(scale=2.0, size=(2_000, 3))
    transform = GaussRankTransform().fit(data)
    restored = transform.inverse_transform(transform.transform(data))
    assert np.allclose(restored, data, rtol=1e-2, atol=1e-2)


def test_inverse_transform_stays_within_the_training_range():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(500, 2))
    transform = GaussRankTransform().fit(data)
    extreme = np.array([[-50.0, 50.0], [50.0, -50.0]])
    restored = transform.inverse_transform(extreme)
    assert np.all(restored >= data.min(axis=0)) and np.all(restored <= data.max(axis=0))
