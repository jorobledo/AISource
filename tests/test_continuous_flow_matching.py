import numpy as np
import pytest

pytest.importorskip("torch")

from aisource.models import BaseGenerator, Particle
from aisource.models.Continuous_flow_matching import (
    ContinuousFlowMatching,
    GaussRankTransform,
)


@pytest.fixture
def particle():
    return Particle({"x": None, "y": None}, "neutron")


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x = rng.exponential(scale=2.0, size=2_000)
    y = 3.0 + 0.5 * x + rng.normal(scale=0.3, size=2_000)
    return np.column_stack([x, y])


def small_model(particle, **kwargs):
    options = dict(width=32, depth=2, steps=300, batch_size=256, eval_every=100, val_size=200)
    options.update(kwargs)
    return ContinuousFlowMatching(particle, **options)


def test_is_a_base_generator(particle):
    assert isinstance(small_model(particle), BaseGenerator)


def test_gauss_rank_transform_round_trips(data):
    transform = GaussRankTransform().fit(data)
    gaussian = transform.transform(data)
    assert np.allclose(gaussian.mean(axis=0), 0, atol=1e-2)
    assert np.allclose(gaussian.std(axis=0), 1, atol=2e-2)
    assert np.allclose(transform.inverse_transform(gaussian), data, rtol=1e-2, atol=1e-2)


def test_fit_and_sample_reproduce_the_data(particle, data):
    model = small_model(particle, steps=3_000, eval_every=1_000)
    model.fit(data)
    samples = model.sample(2_000, seed=1)
    assert samples.shape == (2_000, 2)
    assert model.val_losses[-1][1] < model.val_losses[0][1]
    assert np.all(samples >= data.min(axis=0) - 1e-5) and np.all(samples <= data.max(axis=0) + 1e-5)
    assert np.allclose(np.median(samples, axis=0), np.median(data, axis=0), rtol=0.3)
    assert np.corrcoef(samples.T)[0, 1] > 0.9


def test_sampling_is_deterministic_per_seed(particle, data):
    model = small_model(particle, steps=20, eval_every=10)
    model.fit(data)
    first = model.sample(100, seed=3)
    assert np.array_equal(first, model.sample(100, seed=3))
    assert not np.array_equal(first, model.sample(100, seed=4))


def test_explicit_validation_and_no_transform(particle, data):
    model = small_model(particle, steps=20, eval_every=10, gauss_rank=False)
    model.fit(data[:1_500], data[1_500:])
    assert model.sample(10, seed=0).shape == (10, 2)


def test_input_validation(particle, data):
    model = small_model(particle, steps=5)
    with pytest.raises(RuntimeError, match="fit must be called"):
        model.sample(5, seed=0)
    with pytest.raises(ValueError, match="shape"):
        model.fit(np.ones((10, 3)))
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([[0.0, np.nan]] * 10))
