import numpy as np
import pytest

pytest.importorskip("torch")

import torch

from aisource.models import BaseGenerator, Particle
from aisource.models.Continuous_flow_matching import ContinuousFlowMatching


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
    options = dict(width=32, depth=2, steps=300, batch_size=256, eval_every=100)
    options.update(kwargs)
    return ContinuousFlowMatching(particle, **options)


def test_is_a_base_generator(particle):
    assert isinstance(small_model(particle), BaseGenerator)


def test_fit_and_sample_reproduce_the_data(particle, data):
    model = small_model(particle, steps=3_000, eval_every=1_000)
    model.fit(data[:1_800], data[1_800:])
    samples = model.sample(2_000, seed=1)
    assert samples.shape == (2_000, 2)
    assert model.val_losses[-1][1] < model.val_losses[0][1]
    assert np.all(samples >= data.min(axis=0) - 1e-5)
    assert np.all(samples <= data.max(axis=0) + 1e-5)
    assert np.allclose(np.median(samples, axis=0), np.median(data, axis=0), rtol=0.3)
    assert np.corrcoef(samples.T)[0, 1] > 0.9


def test_sampling_is_reproducible_with_a_global_seed(particle, data):
    model = small_model(particle, steps=20, eval_every=10)
    model.fit(data[:1_800], data[1_800:])
    torch.manual_seed(3)
    first = model.sample(100, seed=3)
    torch.manual_seed(3)
    assert np.array_equal(first, model.sample(100, seed=3))
    torch.manual_seed(4)
    assert not np.array_equal(first, model.sample(100, seed=3))


def test_losses_are_recorded_at_the_evaluation_steps(particle, data):
    model = small_model(particle, steps=25, eval_every=10)
    model.fit(data[:1_800], data[1_800:])
    assert len(model.losses) == 25
    assert [step for step, _ in model.val_losses] == [0, 10, 20, 24]


def test_fit_rejects_invalid_input(particle, data):
    model = small_model(particle, steps=5)
    with pytest.raises(ValueError, match="shape"):
        model.fit(np.ones((10, 3)), data)
    with pytest.raises(ValueError, match="shape"):
        model.fit(data, np.ones((10, 3)))
    with pytest.raises(ValueError, match="finite"):
        model.fit(np.array([[0.0, np.nan]] * 10), data)
    with pytest.raises(ValueError, match="finite"):
        model.fit(data, np.array([[0.0, np.inf]] * 10))
