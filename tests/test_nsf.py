import pytest 

pytest.importorskip('torch')

import torch

from aisource import NSFlow, Particle

def test_nsf_sample():
    particle = Particle(training_phase_space=["x", "y", "z"], particle_type="test")
    model = NSFlow(particle=particle, device="cpu", seed=42)
    assert isinstance(model, NSFlow)