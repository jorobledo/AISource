from pathlib import Path

import pytest

from aisource.benchmark import run_benchmark
from aisource.config import find_placeholders, load_config

TEMPLATE = Path("configs/benchmark.template.yaml")


def test_benchmark_template_is_valid_yaml():
    config = load_config(TEMPLATE)
    assert config["models"] == []
    assert "experiment.name" in find_placeholders(config)


def test_benchmark_stays_unimplemented_without_models():
    config = load_config(TEMPLATE)
    config["experiment"]["name"] = "test"
    config["experiment"]["description"] = "test"
    config["experiment"]["output_dir"] = "outputs/test"
    config["split"] = {
        "train_fraction": 0.7,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
    }
    config["preprocessing"] = {
        "log_features": [],
        "normalize": False,
        "weight_handling": "TODO",
    }
    config["metrics"] = {"names": [], "evaluation_samples": 100}
    config["preprocessing"]["weight_handling"] = "ignore"
    with pytest.raises(NotImplementedError, match="no models"):
        run_benchmark(config)
