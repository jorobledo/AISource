# AISource

Template repository for benchmarking generative AI models on neutron
phase-space distributions stored in MCPL files.

This project follows *Machine Learning for neutron source distributions*
([DOI: 10.1088/2632-2153/ae8d7e](https://doi.org/10.1088/2632-2153/ae8d7e)).

## Status

This repository is intentionally a scaffold. It currently contains no model
implementations and no datasets. We will add these later.

## Planned benchmark cases

### Neutron-source generation

- Reactor source
- Spallation source
- Compact accelerator-driven neutron source (CANS), such as HBS

### Source estimation before the sample

- Diffraction
- SANS
- INS
- TOF

## Structure

```text
src/aisource/
  models/             model interface; implementations will go here
  benchmark.py        benchmark workflow template
  metrics.py          metric placeholders
  training.py         training-loop placeholder
  visualization.py    plotting placeholder

configs/
  benchmark.template.yaml
  datasets/dataset.template.yaml

datasets/
  reference/          local MCPL files; not committed
  README.md           dataset documentation checklist

pretrained/           future checkpoints; not committed
notebooks/            benchmark notebook template
docs/                 Jupyter Book source
tests/                tests to extend with each implementation
```

## What to fill in next

1. Copy and complete the dataset template.
2. Agree on the common MCPL features and preprocessing.
3. Agree on the benchmark metrics and data splits.
4. Add one model under `src/aisource/models/`.
5. Connect that model to `benchmark.py`.
6. Add tests and document the result.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
aisource validate configs/benchmark.template.yaml
```

See [docs/index.md](docs/index.md) for the short project checklist.
