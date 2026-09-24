.PHONY: install test lint benchmark docs clean

install:
	python -m pip install -e ".[all]"

test:
	pytest

lint:
	ruff check .

benchmark:
	aisource validate configs/benchmark.template.yaml

docs:
	jupyter-book build docs

clean:
	rm -rf build dist docs/_build .pytest_cache .ruff_cache htmlcov
