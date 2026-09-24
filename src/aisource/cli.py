"""Small command-line interface for the AISource scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmark import run_benchmark
from .config import find_placeholders, load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aisource")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate a benchmark template")
    validate.add_argument("config", type=Path)
    benchmark = commands.add_parser("benchmark", help="future benchmark entry point")
    benchmark.add_argument("config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        config = load_config(arguments.config)
        placeholders = find_placeholders(config)
        print(f"Valid template: {arguments.config}")
        if placeholders:
            print("Fields to complete:")
            for field in placeholders:
                print(f"- {field}")
        else:
            print("No TODO fields remain.")
        return 0
    if arguments.command == "benchmark":
        run_benchmark(arguments.config)
        return 0
    return 2
