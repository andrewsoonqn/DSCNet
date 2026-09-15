#!/usr/bin/env python3
"""Validate candidate syntax locally or run focused tests on the cluster."""

from __future__ import annotations

import argparse
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOTS = (
    REPO_ROOT / "dscnet" / "models",
    REPO_ROOT / "dscnet" / "training",
    REPO_ROOT / "tests" / "models",
    REPO_ROOT / "tests" / "training",
)


def _compile_candidate_python() -> None:
    for root in PYTHON_ROOTS:
        for path in sorted(root.rglob("*.py")):
            compile(path.read_bytes(), str(path), "exec")


def _run_tests() -> bool:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for relative in ("tests/models", "tests/training"):
        suite.addTests(
            loader.discover(
                start_dir=str(REPO_ROOT / relative),
                pattern="test_*.py",
                top_level_dir=str(REPO_ROOT),
            )
        )
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true")
    options = parser.parse_args()
    _compile_candidate_python()
    return 0 if not options.run_tests or _run_tests() else 1


if __name__ == "__main__":
    raise SystemExit(main())
