"""Confirm that the complete offline ML foundations course can run.

Run after ``python -m pip install -r requirements.txt``:

    python check_setup.py

The check validates Python and dependency versions, imports every concept module,
discovers tests and examples, then runs the capstone twice in memory. It downloads no
model and makes no network call.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).parent
MODULES = (
    "attention",
    "calibration",
    "losses",
    "memory",
    "optimization",
    "probabilities",
    "quantization",
    "tensors",
    "tiny_lm",
    "transformer",
)
EXPECTED_VERSIONS = {"numpy": "2.5.2", "torch": "2.13.0"}


def main() -> int:
    """Run offline readiness checks and return a shell-friendly status."""

    errors: list[str] = []
    print("ML foundations setup")
    print(f"  Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 12):
        errors.append("Python 3.12 or newer is required")

    for package, expected in EXPECTED_VERSIONS.items():
        try:
            observed = version(package)
            public_version = observed.split("+", maxsplit=1)[0]
            if public_version != expected:
                errors.append(f"{package} must be {expected}, found {observed}")
            else:
                print(f"  {package}: {observed} OK")
        except PackageNotFoundError:
            errors.append(f"{package} is not installed")

    try:
        import ml_foundations

        for module in MODULES:
            importlib.import_module(f"ml_foundations.{module}")
        print(f"  package: {ml_foundations.__name__} and {len(MODULES)} modules OK")
    except ImportError as error:
        errors.append(
            f"package import failed ({error}); run: python -m pip install -r requirements.txt"
        )

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    test_count = suite.countTestCases()
    if test_count == 0:
        errors.append("test discovery found zero tests")
    else:
        print(f"  tests: {test_count} discovered OK")

    examples = sorted((ROOT / "examples").glob("[0-9][0-9]_*.py"))
    if len(examples) != 11:
        errors.append(f"expected 11 numbered examples, found {len(examples)}")
    else:
        print("  examples: 11 numbered lessons OK")

    try:
        from hands_on.train_tiny_transformer import (
            ExperimentVerdict,
            run_experiment,
        )

        first = run_experiment()
        second = run_experiment()
        if first != second:
            errors.append("capstone output changed across identical runs")
        elif first.verdict is not ExperimentVerdict.READY_FOR_LAB_USE:
            errors.append(f"capstone verdict was {first.verdict.value}: {first.reason}")
        else:
            print("  capstone: deterministic ready_for_lab_use report OK")
    except Exception as error:  # Setup diagnostics should name unexpected failures.
        errors.append(f"capstone self-check failed: {type(error).__name__}: {error}")

    if errors:
        print("\nFix these before continuing:")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("\nAll lessons are ready. No external service is required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
