"""Pytest configuration and shared fixtures for tsam tests."""

from pathlib import Path

# Path to examples directory (moved to docs/source/examples_notebooks)
EXAMPLES_DIR = Path(__file__).parent.parent / "docs" / "source" / "examples_notebooks"

# Common test data paths
TESTDATA_CSV = EXAMPLES_DIR / "testdata.csv"
RESULTS_DIR = EXAMPLES_DIR / "results"
