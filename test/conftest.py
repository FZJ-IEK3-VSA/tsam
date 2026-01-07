"""Pytest configuration and shared fixtures for tsam tests."""

from pathlib import Path

# Path to test data directory
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"

# Path to examples directory (for testdata.csv which is shared with notebooks)
EXAMPLES_DIR = TEST_DIR.parent / "docs" / "source" / "examples_notebooks"

# Common test data paths
TESTDATA_CSV = EXAMPLES_DIR / "testdata.csv"
RESULTS_DIR = TEST_DATA_DIR  # Expected output CSV files for tests

# Validate paths exist (helpful for debugging if directory structure changes)
if not TESTDATA_CSV.exists():
    raise FileNotFoundError(f"Test data file not found: {TESTDATA_CSV}")
if not RESULTS_DIR.exists():
    raise FileNotFoundError(f"Test results directory not found: {RESULTS_DIR}")
