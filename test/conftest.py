"""Pytest configuration and shared fixtures for tsam tests."""

from pathlib import Path

import pandas as pd
import pytest

# Path to test data directory
TEST_DIR = Path(__file__).parent
TEST_DATA_DIR = TEST_DIR / "data"

# Path to examples directory (for testdata.csv which is shared with notebooks)
EXAMPLES_DIR = TEST_DIR.parent / "docs" / "source" / "examples_notebooks"

# Common test data paths
TESTDATA_CSV = EXAMPLES_DIR / "testdata.csv"
RESULTS_DIR = TEST_DATA_DIR  # Expected output CSV files for tests
GOLDEN_DIR = TEST_DATA_DIR / "golden"

# Truncate test data to 91 days (2184h) — must be divisible by 24 (hoursPerPeriod).
MAX_TIMESTEPS = 2184


def load_testdata(parse_dates: bool = False) -> pd.DataFrame:
    """Load testdata.csv truncated to MAX_TIMESTEPS rows."""
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=parse_dates).iloc[
        :MAX_TIMESTEPS
    ]


# Validate paths exist (helpful for debugging if directory structure changes)
if not TESTDATA_CSV.exists():
    raise FileNotFoundError(f"Test data file not found: {TESTDATA_CSV}")
if not RESULTS_DIR.exists():
    raise FileNotFoundError(f"Test results directory not found: {RESULTS_DIR}")


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Regenerate golden CSV files from the new API results.",
    )


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")
