"""Named assertions for the two v3 -> v4 breaking changes nothing states directly.

Both are already covered incidentally — reverting the column order fails 20
tests, and removing the legacy API would fail every import of it. Neither
failure *says* what broke, though: a reader watching
`test_distributionMinMaxRepresentation` fail has no reason to suspect the
column-order contract. These tests exist so that the breakage names itself.
"""

import importlib.util

import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import aggregate

# testdata.csv is deliberately not alphabetical, so sorting and preserving the
# input order are distinguishable. A fixture in alphabetical order would make
# this test pass under either behaviour.
RAW = pd.read_csv(TESTDATA_CSV, index_col=0)


def test_fixture_is_not_alphabetical():
    """Guard for the two tests below, which are vacuous without it."""
    assert list(RAW.columns) != sorted(RAW.columns)


def test_outputs_keep_the_input_column_order():
    """v3 sorted output columns alphabetically; v4 returns them as given."""
    result = aggregate(RAW.copy(), n_clusters=4, period_duration=24)

    assert list(result.cluster_representatives.columns) == list(RAW.columns)
    assert list(result.reconstructed.columns) == list(RAW.columns)
    assert list(result.original.columns) == list(RAW.columns)


@pytest.mark.parametrize(
    "module", ["tsam.timeseriesaggregation", "tsam.hyperparametertuning"]
)
def test_legacy_modules_are_gone(module):
    """v4 removed the class-based API; `tsam.aggregate()` is the entry point."""
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize(
    "name", ["TimeSeriesAggregation", "unstackToPeriods", "LegacyAPIWarning"]
)
def test_legacy_names_are_gone(name):
    import tsam

    assert not hasattr(tsam, name)
