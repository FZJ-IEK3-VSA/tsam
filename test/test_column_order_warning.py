"""Tests for the v4 column-order FutureWarning emitted by aggregate()."""

import warnings

import pandas as pd

from conftest import TESTDATA_CSV
from tsam import aggregate

# testdata.csv columns ['GHI', 'T', 'Wind', 'Load'] are intentionally not
# alphabetical. The legacy v3 path sorts columns in place, so pass copies.
RAW = pd.read_csv(TESTDATA_CSV, index_col=0)


def _order_warnings(record):
    return [w for w in record if "follow the input DataFrame" in str(w.message)]


def test_warns_and_explains_fixes_when_not_alphabetical():
    assert list(RAW.columns) != sorted(RAW.columns), "fixture must be unsorted"
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        aggregate(RAW.copy(), n_clusters=4, period_duration=24)
    found = _order_warnings(record)
    assert len(found) == 1
    message = str(found[0].message)
    assert "data.sort_index(axis=1)" in message  # sort before
    assert "result.cluster_representatives.sort_index(axis=1)" in message  # after
    assert "by column name" in message  # adopt v4 now
    assert "filterwarnings" in message  # how to silence


def test_documented_silencing_filter_actually_silences():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "ignore", category=FutureWarning, message=".*sorted alphabetically.*"
        )
        aggregate(RAW.copy(), n_clusters=4, period_duration=24)
    assert _order_warnings(record) == []


def test_no_warning_when_already_alphabetical():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        aggregate(RAW.sort_index(axis=1), n_clusters=4, period_duration=24)
    assert _order_warnings(record) == []
