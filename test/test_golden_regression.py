"""Golden-file regression tests for old and new tsam API.

Compares both the old ``TimeSeriesAggregation`` and the new
``tsam.aggregate()`` results against stored CSV files, catching
regressions where both APIs change together.

Golden files live in ``test/data/golden/{config_id}/{dataset}.csv``.

Usage::

    # Normal run — compare against stored golden files
    pytest test/test_golden_regression.py -v

    # Regenerate golden files (after an intentional change)
    pytest test/test_golden_regression.py --update-golden
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import GOLDEN_DIR
from test_old_new_equivalence import (
    CASES,
    EquivalenceCase,
    _case_ids,
    _get_data,
    _run_new,
    _run_old,
)


def _golden_path(case: EquivalenceCase) -> str:
    """Return the golden CSV path for a case (``golden/{config}/{dataset}.csv``)."""
    return GOLDEN_DIR / f"{case.id}.csv"


def _save_golden(df: pd.DataFrame, case: EquivalenceCase) -> None:
    path = _golden_path(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)


def _load_golden(case: EquivalenceCase) -> pd.DataFrame:
    path = _golden_path(case)
    return pd.read_csv(path, index_col=[0, 1])


class TestGoldenRegression:
    """Compare old and new API results against stored golden CSVs."""

    @pytest.mark.parametrize("case", CASES, ids=_case_ids(CASES))
    def test_update_golden(self, case: EquivalenceCase, update_golden):
        """Save new-API results as golden files (only runs with --update-golden)."""
        if not update_golden:
            pytest.skip("use --update-golden to regenerate")

        data = _get_data(case.dataset)
        new_result = _run_new(data, case)
        _save_golden(new_result.cluster_representatives, case)

    @pytest.mark.parametrize("case", CASES, ids=_case_ids(CASES))
    def test_new_api_matches_golden(self, case: EquivalenceCase, update_golden):
        """New API result must match stored golden CSV."""
        if update_golden:
            pytest.skip("updating golden files")

        path = _golden_path(case)
        if not path.exists():
            pytest.skip(
                f"golden file missing: {path.relative_to(GOLDEN_DIR.parent.parent)}"
            )

        data = _get_data(case.dataset)
        new_result = _run_new(data, case)
        golden = _load_golden(case)

        pd.testing.assert_frame_equal(
            new_result.cluster_representatives,
            golden,
            check_names=False,
        )

    @pytest.mark.parametrize("case", CASES, ids=_case_ids(CASES))
    def test_old_api_matches_golden(self, case: EquivalenceCase, update_golden):
        """Old API result must match stored golden CSV."""
        if update_golden:
            pytest.skip("updating golden files")

        path = _golden_path(case)
        if not path.exists():
            pytest.skip(
                f"golden file missing: {path.relative_to(GOLDEN_DIR.parent.parent)}"
            )

        data = _get_data(case.dataset)
        old_result, _ = _run_old(data, case)
        golden = _load_golden(case)

        pd.testing.assert_frame_equal(
            old_result,
            golden,
            check_names=False,
        )
