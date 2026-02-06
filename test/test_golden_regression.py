"""Golden-file regression tests for old and new tsam API.

Stores **reconstructed** time series (``predictOriginalData`` / ``result.reconstructed``)
produced by the old (pre-v3.0.0) API as golden baselines.  Both old and new API
results are compared against these baselines, catching regressions where both
APIs change together.

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

from _configs import GOLDEN_DIR, case_ids, get_data
from test_old_new_equivalence import (
    CASES,
    EquivalenceCase,
    _run_new,
    _run_old,
)


def _golden_path(case: EquivalenceCase) -> str:
    """Return the golden CSV path for a case (``golden/{config}/{dataset}.csv``)."""
    return GOLDEN_DIR / f"{case.id}.csv"


def _save_golden(df: pd.DataFrame, case: EquivalenceCase) -> None:
    path = _golden_path(case)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.round(8).to_csv(path)


def _load_golden(case: EquivalenceCase) -> pd.DataFrame:
    path = _golden_path(case)
    return pd.read_csv(path, index_col=0, parse_dates=True)


class TestGoldenRegression:
    """Compare old and new API reconstructed results against stored golden CSVs."""

    @pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
    def test_update_golden(self, case: EquivalenceCase, update_golden):
        """Save old-API reconstructed results as golden files (only with --update-golden)."""
        if not update_golden:
            pytest.skip("use --update-golden to regenerate")

        data = get_data(case.dataset)
        _, old_agg = _run_old(data, case)
        _save_golden(old_agg.predictOriginalData(), case)

    @pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
    def test_new_api_matches_golden(self, case: EquivalenceCase, update_golden):
        """New API reconstructed result must match stored golden CSV."""
        if update_golden:
            pytest.skip("updating golden files")

        path = _golden_path(case)
        if not path.exists():
            pytest.skip(
                f"golden file missing: {path.relative_to(GOLDEN_DIR.parent.parent)}"
            )

        data = get_data(case.dataset)
        new_result = _run_new(data, case)
        golden = _load_golden(case)

        pd.testing.assert_frame_equal(
            new_result.reconstructed,
            golden,
            check_names=False,
            check_freq=False,
            atol=1e-7,
        )

    @pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
    def test_old_api_matches_golden(self, case: EquivalenceCase, update_golden):
        """Old API reconstructed result must match stored golden CSV."""
        if update_golden:
            pytest.skip("updating golden files")

        path = _golden_path(case)
        if not path.exists():
            pytest.skip(
                f"golden file missing: {path.relative_to(GOLDEN_DIR.parent.parent)}"
            )

        data = get_data(case.dataset)
        _, old_agg = _run_old(data, case)
        golden = _load_golden(case)

        pd.testing.assert_frame_equal(
            old_agg.predictOriginalData(),
            golden,
            check_names=False,
            check_freq=False,
            atol=1e-7,
        )
