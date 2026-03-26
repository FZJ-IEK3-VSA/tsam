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

import warnings
from contextlib import contextmanager

import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from _configs import GOLDEN_DIR, case_ids, get_data
from _old_new_equivalence import (
    CASES,
    EquivalenceCase,
    _run_new,
    _run_old,
)

pytestmark = pytest.mark.filterwarnings(
    "ignore:KMeans is known to have a memory leak on Windows with MKL.*:UserWarning"
)

# Cases where specific warnings are expected and should be suppressed.
_EXPECT_CONVERGENCE = {"kmeans/constant"}
_EXPECT_MAXVAL_WARNING = {"kmaxoids/wide", "hierarchical_distribution_minmax/wide"}
_EXPECT_WINDOWS_KMEANS_WARNING = {
    "kmeans/constant",
    "kmeans/wide",
    "kmeans_segmentation/testdata",
}


@contextmanager
def _expected_warnings(case: EquivalenceCase):
    """Suppress warnings that are expected for specific config/dataset combos."""
    with warnings.catch_warnings():
        if case.id in _EXPECT_CONVERGENCE:
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
        if case.id in _EXPECT_MAXVAL_WARNING:
            warnings.filterwarnings("ignore", "At least one maximal value")
        yield


@contextmanager
def _expected_windows_kmeans_warnings(case: EquivalenceCase):
    """Suppress known Windows-specific OpenMP/KMeans warnings for selected cases."""
    with warnings.catch_warnings():
        if case.id in _EXPECT_WINDOWS_KMEANS_WARNING:
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                module="threadpoolctl",
            )
            warnings.filterwarnings(
                "ignore",
                message="KMeans is known to have a memory leak on Windows with MKL.*",
                category=UserWarning,
            )
        yield


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
        """Save reconstructed results as golden files (only with --update-golden).

        For configs with skip_equivalence (intentional old/new divergence),
        golden is generated from the new API.  Otherwise from the old API.
        """
        if not update_golden:
            pytest.skip("use --update-golden to regenerate")

        data = get_data(case.dataset, max_timesteps=case.max_timesteps)
        if case.skip_equivalence:
            new_result = _run_new(data, case)
            _save_golden(new_result.reconstructed, case)
        else:
            with _expected_warnings(case):
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

        data = get_data(case.dataset, max_timesteps=case.max_timesteps)
        with _expected_warnings(case):
            with _expected_windows_kmeans_warnings(case):
                new_result = _run_new(data, case)
        golden = _load_golden(case)

        pd.testing.assert_frame_equal(
            new_result.reconstructed,
            golden,
            check_names=False,
            check_freq=False,
            check_like=True,
            atol=1e-7,
        )

    @pytest.mark.parametrize("case", CASES, ids=case_ids(CASES))
    def test_old_api_matches_golden(self, case: EquivalenceCase, update_golden):
        """Old API reconstructed result must match stored golden CSV."""
        if update_golden:
            pytest.skip("updating golden files")
        if case.skip_equivalence:
            pytest.skip("golden generated from new API (intentional divergence)")

        path = _golden_path(case)
        if not path.exists():
            pytest.skip(
                f"golden file missing: {path.relative_to(GOLDEN_DIR.parent.parent)}"
            )

        data = get_data(case.dataset, max_timesteps=case.max_timesteps)
        with _expected_warnings(case):
            with _expected_windows_kmeans_warnings(case):
                _, old_agg = _run_old(data, case)
        golden = _load_golden(case)

        pd.testing.assert_frame_equal(
            old_agg.predictOriginalData(),
            golden,
            check_names=False,
            check_freq=False,
            atol=1e-7,
        )
