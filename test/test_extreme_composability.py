"""Tests for composable extreme-period handling (issue #410).

``preserve_n_clusters`` makes the number of typical periods a pure function of the
config (exactly ``n_clusters``), so results from independent datasets align.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, ExtremeConfig, aggregate

N_CLUSTERS = 8
PERIOD_HOURS = 24
# One column's peak and another's trough: two extremes, two distinct periods.
DEFAULT_CRITERIA = {"max_value": ["GHI"], "min_value": ["Load"]}


@pytest.fixture(scope="module")
def real_data():
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)


def n_typical_periods(result):
    return len(result.cluster_representatives.index.get_level_values(0).unique())


def assert_same_representatives(expected, actual):
    pd.testing.assert_frame_equal(
        expected.cluster_representatives,
        actual.cluster_representatives,
        check_exact=False,
        atol=1e-10,
    )


def carve(data, *, method="append", criteria=None, **aggregate_kwargs):
    """Aggregate with the extremes carved out of the cluster budget."""
    return aggregate(
        data,
        n_clusters=N_CLUSTERS,
        period_duration=PERIOD_HOURS,
        extremes=ExtremeConfig(
            method=method,
            preserve_n_clusters=True,
            **(criteria or DEFAULT_CRITERIA),
        ),
        **aggregate_kwargs,
    )


# Each case varies one thing that could plausibly break the carve — the
# integration method, the clustering backend, the space distances are measured
# in, or how many criteria compete for how few periods.
CARVE_CASES: dict[str, dict] = {
    "append": {},
    "new_cluster": {"method": "new_cluster"},
    "hierarchical": {"cluster": ClusterConfig(method="hierarchical")},
    "kmeans": {"cluster": ClusterConfig(method="kmeans")},
    "kmaxoids": {"cluster": ClusterConfig(method="kmaxoids")},
    "averaging": {"cluster": ClusterConfig(method="averaging")},
    "duration curves": {
        "cluster": ClusterConfig(method="hierarchical", use_duration_curves=True)
    },
    "duration curves, new_cluster": {
        "method": "new_cluster",
        "cluster": ClusterConfig(method="hierarchical", use_duration_curves=True),
    },
    "weighted columns": {"weights": {"GHI": 5.0, "Load": 1.0}},
    "column means preserved": {"preserve_column_means": True},
    "four criteria on one column": {
        "criteria": {
            "max_value": ["GHI"],
            "max_period": ["GHI"],
            "min_value": ["GHI"],
            "min_period": ["GHI"],
        }
    },
    # new_cluster lets ordinary periods migrate into the extreme clusters, which
    # is the way a regular cluster could be emptied and drop the total.
    "six criteria, absorbing": {
        "method": "new_cluster",
        "cluster": ClusterConfig(method="hierarchical"),
        "criteria": {
            "max_value": ["GHI", "Load", "Wind"],
            "min_value": ["GHI", "Load", "Wind"],
        },
    },
}


def spiked_series(*, extremes_share_a_period, seed, n_periods=20):
    """Flat noise with one peak and one trough, in one period or in two.

    Only that placement differs between the two variants, and it is enough to
    change how many extremes survive deduplication — which is what makes the
    default count follow the data.
    """
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.4, 0.6, n_periods * PERIOD_HOURS)
    values[3 * PERIOD_HOURS + 5] = 0.7  # the maximum, in period 3
    trough_period = 3 if extremes_share_a_period else 11
    values[trough_period * PERIOD_HOURS + 9] = 0.3
    index = pd.date_range("2020-01-01", periods=values.size, freq="h")
    return pd.DataFrame({"x": values}, index=index)


class TestFixedTotalCount:
    """preserve_n_clusters yields exactly n_clusters, data-independently."""

    @pytest.mark.parametrize("method", ["append", "new_cluster"])
    def test_replaces_a_data_dependent_count_with_a_fixed_one(self, method):
        """The premise of #410 and the fix, over one pair of datasets.

        The two variants differ only in whether the peak and the trough share a
        period. That alone decides how many extremes survive, so by default the
        same config returns a different number of typical periods on each —
        the thing that makes independent runs impossible to align.
        """
        datasets = [
            spiked_series(extremes_share_a_period=share, seed=seed)
            for seed in range(3)
            for share in (True, False)
        ]
        # Small budget on purpose: with many clusters over few periods the
        # extreme periods end up as centers anyway and the difference vanishes.
        n_clusters = 5

        def count(data, *, preserve_n_clusters):
            result = aggregate(
                data,
                n_clusters=n_clusters,
                period_duration=PERIOD_HOURS,
                extremes=ExtremeConfig(
                    method=method,
                    preserve_n_clusters=preserve_n_clusters,
                    max_value=["x"],
                    min_value=["x"],
                ),
            )
            return n_typical_periods(result)

        by_default = {count(data, preserve_n_clusters=False) for data in datasets}
        carved = {count(data, preserve_n_clusters=True) for data in datasets}

        assert len(by_default) > 1, "premise: the default count follows the data"
        assert max(by_default) > n_clusters, "premise: extremes inflate the count"
        assert carved == {n_clusters}

    @pytest.mark.parametrize("case", CARVE_CASES.values(), ids=list(CARVE_CASES))
    def test_count_is_exactly_n_clusters(self, real_data, case):
        result = carve(real_data, **case)
        assert n_typical_periods(result) == N_CLUSTERS
        assert result.clustering.n_clusters == N_CLUSTERS
        # A carved-out extreme must not leave a regular cluster with no members.
        assert all(count > 0 for count in result.cluster_counts.values())

    def test_extremes_are_preserved(self, real_data):
        """The carved extremes still inject the true peak/trough."""
        result = carve(real_data)
        reconstructed = result.reconstructed
        assert reconstructed["GHI"].max() == pytest.approx(real_data["GHI"].max())
        assert reconstructed["Load"].min() == pytest.approx(real_data["Load"].min())

    def test_column_means_still_preserved(self, real_data):
        """Excluding extremes from clustering must not disturb mean rescaling."""
        result = carve(real_data, preserve_column_means=True)
        reconstructed = result.reconstructed
        for column in real_data.columns:
            assert reconstructed[column].mean() == pytest.approx(
                real_data[column].mean(), rel=1e-6
            )

    def test_too_many_extremes_raises(self, real_data):
        criteria = {
            "max_value": ["GHI", "Load", "Wind"],
            "min_value": ["GHI", "Load", "Wind"],
        }
        with pytest.raises(ValueError, match="preserve_n_clusters requires n_clusters"):
            aggregate(
                real_data,
                n_clusters=3,
                period_duration=PERIOD_HOURS,
                extremes=ExtremeConfig(
                    method="append", preserve_n_clusters=True, **criteria
                ),
            )

    def test_replace_ignores_preserve_n_clusters_with_warning(self, real_data):
        with pytest.warns(UserWarning, match="preserve_n_clusters has no effect"):
            extremes = ExtremeConfig(
                method="replace", preserve_n_clusters=True, max_value=["GHI"]
            )
        result = aggregate(
            real_data,
            n_clusters=N_CLUSTERS,
            period_duration=PERIOD_HOURS,
            extremes=extremes,
        )
        assert n_typical_periods(result) == N_CLUSTERS

    @pytest.mark.parametrize("method", ["append", "new_cluster"])
    def test_extreme_that_is_also_a_natural_center_still_counts(self, method):
        """Adversarial: an extreme period that would be its own cluster anyway.

        Naively re-detecting extremes after clustering into (n_clusters - D)
        would find them "already represented" and skip them, yielding fewer than
        n_clusters. Because the preserve_n_clusters path excludes extremes up front and
        adds back exactly D, the total stays n_clusters and each extreme is its
        own distinct cluster.
        """
        rng = np.random.default_rng(0)
        n_normal = 18
        normal = 0.5 + 0.01 * rng.standard_normal((n_normal * PERIOD_HOURS, 1))
        peak = np.full((PERIOD_HOURS, 1), 5.0)
        trough = np.full((PERIOD_HOURS, 1), -5.0)
        values = np.vstack([normal, peak, trough])
        index = pd.date_range("2020-01-01", periods=values.shape[0], freq="h")
        data = pd.DataFrame(values, columns=["x"], index=index)

        result = aggregate(
            data,
            n_clusters=4,
            period_duration=PERIOD_HOURS,
            extremes=ExtremeConfig(
                method=method,
                preserve_n_clusters=True,
                max_value=["x"],
                min_value=["x"],
            ),
        )

        assert n_typical_periods(result) == 4
        reconstructed = result.reconstructed
        assert reconstructed["x"].max() == pytest.approx(5.0)
        assert reconstructed["x"].min() == pytest.approx(-5.0)
        assignments = result.clustering.cluster_assignments
        peak_cluster = assignments[n_normal]
        trough_cluster = assignments[n_normal + 1]
        assert peak_cluster != trough_cluster
        assert assignments.count(peak_cluster) == 1
        assert assignments.count(trough_cluster) == 1

    @pytest.mark.parametrize("method", ["append", "new_cluster"])
    def test_transfers(self, real_data, method):
        direct = carve(real_data, method=method)
        applied = direct.clustering.apply(real_data)
        assert n_typical_periods(applied) == N_CLUSTERS
        assert_same_representatives(direct, applied)


class TestExtremeConfigSerialization:
    def test_preserve_n_clusters_roundtrips(self):
        config = ExtremeConfig(
            method="append", preserve_n_clusters=True, max_value=["GHI"]
        )
        assert config.to_dict()["preserve_n_clusters"] is True
        assert ExtremeConfig.from_dict(config.to_dict()).preserve_n_clusters is True

    def test_preserve_n_clusters_omitted_when_false(self):
        config = ExtremeConfig(method="append", max_value=["GHI"])
        assert "preserve_n_clusters" not in config.to_dict()
        assert ExtremeConfig.from_dict(config.to_dict()).preserve_n_clusters is False


class TestFixedTotalFuzz:
    """Randomized check that the total is always exactly n_clusters.

    The guarantee is logical (extremes are carved out of the budget, regular
    representatives are pinned), so every random combination of data, method,
    criteria and budget must land on exactly n_clusters. Uses hierarchical
    clustering, which returns non-empty clusters by construction.
    """

    @pytest.mark.parametrize("seed", range(50))
    def test_count_is_always_exactly_n_clusters(self, seed):
        rng = np.random.default_rng(seed)
        columns = [f"c{i}" for i in range(int(rng.integers(2, 5)))]
        n_periods = int(rng.integers(15, 45))
        index = pd.date_range("2020-01-01", periods=n_periods * PERIOD_HOURS, freq="h")
        data = pd.DataFrame(
            {c: rng.random(n_periods * PERIOD_HOURS) for c in columns}, index=index
        )

        def pick_columns():
            k = int(rng.integers(0, len(columns) + 1))
            return list(rng.choice(columns, size=k, replace=False)) if k else []

        criteria = {
            "max_value": pick_columns(),
            "min_value": pick_columns(),
            "max_period": pick_columns(),
            "min_period": pick_columns(),
        }
        if not any(criteria.values()):
            criteria["max_value"] = [columns[0]]

        method = str(rng.choice(["append", "new_cluster"]))
        max_distinct = sum(len(v) for v in criteria.values())
        n_clusters = min(
            int(rng.integers(max_distinct + 1, max_distinct + 8)), n_periods
        )

        result = aggregate(
            data,
            n_clusters=n_clusters,
            period_duration=PERIOD_HOURS,
            cluster=ClusterConfig(method="hierarchical"),
            extremes=ExtremeConfig(method=method, preserve_n_clusters=True, **criteria),
        )
        assert n_typical_periods(result) == n_clusters
        assert result.clustering.n_clusters == n_clusters
        assert all(count > 0 for count in result.cluster_counts.values())
