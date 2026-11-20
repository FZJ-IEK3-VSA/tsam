"""A cluster that loses periods to an extreme cluster is represented by the
periods that stay in it.

``ExtremeConfig(method="append")`` and ``method="new_cluster"`` move periods
into extreme clusters after every cluster was represented. The representative
of a cluster they left must not keep describing them: under ``"mean"`` an
extreme period would otherwise count twice, once in its own cluster and once
inside the mean of the cluster it left.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import euclidean_distances

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, Distribution, ExtremeConfig, MinMaxMean, aggregate

MOVING_METHODS = ["append", "new_cluster"]
T = 24


@pytest.fixture(scope="module")
def testdata():
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)


@pytest.fixture
def three_days():
    """Days 0 and 1 are both [1, 1]; day 2 holds the peak, [0, 4]."""
    index = pd.date_range("2020-01-01", periods=6, freq="12h")
    return pd.DataFrame({"load": [1.0, 1.0, 1.0, 1.0, 0.0, 4.0]}, index=index)


def _normalized_periods(data):
    """One row per period, min-max scaled per column like the pipeline does."""
    scaled = (data - data.min()) / (data.max() - data.min())
    return scaled.to_numpy().reshape(-1, T * data.shape[1])


def _regular_clusters(result):
    """(cluster id, member period indices) for every non-extreme cluster."""
    assignments = np.asarray(result.cluster_assignments)
    extreme = set(result.clustering.extreme_cluster_indices or ())
    assert extreme, "the configuration is meant to add an extreme cluster"
    return [
        (int(cluster), np.flatnonzero(assignments == cluster))
        for cluster in sorted(set(assignments) - extreme)
    ]


# --- the example from the issue ---------------------------------------------


@pytest.mark.parametrize("method", MOVING_METHODS)
@pytest.mark.parametrize("preserve_column_means", [False, True])
@pytest.mark.parametrize(
    "representation",
    [
        "mean",
        "distribution",
        Distribution(scope="global"),
        MinMaxMean(max_columns=["load"]),
    ],
    ids=["mean", "distribution", "distribution_global", "minmax_mean"],
)
def test_extreme_period_is_not_counted_twice(
    three_days, method, preserve_column_means, representation
):
    result = aggregate(
        three_days,
        n_clusters=1,
        period_duration="1D",
        cluster=ClusterConfig(representation=representation),
        extremes=ExtremeConfig(method=method, max_value=["load"]),
        preserve_column_means=preserve_column_means,
    )

    representatives = result.cluster_representatives["load"].unstack().to_numpy()
    np.testing.assert_allclose(representatives, [[1.0, 1.0], [0.0, 4.0]], atol=1e-12)
    assert dict(result.cluster_counts) == {0: 2.0, 1: 1.0}
    assert result.reconstructed["load"].sum() == pytest.approx(8.0)


# --- every way of representing a cluster ------------------------------------


@pytest.mark.parametrize("method", MOVING_METHODS)
def test_mean_is_the_mean_of_the_remaining_members(testdata, method):
    result = aggregate(
        testdata,
        n_clusters=8,
        period_duration=T,
        cluster=ClusterConfig(method="hierarchical", representation="mean"),
        extremes=ExtremeConfig(method=method, max_value=["Load"]),
        preserve_column_means=False,
    )

    for cluster, members in _regular_clusters(result):
        representative = result.cluster_representatives.loc[cluster].sort_index()
        for column in testdata.columns:
            periods = testdata[column].to_numpy().reshape(-1, T)
            np.testing.assert_allclose(
                representative[column].to_numpy(),
                periods[members].mean(axis=0),
                rtol=1e-9,
                atol=1e-9,
                err_msg=f"cluster {cluster}, column {column}",
            )


def test_append_with_mean_preserves_column_sums_without_rescaling(testdata):
    """Each period is represented once: the extreme period by itself, every
    other period by the mean of its cluster."""
    result = aggregate(
        testdata,
        n_clusters=8,
        period_duration=T,
        cluster=ClusterConfig(method="hierarchical", representation="mean"),
        extremes=ExtremeConfig(method="append", max_value=["Load"]),
        preserve_column_means=False,
    )

    np.testing.assert_allclose(
        result.reconstructed.sum().to_numpy(), testdata.sum().to_numpy(), rtol=1e-12
    )


@pytest.mark.parametrize("method", MOVING_METHODS)
def test_medoid_is_the_medoid_of_the_remaining_members(testdata, method):
    result = aggregate(
        testdata,
        n_clusters=8,
        period_duration=T,
        cluster=ClusterConfig(method="hierarchical", representation="medoid"),
        extremes=ExtremeConfig(method=method, max_value=["Load"]),
    )
    periods = _normalized_periods(testdata)
    centers = result.clustering.cluster_centers

    for cluster, members in _regular_clusters(result):
        summed = euclidean_distances(periods[members]).sum(axis=0)
        medoid = members[int(np.argmin(np.round(summed, 10)))]
        assert centers[cluster] == medoid, f"cluster {cluster}"


@pytest.mark.parametrize("method", MOVING_METHODS)
def test_duration_curve_representative_follows_the_remaining_members(testdata, method):
    """With ``use_duration_curves`` a cluster is represented by the member
    closest to the centroid of its sorted profiles."""
    result = aggregate(
        testdata,
        n_clusters=8,
        period_duration=T,
        cluster=ClusterConfig(method="hierarchical", use_duration_curves=True),
        extremes=ExtremeConfig(method=method, max_value=["Load"]),
    )
    n_columns = testdata.shape[1]
    periods = _normalized_periods(testdata).reshape(-1, T, n_columns)
    sorted_periods = np.sort(periods, axis=1).reshape(len(periods), -1)
    centers = result.clustering.cluster_centers

    for cluster, members in _regular_clusters(result):
        centroid = sorted_periods[members].mean(axis=0)
        distance = np.square(sorted_periods[members] - centroid).sum(axis=1)
        closest = members[int(np.argmin(np.round(distance, 10)))]
        assert centers[cluster] == closest, f"cluster {cluster}"


@pytest.mark.parametrize("method", MOVING_METHODS)
def test_maxoid_is_scored_against_the_whole_dataset(testdata, method):
    """The maxoid is the member farthest from *all* periods, the extreme ones
    included, so a member leaving the cluster does not change it."""
    result = aggregate(
        testdata,
        n_clusters=8,
        period_duration=T,
        cluster=ClusterConfig(method="hierarchical", representation="maxoid"),
        extremes=ExtremeConfig(method=method, max_value=["Load"]),
    )
    periods = _normalized_periods(testdata)
    centers = result.clustering.cluster_centers

    for cluster, members in _regular_clusters(result):
        summed = euclidean_distances(periods, periods[members]).sum(axis=0)
        maxoid = members[int(np.argmax(np.round(summed, 10)))]
        assert centers[cluster] == maxoid, f"cluster {cluster}"


def test_cluster_that_loses_every_member_is_dropped():
    """Days 2 and 3 form one cluster; each holds the maximum of one column."""
    index = pd.date_range("2020-01-01", periods=8, freq="12h")
    data = pd.DataFrame(
        {
            "a": [1.0, 1.0, 1.1, 1.0, 9.0, 8.0, 8.0, 8.0],
            "b": [1.0, 1.0, 1.0, 1.1, 8.0, 8.0, 8.0, 9.0],
        },
        index=index,
    )

    result = aggregate(
        data,
        n_clusters=2,
        period_duration="1D",
        cluster=ClusterConfig(method="hierarchical", representation="mean"),
        extremes=ExtremeConfig(method="append", max_value=["a", "b"]),
        preserve_column_means=False,
    )

    assert [int(c) for c in result.cluster_assignments] == [0, 0, 1, 2]
    assert result.clustering.extreme_cluster_indices == (1, 2)
    np.testing.assert_allclose(
        result.cluster_representatives.loc[0].sort_index()["a"].to_numpy(),
        [1.05, 1.0],
    )
    np.testing.assert_allclose(
        result.reconstructed.sum().to_numpy(), data.sum().to_numpy()
    )


# --- transfer ---------------------------------------------------------------


def _replay(result, data):
    """Replay a clustering; return the result and whether it warned."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replayed = result.clustering.apply(data)
    warned = any("cannot be replayed exactly" in str(w.message) for w in caught)
    return replayed, warned


@pytest.mark.parametrize(
    "representation",
    ["mean", "distribution", MinMaxMean(max_columns=["Load"])],
    ids=["mean", "distribution", "minmax_mean"],
)
def test_append_with_a_computed_representation_replays_exactly(
    testdata, representation
):
    result = aggregate(
        testdata,
        n_clusters=8,
        period_duration=T,
        cluster=ClusterConfig(method="hierarchical", representation=representation),
        extremes=ExtremeConfig(method="append", max_value=["Load"]),
    )

    replayed, warned = _replay(result, testdata)

    assert not warned
    pd.testing.assert_frame_equal(
        replayed.cluster_representatives, result.cluster_representatives, atol=1e-10
    )


def test_new_cluster_with_mean_replays_all_but_the_extreme_cluster(testdata):
    """The extreme cluster holds reassigned periods; a transfer computes it
    from all of them, the aggregation represents it by the extreme period."""
    result = aggregate(
        testdata,
        n_clusters=8,
        period_duration=T,
        cluster=ClusterConfig(method="hierarchical", representation="mean"),
        extremes=ExtremeConfig(method="new_cluster", max_value=["Load"]),
        preserve_column_means=False,
    )
    extreme = list(result.clustering.extreme_cluster_indices)
    assert max(result.cluster_counts[cluster] for cluster in extreme) > 1

    replayed, warned = _replay(result, testdata)

    assert warned
    pd.testing.assert_frame_equal(
        replayed.cluster_representatives.drop(extreme, level=0),
        result.cluster_representatives.drop(extreme, level=0),
        atol=1e-10,
    )


@pytest.mark.parametrize("method", MOVING_METHODS)
def test_global_distribution_with_extremes_warns_on_transfer(testdata, method):
    result = aggregate(
        testdata,
        n_clusters=8,
        period_duration=T,
        cluster=ClusterConfig(
            method="hierarchical", representation=Distribution(scope="global")
        ),
        extremes=ExtremeConfig(method=method, max_value=["Load"]),
    )

    _, warned = _replay(result, testdata)

    assert warned
