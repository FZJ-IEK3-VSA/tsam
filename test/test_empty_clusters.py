"""Empty clusters must never leave the pipeline.

``method="new_cluster"`` reassigns every period that sits closer to an extreme
profile than to its own center, and nothing stops it from taking all of them.
The cluster it empties used to survive with a representative standing for no
periods, which put a hole in the label space (issues #478 and #480).
"""

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, ExtremeConfig, aggregate
from tsam.algorithms.clustering import assign_clusters
from tsam.algorithms.representations import representations

# The combination that empties a regular cluster on this dataset: the
# distribution_minmax representatives are extreme enough for the peak-Load day
# to absorb every member of one cluster.
EMPTYING_CASE = {
    "n_clusters": 8,
    "period_duration": 24,
    "cluster": ClusterConfig(
        method="hierarchical", representation="distribution_minmax"
    ),
    "extremes": ExtremeConfig(method="new_cluster", max_value=["Load"]),
}


@pytest.fixture
def raw():
    return pd.read_csv(TESTDATA_CSV, index_col=0)


def test_every_cluster_represents_at_least_one_period(raw):
    result = aggregate(raw, **EMPTYING_CASE)

    representative_ids = set(
        result.cluster_representatives.index.get_level_values(0).unique().tolist()
    )
    used_labels = {int(label) for label in result.cluster_assignments}

    assert representative_ids == used_labels
    assert representative_ids == set(range(len(representative_ids)))
    assert result.n_clusters == result.clustering.n_clusters


def test_clustering_that_dropped_a_cluster_still_transfers(raw):
    for preserve_column_means in (True, False):
        result = aggregate(
            raw, preserve_column_means=preserve_column_means, **EMPTYING_CASE
        )
        transferred = result.clustering.apply(raw)

        assert transferred.n_clusters == result.n_clusters
        np.testing.assert_array_equal(
            transferred.cluster_assignments, result.cluster_assignments
        )


def test_representations_rejects_a_label_space_with_holes():
    candidates = np.arange(6 * 4, dtype=float).reshape(6, 4)
    order = np.array([0, 0, 1, 1, 3, 3])  # label 2 unused, 3 is the highest

    with pytest.raises(ValueError, match="every label from 0 to 3"):
        representations(
            candidates,
            order,
            default="mean",
            representation_method="mean",
            representation_dict=None,
            n_timesteps_per_period=4,
        )


# --- clustering alone, no extremes configured -----------------------------
#
# Twenty days over three distinct day shapes. No clustering method can fill
# eight clusters from three shapes, so some are left empty — with no
# ExtremeConfig anywhere in sight.
SHAPE_STARVED_CASE = {
    "n_clusters": 8,
    "cluster": ClusterConfig(method="kmaxoids", representation="mean"),
}


@pytest.fixture
def three_shapes():
    index = pd.date_range("2020-01-01", periods=24 * 20, freq="h")
    values = np.concatenate([[1.0, 5.0, 9.0][i % 3] * np.ones(24) for i in range(20)])
    return pd.DataFrame({"x": values, "y": values * 2}, index=index)


def test_clustering_alone_cannot_leave_a_hole(three_shapes):
    np.random.seed(0)  # kmaxoids draws from numpy.random
    with pytest.warns(UserWarning, match="non-empty clusters"):
        result = aggregate(three_shapes, **SHAPE_STARVED_CASE)

    used_labels = {int(label) for label in result.cluster_assignments}

    assert used_labels == set(range(result.n_clusters))
    assert result.n_clusters == result.clustering.n_clusters
    assert result.n_clusters < SHAPE_STARVED_CASE["n_clusters"]
    # Every period is still accounted for by exactly one cluster.
    assert sum(result.cluster_counts.values()) == pytest.approx(20)


@pytest.mark.parametrize("preserve_column_means", [True, False])
def test_shape_starved_clustering_still_transfers(three_shapes, preserve_column_means):
    np.random.seed(0)
    with pytest.warns(UserWarning, match="non-empty clusters"):
        result = aggregate(
            three_shapes,
            preserve_column_means=preserve_column_means,
            **SHAPE_STARVED_CASE,
        )

    transferred = result.clustering.apply(three_shapes)

    assert transferred.n_clusters == result.n_clusters
    np.testing.assert_array_equal(
        transferred.cluster_assignments, result.cluster_assignments
    )


def test_assign_clusters_closes_gaps_in_the_label_space():
    # Three distinct rows, eight clusters asked for: kmaxoids cannot fill them.
    candidates = np.repeat([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]], 4, axis=0)

    np.random.seed(0)
    order = assign_clusters(candidates, n_clusters=8, cluster_method="kmaxoids")

    labels = np.unique(order)
    np.testing.assert_array_equal(labels, np.arange(labels.size))
    assert labels.size < 8


def test_assign_clusters_leaves_a_dense_label_space_alone():
    candidates = np.repeat([[0.0, 0.0], [5.0, 5.0], [9.0, 9.0]], 4, axis=0)

    order = assign_clusters(candidates, n_clusters=3, cluster_method="hierarchical")

    np.testing.assert_array_equal(np.unique(order), np.arange(3))
