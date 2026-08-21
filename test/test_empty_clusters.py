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
