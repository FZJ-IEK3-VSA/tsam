import time

import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, SegmentConfig, aggregate


def test_properties():
    period_duration = 24

    n_segments = 8

    n_clusters = 8

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    starttime = time.time()

    result1 = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(method="hierarchical"),
        segments=SegmentConfig(n_segments=n_segments),
    )

    print("Clustering took " + str(time.time() - starttime))

    # with segmentation the timestep index runs over the segments
    np.testing.assert_array_almost_equal(
        result1.timestep_index, np.arange(n_segments), decimal=4
    )

    result2 = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(method="hierarchical"),
    )

    # without segmentation the timestep index runs over the timesteps per period
    np.testing.assert_array_almost_equal(
        result2.timestep_index, np.arange(period_duration), decimal=4
    )

    result3 = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(method="hierarchical"),
    )

    np.testing.assert_array_almost_equal(
        result3.period_index, np.arange(n_clusters), decimal=4
    )

    result4 = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(method="hierarchical"),
        segments=SegmentConfig(n_segments=n_segments),
    )

    appearances = np.unique(result4.cluster_assignments, return_counts=True)[1].tolist()

    occurrenceDict = {i: j for i, j in enumerate(appearances)}

    # make sure that the cluster_weights equal the number of appearances in the cluster_assignments
    np.testing.assert_array_almost_equal(
        [result4.cluster_weights[k] for k in sorted(result4.cluster_weights)],
        list(occurrenceDict.values()),
        decimal=4,
    )

    result5 = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(method="hierarchical"),
    )

    # without segmentation there are no segment durations
    assert result5.segment_durations is None

    result6 = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(method="hierarchical"),
        segments=SegmentConfig(n_segments=n_segments),
    )

    # make sure that the sum of all segment durations in each period equals the timesteps per period
    for i in range(n_clusters):
        np.testing.assert_array_almost_equal(
            sum(result6.segment_durations[i]), period_duration, decimal=4
        )

    result7 = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(method="hierarchical"),
        segments=SegmentConfig(n_segments=n_segments),
    )

    indexTable = result7.assignments

    # make sure that the cluster_idx column (per period) equals the cluster_assignments
    np.testing.assert_array_almost_equal(
        indexTable.iloc[::24]["cluster_idx"].values,
        result7.cluster_assignments,
        decimal=4,
    )

    # make sure that the timestep indices equal the number of timesteps per period arranged as array
    np.testing.assert_array_almost_equal(
        pd.unique(indexTable.loc[:, "timestep_idx"]),
        np.arange(period_duration, dtype="int64"),
        decimal=4,
    )

    # make sure that the segment indices equal the number of segments arranged as array
    np.testing.assert_array_almost_equal(
        np.sort(pd.unique(indexTable.loc[:, "segment_idx"])),
        np.arange(n_segments, dtype="int64"),
        decimal=4,
    )


if __name__ == "__main__":
    test_properties()
