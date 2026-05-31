import os
import time

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import (
    ClusterConfig,
    Distribution,
    SegmentConfig,
    aggregate,
)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:KMeans is known to have a memory leak on Windows with MKL.*:UserWarning"
    ),
]


def test_durationRepresentation():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    starttime = time.time()

    # Silence warning on machines that cannot detect their physical cpu cores
    os.environ["OMP_NUM_THREADS"] = "1"

    # Set seed for deterministic k-means results
    np.random.seed(42)

    aggregation1 = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(method="kmeans", use_duration_curves=False),
        preserve_column_means=False,
    )

    print("Clustering took " + str(time.time() - starttime))

    starttime = time.time()

    aggregation2 = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(
            method="kmeans",
            use_duration_curves=False,
            representation="distribution",
        ),
        preserve_column_means=False,
    )

    print("Clustering took " + str(time.time() - starttime))

    starttime = time.time()

    aggregation3 = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(
            method="kmeans",
            use_duration_curves=False,
            representation=Distribution(scope="global"),
        ),
        preserve_column_means=False,
    )

    print("Clustering took " + str(time.time() - starttime))

    # make sure that the sum of the attribute specific RMSEs is smaller for the k-means clustering with centroid
    # representation than for the duration curve representation
    np.testing.assert_array_less(
        aggregation1.accuracy.rmse.sum(),
        aggregation3.accuracy.rmse.sum(),
        aggregation2.accuracy.rmse.sum(),
    )

    # make sure that the sum of the attribute specific duration curve RMSEs is smaller for the k-means clustering with
    # duration curve representation than for the centroid representation
    np.testing.assert_array_less(
        aggregation3.accuracy.rmse_duration.sum(),
        aggregation2.accuracy.rmse_duration.sum(),
        aggregation1.accuracy.rmse_duration.sum(),
    )


@pytest.mark.filterwarnings("ignore:The cluster is too small:UserWarning")
def test_distributionMinMaxRepresentation():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation = aggregate(
        raw,
        n_clusters=24,
        period_duration=24,
        segments=SegmentConfig(
            n_segments=8,
            representation=Distribution(scope="global", preserve_minmax=True),
        ),
        cluster=ClusterConfig(
            method="hierarchical",
            use_duration_curves=False,
            representation=Distribution(scope="global", preserve_minmax=True),
        ),
        preserve_column_means=False,
    )

    predictedPeriods = aggregation.reconstructed

    # make sure that max and min of the newly predicted time series are the same as
    #  from the original
    np.testing.assert_array_equal(
        raw.max(),
        predictedPeriods.max(),
    )
    np.testing.assert_array_equal(
        raw.min(),
        predictedPeriods.min(),
    )

    assert np.isclose(raw.mean(), predictedPeriods.mean(), atol=1e-4).all()


def test_distributionRepresentation_keeps_mean():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        segments=SegmentConfig(
            n_segments=8,
            representation=Distribution(scope="global"),
        ),
        cluster=ClusterConfig(
            method="hierarchical",
            use_duration_curves=False,
            representation=Distribution(scope="global"),
        ),
        preserve_column_means=False,  # even without rescaling
    )

    predictedPeriods = aggregation.reconstructed

    assert np.isclose(raw.mean(), predictedPeriods.mean(), atol=1e-4).all()


if __name__ == "__main__":
    test_durationRepresentation()
    test_distributionMinMaxRepresentation()
