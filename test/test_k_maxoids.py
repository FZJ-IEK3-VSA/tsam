import os
import time

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, aggregate

pytestmark = pytest.mark.filterwarnings(
    "ignore:KMeans is known to have a memory leak on Windows with MKL.*:UserWarning"
)


def test_k_maxoids():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    starttime = time.time()

    # Silence warning on machines that cannot detect their physical cpu cores
    os.environ["OMP_NUM_THREADS"] = "1"

    # Set seed for deterministic k-means results
    np.random.seed(42)

    result1 = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(method="kmeans"),
        preserve_column_means=False,
    )

    predictedPeriods1 = result1.reconstructed

    print("Clustering took " + str(time.time() - starttime))

    starttime = time.time()

    result2 = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(method="kmaxoids"),
        preserve_column_means=False,
    )

    predictedPeriods2 = result2.reconstructed

    print("Clustering took " + str(time.time() - starttime))

    # make sure that the maximum values of the time series predicted by k-maxoids are bigger than those predicted by
    # k-means
    np.testing.assert_array_less(predictedPeriods1.max(), predictedPeriods2.max())

    # make sure that the minimum values of the time series predicted by k-maxoids are smaller than those predicted by
    # k-means except for those of GHI since the minimum value of 0 during night time is found by both algorithms
    np.testing.assert_array_less(
        predictedPeriods2.min()[1:], predictedPeriods1.min()[1:]
    )


if __name__ == "__main__":
    test_k_maxoids()
