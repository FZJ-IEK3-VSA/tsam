import time

import numpy as np
import pandas as pd

from tsam import ClusterConfig, SegmentConfig, aggregate


def test_subhourly_periods():
    # Create linearly growing test data for two hours.
    testData = pd.DataFrame(np.arange(1, 121), columns=["testdata"])
    testData.index = pd.date_range(
        "2050-01-01 00:30:00", periods=120, freq=("1Min"), tz="Europe/Berlin"
    )

    # Aggregate every quarter hour to one time step with mean representation.
    starttime = time.time()

    aggregation = aggregate(
        testData,
        n_clusters=8,
        period_duration=0.25,
        cluster=ClusterConfig(method="hierarchical"),
        segments=SegmentConfig(n_segments=1),
    )

    results = aggregation.reconstructed

    print("Clustering took " + str(time.time() - starttime))

    # Compare it to the expected result of eight linearly growing mean values.
    np.testing.assert_array_almost_equal(
        np.unique(results.values), np.arange(8, 114, 15), decimal=4
    )


if __name__ == "__main__":
    test_subhourly_periods()
