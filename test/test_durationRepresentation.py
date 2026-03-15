import os
import time

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_durationRepresentation():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    starttime = time.time()

    # Silence warning on machines that cannot detect their physical cpu cores
    os.environ["OMP_NUM_THREADS"] = "1"

    # Set seed for deterministic k-means results
    np.random.seed(42)

    aggregation1 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=8,
        hours_per_period=24,
        sort_values=False,
        cluster_method="k_means",
        rescale_cluster_periods=False,
    )

    predictedPeriods1 = aggregation1.predict_original_data()

    print("Clustering took " + str(time.time() - starttime))

    starttime = time.time()

    aggregation2 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=8,
        hours_per_period=24,
        sort_values=False,
        cluster_method="k_means",
        representation_method="durationRepresentation",
        rescale_cluster_periods=False,
    )

    predictedPeriods2 = aggregation2.predict_original_data()

    print("Clustering took " + str(time.time() - starttime))

    starttime = time.time()

    aggregation3 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=8,
        hours_per_period=24,
        sort_values=False,
        cluster_method="k_means",
        representation_method="durationRepresentation",
        distribution_period_wise=False,
        rescale_cluster_periods=False,
    )

    predictedPeriods3 = aggregation3.predict_original_data()

    print("Clustering took " + str(time.time() - starttime))

    # make sure that the sum of the attribute specific RMSEs is smaller for the k-means clustering with centroid
    # representation than for the duration curve representation
    np.testing.assert_array_less(
        aggregation1.accuracy_indicators().loc[:, "RMSE"].sum(),
        aggregation3.accuracy_indicators().loc[:, "RMSE"].sum(),
        aggregation2.accuracy_indicators().loc[:, "RMSE"].sum(),
    )

    # make sure that the sum of the attribute specific duration curve RMSEs is smaller for the k-means clustering with
    # duration curve representation than for the centroid representation
    np.testing.assert_array_less(
        aggregation3.accuracy_indicators().loc[:, "RMSE_duration"].sum(),
        aggregation2.accuracy_indicators().loc[:, "RMSE_duration"].sum(),
        aggregation1.accuracy_indicators().loc[:, "RMSE_duration"].sum(),
    )


def test_distributionMinMaxRepresentation():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=24,
        segmentation=True,
        no_segments=8,
        hours_per_period=24,
        sort_values=False,
        cluster_method="hierarchical",
        representation_method="distributionAndMinMaxRepresentation",
        distribution_period_wise=False,
        rescale_cluster_periods=False,
    )

    predictedPeriods = aggregation.predict_original_data()

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

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=8,
        hours_per_period=24,
        segmentation=True,
        no_segments=8,
        sort_values=False,
        cluster_method="hierarchical",
        representation_method="distributionRepresentation",
        distribution_period_wise=False,
        rescale_cluster_periods=False,  # even without rescaling
    )

    predictedPeriods = aggregation.predict_original_data()

    assert np.isclose(raw.mean(), predictedPeriods.mean(), atol=1e-4).all()


if __name__ == "__main__":
    test_durationRepresentation()
    test_distributionMinMaxRepresentation()
