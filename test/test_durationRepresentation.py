import os
import time

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam


def test_durationRepresentation():

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )

    starttime = time.time()

    # Silence warning on machines that cannot detect their physical cpu cores
    os.environ["OMP_NUM_THREADS"] = "1"

    aggregation1 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        sortValues=False,
        clusterMethod="k_means",
        rescaleClusterPeriods=False,
    )

    predictedPeriods1 = aggregation1.predictOriginalData()

    print("Clustering took " + str(time.time() - starttime))

    starttime = time.time()

    aggregation2 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        sortValues=False,
        clusterMethod="k_means",
        representationMethod="durationRepresentation",
        rescaleClusterPeriods=False,
    )

    predictedPeriods2 = aggregation2.predictOriginalData()

    print("Clustering took " + str(time.time() - starttime))

    starttime = time.time()

    aggregation3 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        sortValues=False,
        clusterMethod="k_means",
        representationMethod="durationRepresentation",
        distributionPeriodWise=False,
        rescaleClusterPeriods=False,
    )

    predictedPeriods3 = aggregation2.predictOriginalData()

    print("Clustering took " + str(time.time() - starttime))

    # make sure that the sum of the attribute specific RMSEs is smaller for the k-means clustering with centroid
    # representation than for the duration curve representation
    np.testing.assert_array_less(
        aggregation1.accuracyIndicators().loc[:, "RMSE"].sum(),
        aggregation3.accuracyIndicators().loc[:, "RMSE"].sum(),
        aggregation2.accuracyIndicators().loc[:, "RMSE"].sum(),
    )

    # make sure that the sum of the attribute specific duration curve RMSEs is smaller for the k-means clustering with
    # duration curve representation than for the centroid representation
    np.testing.assert_array_less(
        aggregation3.accuracyIndicators().loc[:, "RMSE_duration"].sum(),
        aggregation2.accuracyIndicators().loc[:, "RMSE_duration"].sum(),
        aggregation1.accuracyIndicators().loc[:, "RMSE_duration"].sum(),
    )


def test_durationRepresentation_hierarchical():

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )
    test_data_path = os.path.join(
        os.path.dirname(__file__), "data"
    )
    
    starttime = time.time()

    # Silence warning on machines that cannot detect their physical cpu cores
    os.environ["OMP_NUM_THREADS"] = "1"

    aggregation1 = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        sortValues=False,
        clusterMethod="hierarchical",
        representationMethod="durationRepresentation",
        rescaleClusterPeriods=False,
    )

    predictedPeriods1 = aggregation1.predictOriginalData()
    print("Clustering took " + str(time.time() - starttime))
   
    # compare against predicted test data
    testDataPredictedPeriods1 = pd.read_csv(
        os.path.join(test_data_path, "testData_durationRepresentation_hierarchical.csv"),
        index_col=0,
    ) # test data based on results from v2.3.7
    np.testing.assert_allclose(
        predictedPeriods1.values,
        testDataPredictedPeriods1.values,
    )
    

def test_distributionMinMaxRepresentation():

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=24,
        segmentation=True,
        noSegments=8,
        hoursPerPeriod=24,
        sortValues=False,
        clusterMethod="hierarchical",
        representationMethod="distributionAndMinMaxRepresentation",
        distributionPeriodWise=False,
        rescaleClusterPeriods=False,
    )

    predictedPeriods = aggregation.predictOriginalData()

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

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        segmentation=True,
        noSegments=8,
        sortValues=False,
        clusterMethod="hierarchical",
        representationMethod="distributionRepresentation",
        distributionPeriodWise=False,
        rescaleClusterPeriods=False,  # even without rescaling
    )

    predictedPeriods = aggregation.predictOriginalData()

    assert np.isclose(raw.mean(), predictedPeriods.mean(), atol=1e-4).all()


if __name__ == "__main__":
    test_durationRepresentation()
    test_distributionMinMaxRepresentation()
