import time

import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV

pytestmark = pytest.mark.filterwarnings("ignore::tsam.exceptions.LegacyAPIWarning")


@pytest.mark.filterwarnings("ignore:Segmentation is turned off:UserWarning")
def test_properties():
    hours_per_period = 24

    no_segments = 8

    no_typical_periods = 8

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    starttime = time.time()

    aggregation1 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        segmentation=True,
        no_segments=no_segments,
    )

    print("Clustering took " + str(time.time() - starttime))

    np.testing.assert_array_almost_equal(
        aggregation1.step_idx, np.arange(no_segments), decimal=4
    )

    starttime = time.time()

    aggregation2 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
    )

    print("Clustering took " + str(time.time() - starttime))

    np.testing.assert_array_almost_equal(
        aggregation2.step_idx, np.arange(hours_per_period), decimal=4
    )

    starttime = time.time()

    aggregation3 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
    )

    print("Clustering took " + str(time.time() - starttime))

    np.testing.assert_array_almost_equal(
        aggregation3.cluster_period_idx, np.arange(no_typical_periods), decimal=4
    )

    starttime = time.time()

    aggregation4 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        segmentation=True,
        no_segments=no_segments,
    )

    print("Clustering took " + str(time.time() - starttime))

    appearances = np.unique(aggregation4.cluster_order, return_counts=True)[1].tolist()

    occurrenceDict = {i: j for i, j in enumerate(appearances)}

    # make sure that the cluster_period_no_occur equals the number of appearances in the cluster_order
    np.testing.assert_array_almost_equal(
        list(aggregation4.cluster_period_no_occur.values()),
        list(occurrenceDict.values()),
        decimal=4,
    )

    starttime = time.time()

    aggregation5 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        segmentation=True,
        no_segments=no_segments,
    )

    print("Clustering took " + str(time.time() - starttime))

    # make sure that the values of the cluster_period_dict equal those from the typicalPeriods-dataframe
    np.testing.assert_array_almost_equal(
        pd.DataFrame.from_dict(data=aggregation5.cluster_period_dict).values,
        aggregation5.create_typical_periods().values,
        decimal=4,
    )

    starttime = time.time()

    aggregation6 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
    )

    print("Clustering took " + str(time.time() - starttime))

    # make sure that the sum of all segment durations in each period equals the hours per period
    for i in range(no_typical_periods):
        print(i)
        print(pd.DataFrame.from_dict(aggregation6.segment_duration_dict).loc[(i,), :])
        # print(
        #     pd.DataFrame.from_dict(aggregation6.segment_duration_dict)
        #     .iloc[
        #         pd.DataFrame.from_dict(
        #             aggregation6.segment_duration_dict
        #         ).index.get_level_values(0)
        #     ]
        # )
        print("\n")
        np.testing.assert_array_almost_equal(
            pd.DataFrame.from_dict(aggregation6.segment_duration_dict)
            .loc[(i,), :]
            .sum()
            .iloc[0],
            hours_per_period,
            decimal=4,
        )
        print("")

    starttime = time.time()

    aggregation7 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        segmentation=True,
        no_segments=no_segments,
    )

    print("Clustering took " + str(time.time() - starttime))

    # make sure that the sum of all segment durations in each period equals the hours per period
    for i in range(no_typical_periods):
        np.testing.assert_array_almost_equal(
            pd.DataFrame.from_dict(aggregation7.segment_duration_dict)
            .loc[i]
            .sum()
            .iloc[0],
            hours_per_period,
            decimal=4,
        )

    starttime = time.time()

    aggregation8 = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=no_typical_periods,
        hours_per_period=hours_per_period,
        cluster_method="hierarchical",
        segmentation=True,
        no_segments=no_segments,
    )

    print("Clustering took " + str(time.time() - starttime))

    indexTable = aggregation8.index_matching()

    # make sure that the PeriodNum column equals the cluster_order
    np.testing.assert_array_almost_equal(
        indexTable.loc[::24, "PeriodNum"].values, aggregation8.cluster_order, decimal=4
    )

    # make sure that the TimeStep indices equal the number of hours_per_period arranged as array
    np.testing.assert_array_almost_equal(
        pd.unique(indexTable.loc[:, "TimeStep"]),
        np.arange(hours_per_period, dtype="int64"),
        decimal=4,
    )

    # make sure that the SegmentIndex indices equal the number of no_segments arranged as array
    np.testing.assert_array_almost_equal(
        pd.unique(indexTable.loc[:, "SegmentIndex"]),
        np.arange(no_segments, dtype="int64"),
        decimal=4,
    )


if __name__ == "__main__":
    test_properties()
