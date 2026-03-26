import copy

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_assert_raises():
    # important: special signs such as brackets must be marked with '\' when matching error message

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # check error message for wrong time series
    np.testing.assert_raises_regex(
        ValueError,
        r"time_series has to be of type pandas.DataFrame\(\) or of type np.array\(\) in "
        "initialization of object of class TimeSeriesAggregation",
        tsam.TimeSeriesAggregation,
        time_series="erroneousTimeSeries",
    )

    # check error messages for wrong attribute names added for extreme period methods
    np.testing.assert_raises_regex(
        ValueError,
        'erroneousAttribute listed in "add_peak_min" does not occur as time_series column',
        tsam.TimeSeriesAggregation,
        time_series=raw,
        add_peak_min=["erroneousAttribute"],
    )
    np.testing.assert_raises_regex(
        ValueError,
        'erroneousAttribute listed in "add_peak_max" does not occur as time_series column',
        tsam.TimeSeriesAggregation,
        time_series=raw,
        add_peak_max=["erroneousAttribute"],
    )
    np.testing.assert_raises_regex(
        ValueError,
        'erroneousAttribute listed in "add_mean_min" does not occur as time_series column',
        tsam.TimeSeriesAggregation,
        time_series=raw,
        add_mean_min=["erroneousAttribute"],
    )
    np.testing.assert_raises_regex(
        ValueError,
        'erroneousAttribute listed in "add_mean_max" does not occur as time_series column',
        tsam.TimeSeriesAggregation,
        time_series=raw,
        add_mean_max=["erroneousAttribute"],
    )

    # check error message for missing datetime index and missing resolution argument
    np.testing.assert_raises_regex(
        ValueError,
        "'resolution' argument has to be nonnegative float or int or the given "
        "timeseries needs a datetime index",
        tsam.TimeSeriesAggregation,
        time_series=raw.reset_index(),
    )
    # overwrite one of the datetime-like string indices in the raw data to an index that cannot be converted
    rawErrInd = copy.deepcopy(raw)
    as_list = rawErrInd.index.tolist()
    idx = as_list.index("2010-01-01 00:30:00")
    as_list[idx] = "erroneousDatetimeIndex"
    rawErrInd.index = as_list
    # check error message for erroneous datetime index and missing resolution argument
    np.testing.assert_raises_regex(
        ValueError,
        "'resolution' argument has to be nonnegative float or int or the given "
        "timeseries needs a datetime index",
        tsam.TimeSeriesAggregation,
        time_series=rawErrInd,
    )
    # check erroneous resolution argument
    np.testing.assert_raises_regex(
        ValueError,
        "resolution has to be nonnegative float or int",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        resolution="erroneousResolution",
    )

    # check erroneous hours_per_period argument
    np.testing.assert_raises_regex(
        ValueError,
        "hours_per_period has to be nonnegative float or int",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        hours_per_period=None,
    )

    # check erroneous no_typical_periods argument
    np.testing.assert_raises_regex(
        ValueError,
        "no_typical_periods has to be nonnegative integer",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        no_typical_periods=None,
    )

    # check non-integer time step number per typical period
    np.testing.assert_raises_regex(
        ValueError,
        "The combination of hours_per_period and the "
        "resolution does not result in an integer "
        "number of time steps per period",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        hours_per_period=23,
        resolution=2,
    )

    # check warning when number of segments per period is higher than the number of time steps per period
    np.testing.assert_warns(
        Warning,
        tsam.TimeSeriesAggregation,
        time_series=raw,
        segmentation=True,
        no_segments=25,
    )

    # check erroneous cluster_method argument
    np.testing.assert_raises_regex(
        ValueError,
        r"cluster_method needs to be one of the following: \['averaging', 'k_means', "
        r"'k_medoids', 'k_maxoids', 'hierarchical', 'adjacent_periods'\]",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        cluster_method="erroneousClusterMethod",
    )

    # check erroneous representation_method argument
    np.testing.assert_raises(
        ValueError,
        tsam.TimeSeriesAggregation,
        time_series=raw,
        representation_method="erroneousRepresentationMethod",
    )

    # check erroneous extreme_period_method argument
    np.testing.assert_raises_regex(
        ValueError,
        r"extreme_period_method needs to be one of the following: \['None', 'append', "
        r"'new_cluster_center', 'replace_cluster_center'\]",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        extreme_period_method="erroneousExtremePeriodMethod",
    )

    # check erroneous eval_sum_periods argument
    np.testing.assert_raises_regex(
        ValueError,
        "eval_sum_periods has to be boolean",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        eval_sum_periods="erroneousEvalSumPeriods",
    )

    # check erroneous sort_values argument
    np.testing.assert_raises_regex(
        ValueError,
        "sort_values has to be boolean",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        sort_values="erroneousSortValues",
    )

    # check erroneous same_mean argument
    np.testing.assert_raises_regex(
        ValueError,
        "same_mean has to be boolean",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        same_mean="erroneousSameMean",
    )

    # check erroneous rescale_cluster_periods argument
    np.testing.assert_raises_regex(
        ValueError,
        "rescale_cluster_periods has to be boolean",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        rescale_cluster_periods="erroneousrescaleClusterPeriods",
    )

    # check erroneous predef_cluster_order argument
    np.testing.assert_raises_regex(
        ValueError,
        "predef_cluster_order has to be an array or list",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        predef_cluster_order="erroneousPredefClusterOrder",
    )

    # get a cluster order from a preceding clustering run
    aggregation = tsam.TimeSeriesAggregation(time_series=raw)
    periodOrder = aggregation.cluster_order
    # check erroneous predef_cluster_center_indices argument
    np.testing.assert_raises_regex(
        ValueError,
        "predef_cluster_center_indices has to be an array or list",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        predef_cluster_order=periodOrder,
        predef_cluster_center_indices="erroneousPredefClusterCenterIndices",
    )

    # check error, when predef_cluster_center_indices are defined but not predef_cluster_order
    np.testing.assert_raises_regex(
        ValueError,
        'If "predef_cluster_center_indices" is defined, "predef_cluster_order" needs to be '
        "defined as well",
        tsam.TimeSeriesAggregation,
        time_series=raw,
        predef_cluster_center_indices="erroneousPredefClusterCenterIndices",
    )

    # check erroneous dataframe containing NaN values
    rawNan = copy.deepcopy(raw)
    rawNan.iloc[10, :] = np.nan
    aggregation = tsam.TimeSeriesAggregation(time_series=rawNan)
    np.testing.assert_raises_regex(
        ValueError,
        "Pre processed data includes NaN. Please check the time_series input data.",
        aggregation.create_typical_periods,
    )


if __name__ == "__main__":
    test_assert_raises()
