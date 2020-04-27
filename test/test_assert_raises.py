import os
import copy

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam

def test_assert_raises():
    # important: special signs such as brackets must be marked with '\' when matching error message

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'examples', 'testdata.csv'), index_col=0)

    # check error message for wrong time series
    np.testing.assert_raises_regex(ValueError,
                                   'timeSeries has to be of type pandas.DataFrame\(\) or of type np.array\(\) in '
                                   'initialization of object of class TimeSeriesAggregation',
                                   tsam.TimeSeriesAggregation, timeSeries='erroneousTimeSeries')

    # check error messages for wrong attribute names added for extreme period methods
    np.testing.assert_raises_regex(ValueError,
                                   'erroneousAttribute listed in "addPeakMin" does not occur as timeSeries column',
                                   tsam.TimeSeriesAggregation, timeSeries=raw, addPeakMin = ['erroneousAttribute'])
    np.testing.assert_raises_regex(ValueError,
                                   'erroneousAttribute listed in "addPeakMax" does not occur as timeSeries column',
                                   tsam.TimeSeriesAggregation, timeSeries=raw, addPeakMax = ['erroneousAttribute'])
    np.testing.assert_raises_regex(ValueError,
                                   'erroneousAttribute listed in "addMeanMin" does not occur as timeSeries column',
                                   tsam.TimeSeriesAggregation, timeSeries=raw, addMeanMin = ['erroneousAttribute'])
    np.testing.assert_raises_regex(ValueError,
                                   'erroneousAttribute listed in "addMeanMax" does not occur as timeSeries column',
                                   tsam.TimeSeriesAggregation, timeSeries=raw, addMeanMax = ['erroneousAttribute'])

    # check error message for missing datetime index and missing resolution argument
    np.testing.assert_raises_regex(ValueError,
                                   '\'resolution\' argument has to be nonnegative float or int or the given ' \
                                   'timeseries needs a datetime index',
                                   tsam.TimeSeriesAggregation, timeSeries=raw.reset_index())
    # overwrite one of the datetime-like string indices in the raw data to an index that cannot be converted
    rawErrInd = copy.deepcopy(raw)
    as_list = rawErrInd.index.tolist()
    idx = as_list.index('2010-01-01 00:30:00')
    as_list[idx] = 'erroneousDatetimeIndex'
    rawErrInd.index = as_list
    # check error message for erroneous datetime index and missing resolution argument
    np.testing.assert_raises_regex(ValueError,
                                   '\'resolution\' argument has to be nonnegative float or int or the given ' \
                                   'timeseries needs a datetime index',
                                   tsam.TimeSeriesAggregation, timeSeries=rawErrInd)
    # check erroneous resolution argument
    np.testing.assert_raises_regex(ValueError,
                                   'resolution has to be nonnegative float or int',
                                   tsam.TimeSeriesAggregation, timeSeries=raw, resolution='erroneousResolution')

    # check erroneous hoursPerPeriod argument
    np.testing.assert_raises_regex(ValueError,
                                   'hoursPerPeriod has to be nonnegative integer',
                                   tsam.TimeSeriesAggregation, timeSeries=raw, hoursPerPeriod=None)

    # check non-integer time step number per typical period
    np.testing.assert_raises_regex(ValueError,
                                   'The combination of hoursPerPeriod and the resulution does not result in an integer '
                                   'number of time steps per period',
                                   tsam.TimeSeriesAggregation, timeSeries=raw, hoursPerPeriod=23, resolution=2)

    # check warning when number of segments per period is higher than the number of time steps per period
    np.testing.assert_warns(Warning, tsam.TimeSeriesAggregation, timeSeries=raw, segmentation=True, noSegments=25)

    # check erroneous clusterMethod argument
    np.testing.assert_raises_regex(ValueError,
                                   'clusterMethod needs to be one of the following: \[\'averaging\', \'k_medoids\', '
                                   '\'k_means\', \'hierarchical\', \'adjacent_periods\'\]',
                                   tsam.TimeSeriesAggregation, timeSeries=raw, clusterMethod='erroneousClusterMethod')

    # check erroneous representationMethod argument
    np.testing.assert_raises_regex(ValueError,
                                   'If specified, representationMethod needs to be one of the following: '
                                   '\[\'meanRepresentation\', \'medoidRepresentaion\', \'minmaxRepresentation\'\]',
                                   tsam.TimeSeriesAggregation, timeSeries=raw,
                                   representationMethod='erroneousRepresentationMethod')

    # check erroneous extremePeriodMethod argument
    np.testing.assert_raises_regex(ValueError,
                                   'extremePeriodMethod needs to be one of the following: \[\'None\', \'append\', '
                                   '\'new_cluster_center\', \'replace_cluster_center\'\]',
                                   tsam.TimeSeriesAggregation, timeSeries=raw,
                                   extremePeriodMethod='erroneousExtremePeriodMethod')

if __name__ == "__main__":
    test_assert_raises()