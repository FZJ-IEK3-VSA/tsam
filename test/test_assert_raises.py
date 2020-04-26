import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam

def test_assert_raises():
    # important: special signs such as brackets must be marked with '\' when matching error message

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'examples', 'testdata.csv'), index_col=0)

    # Check error message for wrong time series
    np.testing.assert_raises_regex(ValueError,
                                   'timeSeries has to be of type pandas.DataFrame\(\) or of type np.array\(\) in '
                                   'initialization of object of class TimeSeriesAggregation',
                                   tsam.TimeSeriesAggregation, timeSeries='erroneousTimeSeries')

    # Check error messages for wrong attribute names added for extreme period methods
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

if __name__ == "__main__":
    test_assert_raises()