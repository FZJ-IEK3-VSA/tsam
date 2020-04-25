import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam

def test_assert_raises():

    # Check error message for wrong time series (important: special signs such as brackets must be marked with '\' when
    # matching error message
    np.testing.assert_raises_regex(ValueError,
                                   'timeSeries has to be of type pandas.DataFrame\(\) or of type np.array\(\) in '
                                   'initialization of object of class TimeSeriesAggregation',
                                   tsam.TimeSeriesAggregation, timeSeries='erroneousTimeSeries')

if __name__ == "__main__":
    test_assert_raises()