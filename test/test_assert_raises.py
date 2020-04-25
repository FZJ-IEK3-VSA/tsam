import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam

def test_assert_raises():

    raw = pd.read_csv(os.path.join(os.path.dirname(__file__), '..', 'examples', 'testdata.csv'), index_col=0)

    def div(x, y=1):
        if x==1:
            raise ValueError('bla')

    np.testing.assert_raises_regex(ValueError, 'bla', div, 1)

if __name__ == "__main__":
    test_assert_raises()