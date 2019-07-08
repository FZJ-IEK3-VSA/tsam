import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam



def test_aggregate_hiearchical():

    normalizedPeriodlyProfiles = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','examples','results','preprocessed_wind.csv'), index_col = [0], header = [0,1])

    clusterCenters, clusterCenterIndices, clusterOrder = tsam.aggregatePeriods(normalizedPeriodlyProfiles.values, n_clusters=8, clusterMethod='hierarchical',)

    orig = [0,1,1,0,0,1,3,2,2,3,1,1,1,1,1,1,7,3,6,6,4,1,4,1,1,1,5,4,0,1,0,3,1,1,4,5,2,
            7,6,2,1,3,0,1,1,5,7,6,5,0,5,5,0,2,5,1,2,6,5,0,4,0,1,3,3,1,4,1,2,0,2,2,1,1,
            5,2,2,3,5,2,2,4,1,3,4,3,7,7,7,2,2,2,7,3,4,1,5,1,3,5,4,3,2,3,4,1,2,2,2,6,2,
            5,2,2,2,7,7,5,3,2,1,2,6,7,3,3,2,1,4,6,1,2,2,6,7,2,4,7,7,1,2,2,2,6,2,2,7,2,
            2,2,2,2,3,2,5,2,2,2,6,6,2,6,2,7,7,2,7,2,3,2,2,6,2,2,7,2,2,2,7,2,7,6,6,6,3,
            6,7,6,6,2,2,2,2,2,6,6,2,2,2,2,3,7,2,6,2,7,2,2,7,2,2,7,6,2,2,6,3,5,1,0,3,7,
            2,7,3,2,7,7,2,3,3,6,2,2,6,6,6,7,7,7,7,2,5,3,2,3,3,2,2,4,3,7,6,2,5,3,7,2,7,
            7,3,2,7,7,6,2,3,7,6,7,6,7,2,2,6,2,1,1,5,3,6,3,3,1,4,3,4,2,6,1,0,0,0,0,4,1,
            0,1,1,3,2,5,2,6,5,5,1,6,6,4,5,4,2,2,5,7,6,2,3,7,2,6,6,2,2,2,7,5,5,7,6,6,6,
            1,1,6,5,1,2,5,5,1,5,0,1,1,1,5,6,6,5,1,2,6,6,2,3,2,2,5,0,1,0,1,1]

    np.testing.assert_array_almost_equal(orig, clusterOrder,decimal=4)

    
if __name__ == "__main__":
    test_aggregate_hiearchical()