import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam

def test_getPeriodPair():
    """Tests if the number of periods is properly defined if a datareduction is set
    """
    noRawTimeSteps=100
    segmentsPerPeriod=10
    dataReduction=0.5
    noPeriods=tsam.getNoPeriodsForDataReduction(noRawTimeSteps, segmentsPerPeriod, dataReduction)
    assert noPeriods==5

    noRawTimeSteps=101
    noPeriods=tsam.getNoPeriodsForDataReduction(noRawTimeSteps, segmentsPerPeriod, dataReduction)
    assert noPeriods==5
    
    segmentsPerPeriod=2
    noPeriods=tsam.getNoPeriodsForDataReduction(noRawTimeSteps, segmentsPerPeriod, dataReduction)
    assert noPeriods==25
    
def test_optimalPair():

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )

    datareduction=0.1

    # just take wind
    aggregation_wind = tsam.TimeSeriesAggregation(
        raw.loc[:, ['Wind']],
        hoursPerPeriod=24,
        clusterMethod="hierarchical",
        representationMethod="durationRepresentation",
        distributionPeriodWise=False,
        rescaleClusterPeriods=False,
        segmentation=True,
    )

    # and identify the best combination for a data reduction of to ~10%. 
    aggregation_wind.identifyOptimalSegmentPeriodCombination(dataReduction=datareduction)

    # just take solar irradiation
    aggregation_solar = tsam.TimeSeriesAggregation(
        raw.loc[:, ['GHI']],
        hoursPerPeriod=24,
        clusterMethod="hierarchical",
        representationMethod="durationRepresentation",
        distributionPeriodWise=False,
        rescaleClusterPeriods=False,
        segmentation=True,
    )

    # and identify the best combination for a data reduction of to ~10%. 
    aggregation_solar.identifyOptimalSegmentPeriodCombination(dataReduction=datareduction)


    # according to Hoffmann et al. 2022 is for solar more segments and less days better than for wind
    assert aggregation_wind.noTypicalPeriods > aggregation_solar.noTypicalPeriods
    assert aggregation_wind.noSegments < aggregation_solar.noSegments

    # check if the number time steps is in the targeted range
    assert aggregation_wind.noTypicalPeriods * aggregation_wind.noSegments <= len(raw["Wind"])*datareduction
    assert aggregation_wind.noTypicalPeriods * aggregation_wind.noSegments >= len(raw["Wind"])*datareduction * 0.8

if __name__ == "__main__":
    test_getPeriodPair()
    test_optimalPair()
