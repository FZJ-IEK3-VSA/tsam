import os
import time

import pandas as pd
import numpy as np

import tsam.timeseriesaggregation as tsam
import tsam.hyperparametertuning as tune

def test_getPeriodPair():
    """Tests if the number of periods is properly defined if a datareduction is set
    """
    noRawTimeSteps=100
    segmentsPerPeriod=10
    dataReduction=0.5
    noPeriods=tune.getNoPeriodsForDataReduction(noRawTimeSteps, segmentsPerPeriod, dataReduction)
    assert noPeriods==5

    noRawTimeSteps=101
    noPeriods=tune.getNoPeriodsForDataReduction(noRawTimeSteps, segmentsPerPeriod, dataReduction)
    assert noPeriods==5
    
    segmentsPerPeriod=2
    noPeriods=tune.getNoPeriodsForDataReduction(noRawTimeSteps, segmentsPerPeriod, dataReduction)
    assert noPeriods==25
    
def test_optimalPair():

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )

    datareduction=0.1

    # just take wind
    aggregation_wind = tune.HyperTunedAggregations(
        tsam.TimeSeriesAggregation(
            raw.loc[:, ['Wind']],
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="durationRepresentation",
            distributionPeriodWise=False,
            rescaleClusterPeriods=False,
            segmentation=True,
        )
    )

    # and identify the best combination for a data reduction of to ~10%. 
    windSegments, windPeriods= aggregation_wind.identifyOptimalSegmentPeriodCombination(dataReduction=datareduction)

    # just take solar irradiation
    aggregation_solar = tune.HyperTunedAggregations(
        tsam.TimeSeriesAggregation(
            raw.loc[:, ['GHI']],
            hoursPerPeriod=24,
            clusterMethod="hierarchical",
            representationMethod="durationRepresentation",
            distributionPeriodWise=False,
            rescaleClusterPeriods=False,
            segmentation=True,
        )
    )

    # and identify the best combination for a data reduction of to ~10%. 
    solarSegments, solarPeriods = aggregation_solar.identifyOptimalSegmentPeriodCombination(dataReduction=datareduction)


    # according to Hoffmann et al. 2022 is for solar more segments and less days better than for wind
    assert windPeriods > solarPeriods
    assert windSegments < solarSegments

    # check if the number time steps is in the targeted range
    assert windPeriods * windSegments <= len(raw["Wind"])*datareduction
    assert windPeriods * windSegments >= len(raw["Wind"])*datareduction * 0.8

def test_paretoOptimalAggregation():

    raw = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "examples", "testdata.csv"),
        index_col=0,
    )

    # reduce the set, since it takes otherwise too long
    raw=raw.iloc[:240,:]

    # set tuned aggregation
    tunedAggregations = tune.HyperTunedAggregations(
        tsam.TimeSeriesAggregation(
            raw,
            hoursPerPeriod=12,
            clusterMethod="hierarchical",
            representationMethod="durationRepresentation",
            distributionPeriodWise=False,
            rescaleClusterPeriods=False,
            segmentation=True,
        )
    )

    # determine pareto optimal aggregation
    tunedAggregations.identifyParetoOptimalAggregation()

    # test if last RMSE is 0
    assert tunedAggregations._RMSEHistory[-1] == 0

    # test if RMSE is continously decreasing
    for i, RMSE in enumerate(tunedAggregations._RMSEHistory[1:]):
        assert RMSE <= tunedAggregations._RMSEHistory[i]


if __name__ == "__main__":
    test_paretoOptimalAggregation()
