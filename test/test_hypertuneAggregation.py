import numpy as np
import pandas as pd
import pytest

import tsam.hyperparametertuning as tune
import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_getPeriodPair():
    """Tests if the number of periods is properly defined if a datareduction is set"""
    n_raw_timesteps = 100
    segments_per_period = 10
    data_reduction = 0.5
    noPeriods = tune.get_no_periods_for_data_reduction(
        n_raw_timesteps, segments_per_period, data_reduction
    )
    assert noPeriods == 5

    n_raw_timesteps = 101
    noPeriods = tune.get_no_periods_for_data_reduction(
        n_raw_timesteps, segments_per_period, data_reduction
    )
    assert noPeriods == 5

    segments_per_period = 2
    noPeriods = tune.get_no_periods_for_data_reduction(
        n_raw_timesteps, segments_per_period, data_reduction
    )
    assert noPeriods == 25


def test_optimalPair():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    datareduction = 0.01

    # just take wind
    aggregation_wind = tune.HyperTunedAggregations(
        tsam.TimeSeriesAggregation(
            raw.loc[:, ["Wind"]],
            hours_per_period=24,
            cluster_method="hierarchical",
            representation_method="durationRepresentation",
            distribution_period_wise=False,
            rescale_cluster_periods=False,
            segmentation=True,
        )
    )

    # and identify the best combination for a data reduction of to ~10%.
    windSegments, windPeriods, _windRMSE = (
        aggregation_wind.identify_optimal_segment_period_combination(
            data_reduction=datareduction
        )
    )

    # just take solar irradiation
    aggregation_solar = tune.HyperTunedAggregations(
        tsam.TimeSeriesAggregation(
            raw.loc[:, ["GHI"]],
            hours_per_period=24,
            cluster_method="hierarchical",
            representation_method="durationRepresentation",
            distribution_period_wise=False,
            rescale_cluster_periods=False,
            segmentation=True,
        )
    )

    # and identify the best combination for a data reduction of to ~10%.
    solarSegments, solarPeriods, _solarRMSE = (
        aggregation_solar.identify_optimal_segment_period_combination(
            data_reduction=datareduction
        )
    )

    # according to Hoffmann et al. 2022 is for solar more segments and less days better than for wind
    assert windPeriods > solarPeriods
    assert windSegments < solarSegments

    # check if the number time steps is in the targeted range
    assert windPeriods * windSegments <= len(raw["Wind"]) * datareduction
    assert windPeriods * windSegments >= len(raw["Wind"]) * datareduction * 0.8


@pytest.mark.skip(reason="This test is too slow")
def test_steepest_gradient_leads_to_optima():
    """
    Based on the hint of Eva Simarik, check if the RMSE is for the optimized combination
    of segments and periods smaller than sole segmentation approach
    """

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    SEGMENTS_TESTED = 5

    datareduction = (SEGMENTS_TESTED * 365) / 8760

    # just take wind
    tunedAggregations = tune.HyperTunedAggregations(
        tsam.TimeSeriesAggregation(
            raw,
            hours_per_period=24,
            cluster_method="hierarchical",
            representation_method="meanRepresentation",
            rescale_cluster_periods=False,
            segmentation=True,
        )
    )

    # and identify the best combination for a data reduction.
    _segmentsOpt, _periodsOpt, RMSEOpt = (
        tunedAggregations.identify_optimal_segment_period_combination(
            data_reduction=datareduction
        )
    )

    # test steepest
    tunedAggregations.identify_pareto_optimal_aggregation(
        until_total_timesteps=365 * SEGMENTS_TESTED
    )
    steepestAggregation = tunedAggregations.aggregation_history[-1]
    RMSEsteepest = steepestAggregation.total_accuracy_indicators()["RMSE"]

    # only segments
    aggregation = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=365,
        hours_per_period=24,
        segmentation=True,
        no_segments=SEGMENTS_TESTED,
        cluster_method="hierarchical",
        representation_method="meanRepresentation",
    )

    RMSESegments = aggregation.total_accuracy_indicators()["RMSE"]

    assert RMSEsteepest < RMSESegments

    assert np.isclose(RMSEsteepest, RMSEOpt, atol=1e-3)


def test_paretoOptimalAggregation():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # reduce the set, since it takes otherwise too long
    raw = raw.iloc[:240, :]

    # set tuned aggregation
    tunedAggregations = tune.HyperTunedAggregations(
        tsam.TimeSeriesAggregation(
            raw,
            hours_per_period=12,
            cluster_method="hierarchical",
            representation_method="meanRepresentation",
            distribution_period_wise=False,
            rescale_cluster_periods=False,
            segmentation=True,
        )
    )

    # determine pareto optimal aggregation
    tunedAggregations.identify_pareto_optimal_aggregation()

    # test if last RMSE is 0
    assert tunedAggregations._rmse_history[-1] == 0

    # test if RMSE is continously decreasing
    for i, RMSE in enumerate(tunedAggregations._rmse_history[1:]):
        assert RMSE <= tunedAggregations._rmse_history[i]


if __name__ == "__main__":
    test_paretoOptimalAggregation()
