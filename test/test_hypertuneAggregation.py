import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, Distribution, SegmentConfig, aggregate
from tsam.tuning import (
    find_clusters_for_reduction,
    find_optimal_combination,
    find_pareto_front,
)


def test_getPeriodPair():
    """Tests if the number of periods is properly defined if a datareduction is set"""
    noRawTimeSteps = 100
    segmentsPerPeriod = 10
    dataReduction = 0.5
    noPeriods = find_clusters_for_reduction(
        noRawTimeSteps, segmentsPerPeriod, dataReduction
    )
    assert noPeriods == 5

    noRawTimeSteps = 101
    noPeriods = find_clusters_for_reduction(
        noRawTimeSteps, segmentsPerPeriod, dataReduction
    )
    assert noPeriods == 5

    segmentsPerPeriod = 2
    noPeriods = find_clusters_for_reduction(
        noRawTimeSteps, segmentsPerPeriod, dataReduction
    )
    assert noPeriods == 25


def test_optimalPair():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    datareduction = 0.01

    cluster = ClusterConfig(
        method="hierarchical",
        representation=Distribution(scope="global"),
    )

    # just take wind, and identify the best combination for a data reduction of ~1%.
    wind = find_optimal_combination(
        raw.loc[:, ["Wind"]],
        data_reduction=datareduction,
        period_duration=24,
        cluster=cluster,
        preserve_column_means=False,
        show_progress=False,
    )
    windSegments, windPeriods = wind.n_segments, wind.n_clusters

    # just take solar irradiation, and identify the best combination for ~1%.
    solar = find_optimal_combination(
        raw.loc[:, ["GHI"]],
        data_reduction=datareduction,
        period_duration=24,
        cluster=cluster,
        preserve_column_means=False,
        show_progress=False,
    )
    solarSegments, solarPeriods = solar.n_segments, solar.n_clusters

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

    cluster = ClusterConfig(method="hierarchical", representation="mean")

    # identify the best combination for a data reduction.
    optimal = find_optimal_combination(
        raw,
        data_reduction=datareduction,
        period_duration=24,
        cluster=cluster,
        preserve_column_means=False,
        show_progress=False,
    )
    RMSEOpt = optimal.rmse

    # test steepest descent up to the same number of total timesteps
    pareto = find_pareto_front(
        raw,
        period_duration=24,
        max_timesteps=365 * SEGMENTS_TESTED,
        cluster=cluster,
        preserve_column_means=False,
        show_progress=False,
    )
    RMSEsteepest = pareto.history[-1]["rmse"]

    # only segments
    aggregation = aggregate(
        raw,
        n_clusters=365,
        period_duration=24,
        cluster=cluster,
        segments=SegmentConfig(n_segments=SEGMENTS_TESTED),
    )

    RMSESegments = aggregation.accuracy.weighted_rmse

    assert RMSEsteepest < RMSESegments

    assert np.isclose(RMSEsteepest, RMSEOpt, atol=1e-3)


def test_paretoOptimalAggregation():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # reduce the set, since it takes otherwise too long
    raw = raw.iloc[:240, :]

    # determine pareto optimal aggregation
    pareto = find_pareto_front(
        raw,
        period_duration=12,
        cluster=ClusterConfig(
            method="hierarchical",
            representation=Distribution(scope="global"),
        ),
        preserve_column_means=False,
        show_progress=False,
    )

    rmse_history = [entry["rmse"] for entry in pareto.history]

    # test if last RMSE is 0
    assert rmse_history[-1] == 0

    # test if RMSE is continously decreasing
    for i, RMSE in enumerate(rmse_history[1:]):
        assert RMSE <= rmse_history[i]


if __name__ == "__main__":
    test_paretoOptimalAggregation()
