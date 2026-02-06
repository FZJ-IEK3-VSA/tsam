import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import RESULTS_DIR, TESTDATA_CSV


def test_preprocess():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    raw_wind = raw.loc[:, "Wind"].to_frame()

    aggregation_wind = tsam.TimeSeriesAggregation(
        raw_wind,
        no_typical_periods=8,
        hours_per_period=24,
        cluster_method="hierarchical",
    )

    aggregation_wind._pre_process_time_series()

    test = aggregation_wind.normalized_periodly_profiles

    orig = pd.read_csv(
        RESULTS_DIR / "preprocessed_wind.csv",
        index_col=[0],
        header=[0, 1],
    )

    np.testing.assert_array_almost_equal(test.values, orig.values, decimal=15)


if __name__ == "__main__":
    test_preprocess()
