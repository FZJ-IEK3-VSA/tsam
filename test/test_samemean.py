import os
import time

import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV

pytestmark = pytest.mark.filterwarnings("ignore::tsam.exceptions.LegacyAPIWarning")


@pytest.mark.filterwarnings("ignore::RuntimeWarning:threadpoolctl")
@pytest.mark.filterwarnings(
    "ignore:KMeans is known to have a memory leak on Windows with MKL.*:UserWarning"
)
def test_samemean():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    # get all columns as floats to avoid warning
    for col in raw.columns:
        raw[col] = raw[col].astype(float)

    starttime = time.time()

    # Silence warning on machines that cannot detect their physical cpu cores
    os.environ["OMP_NUM_THREADS"] = "1"
    aggregation = tsam.TimeSeriesAggregation(
        raw,
        no_typical_periods=8,
        hours_per_period=24,
        cluster_method="k_means",
        same_mean=True,
    )

    typPeriods = aggregation.create_typical_periods()
    print("Clustering took " + str(time.time() - starttime))

    # repredict the original data
    rearangedData = aggregation.predict_original_data()

    # test if the mean fits the mean of the raw time series --> should always hold for k-means independent from same_mean True or False
    np.testing.assert_array_almost_equal(
        raw.mean(), rearangedData[raw.columns].mean(), decimal=4
    )


if __name__ == "__main__":
    test_samemean()
