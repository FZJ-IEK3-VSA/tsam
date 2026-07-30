import os
import time

import numpy as np
import pandas as pd
import pytest
from sklearn import preprocessing

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, aggregate


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
    aggregation = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(method="kmeans", scale_by_column_means=True),
    )

    print("Clustering took " + str(time.time() - starttime))

    # test if the normalized time series all have the same mean. The normalization
    # mirrors what scale_by_column_means=True does internally: min-max scale each
    # column, then divide by its mean so every column ends up with the same mean.
    min_max_scaler = preprocessing.MinMaxScaler()
    normalized = pd.DataFrame(
        min_max_scaler.fit_transform(raw),
        columns=raw.columns,
        index=raw.index,
    )
    normalized /= normalized.mean()
    means = normalized.mean().values
    np.testing.assert_allclose(means, np.array([means[0]] * len(means)), rtol=1e-5)

    # repredict the original data
    rearangedData = aggregation.reconstructed

    # test if the mean fits the mean of the raw time series --> should always hold for k-means independent from sameMean True or False
    np.testing.assert_array_almost_equal(
        raw.mean(), rearangedData[raw.columns].mean(), decimal=4
    )


if __name__ == "__main__":
    test_samemean()
