import copy
import time

import numpy as np
import pandas as pd

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, aggregate, unstack_to_periods


def test_durationCurve():
    # do everything for one attribute only to make sure that scaling does not play a role
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)["GHI"].to_frame()

    n_clusters = 8

    period_duration = 24

    starttime = time.time()

    result = aggregate(
        raw,
        n_clusters=n_clusters,
        period_duration=period_duration,
        cluster=ClusterConfig(method="hierarchical", use_duration_curves=True),
        preserve_column_means=False,
    )

    typPeriods = result.cluster_representatives

    print("Clustering took " + str(time.time() - starttime))

    # sort every attribute in every period in descending order for both, the found typical period and the days
    # that belong to the corresponding cluster
    for i in range(n_clusters):
        calculated = unstack_to_periods(raw, period_duration).loc[
            np.where(result.cluster_assignments == i)[0], :
        ]
        calculatedSorted = copy.deepcopy(calculated)
        algorithmResult = unstack_to_periods(typPeriods.loc[i], period_duration)
        for j in raw.columns:
            dfR = algorithmResult[j]
            dfR[dfR.columns] = np.sort(dfR)[:, ::-1]
            algorithmResult[j] = dfR
            df = calculatedSorted[j]
            df[df.columns] = np.sort(df)[:, ::-1]
            calculatedSorted[j] = df

        # make sure that the found typical period is always the one that is closest to the clusters centroid
        currentMeant = calculatedSorted.mean(axis=0)
        minIdx = np.square(calculatedSorted - currentMeant).sum(axis=1).idxmin()
        np.testing.assert_array_almost_equal(
            calculatedSorted.loc[minIdx], algorithmResult.iloc[0], decimal=4
        )


if __name__ == "__main__":
    test_durationCurve()
