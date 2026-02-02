import time

import numpy as np
import pandas as pd

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV


def test_averaging():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    noTypicalPeriods = 8

    hoursPerPeriod = 24

    starttime = time.time()

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=noTypicalPeriods,
        hoursPerPeriod=hoursPerPeriod,
        clusterMethod="averaging",
        representationMethod="meanRepresentation",
    )

    typPeriods = aggregation.createTypicalPeriods()

    print("Clustering took " + str(time.time() - starttime))

    # check whether the clusterOrder consists of noTypicalPeriods blocks of the same number
    np.testing.assert_array_almost_equal(
        np.size(np.where(np.diff(aggregation.clusterOrder) != 0)),
        noTypicalPeriods - 1,
        decimal=4,
    )

    # check whether the cluster centers are in line with the average of the candidates assigned to the different
    # clusters
    for i in range(noTypicalPeriods):
        calc = (
            tsam.unstackToPeriods(raw, hoursPerPeriod)[0]
            .loc[np.where(aggregation.clusterOrder == i)]
            .mean(axis=0)
            .to_frame()
            .values
        )
        orig = tsam.unstackToPeriods(typPeriods.loc[i], hoursPerPeriod)[0].T.values
        np.testing.assert_array_almost_equal(calc, orig, decimal=4)


if __name__ == "__main__":
    test_averaging()
