import numpy as np
import pandas as pd

import tsam
from tsam import ClusterConfig


def test_same_cluster_as_input_data():
    costs = pd.DataFrame(
        [
            np.array(
                [
                    0.05,
                    0.0,
                    0.1,
                    0.051,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
    ).T
    revenues = pd.DataFrame(
        [
            np.array(
                [
                    0.0,
                    0.01,
                    0.0,
                    0.0,
                ]
            ),
            np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ),
        ],
        index=["ElectrolyzerLocationRevenue", "IndustryLocationRevenue"],
    ).T

    timeSeriesData = pd.concat([costs, revenues], axis=1)
    timeSeriesData.index = pd.date_range(
        "2050-01-01 00:30:00",
        periods=4,
        freq=(str(1) + "h"),
        tz="Europe/Berlin",
    )

    results = tsam.aggregate(
        timeSeriesData,
        n_clusters=4,
        period_duration=1,
        cluster=ClusterConfig(
            method="hierarchical",
            representation="distribution",
            use_duration_curves=True,
        ),
    )
    pd.testing.assert_frame_equal(results.reconstructed, timeSeriesData)
