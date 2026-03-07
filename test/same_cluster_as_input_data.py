import itertools

import numpy as np
import pandas as pd
import pytest

import tsam
from tsam import ClusterConfig

# All clustering methods (excluding "averaging" which does not cluster into n_clusters)
_METHODS = ["kmeans", "kmedoids", "kmaxoids", "hierarchical", "contiguous"]

# All representation methods
_REPRESENTATIONS = ["mean", "medoid", "maxoid", "distribution", "distribution_minmax"]

# Use duration curves when clustering by value distribution
_DISTRIBUTION_REPS = {"distribution", "distribution_minmax"}

_PARAMS = [
    pytest.param(
        method,
        rep,
        rep in _DISTRIBUTION_REPS,
        id=f"{method}_{rep}",
    )
    for method, rep in itertools.product(_METHODS, _REPRESENTATIONS)
]


@pytest.fixture(scope="module")
def input_data() -> pd.DataFrame:
    costs = pd.DataFrame(
        [
            np.array([0.05, 0.0, 0.1, 0.051]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ],
        index=["ElectrolyzerLocation", "IndustryLocation"],
    ).T
    revenues = pd.DataFrame(
        [
            np.array([0.0, 0.01, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0]),
        ],
        index=["ElectrolyzerLocationRevenue", "IndustryLocationRevenue"],
    ).T

    timeSeriesData = pd.concat([costs, revenues], axis=1)
    timeSeriesData.index = pd.date_range(
        "2050-01-01 00:30:00",
        periods=4,
        freq="1h",
        tz="Europe/Berlin",
    )
    return timeSeriesData


@pytest.mark.parametrize("method,representation,use_duration_curves", _PARAMS)
def test_same_cluster_as_input_data(
    input_data: pd.DataFrame,
    method: str,
    representation: str,
    use_duration_curves: bool,
) -> None:
    """When n_clusters equals the number of input periods, reconstruction must
    be identical to the original time series for every method/representation."""
    results = tsam.aggregate(
        input_data,
        n_clusters=4,
        period_duration=1,
        cluster=ClusterConfig(
            method=method,
            representation=representation,
            use_duration_curves=use_duration_curves,
        ),
    )
    pd.testing.assert_frame_equal(results.reconstructed, input_data)
