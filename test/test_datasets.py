"""The designed datasets keep the properties the documentation relies on."""

import numpy as np
import pandas as pd

import tsam
from tsam import ClusterConfig
from tsam.datasets import representation_days


def test_representation_days_shape_and_resolution():
    data = representation_days()
    assert list(data.columns) == ["solar", "load"]
    assert len(data) == 80
    assert data.index.freq == pd.tseries.frequencies.to_offset("3h")
    assert (data >= 0).all().all()


def test_representation_days_peaks_shift_within_the_high_solar_days():
    days = tsam.unstack_to_periods(representation_days(), period_duration="1D")
    solar_peak_step = days["solar"].to_numpy().argmax(axis=1)
    assert solar_peak_step[:6].tolist() == [4, 5, 4, 5, 4, 5]
    load_peak_step = days["load"].to_numpy().argmax(axis=1)
    assert load_peak_step[:6].tolist() == [6, 7, 6, 7, 6, 7]
    high_peaks = days["solar"].iloc[:6].max(axis=1)
    low_peaks = days["solar"].iloc[6:].max(axis=1)
    assert high_peaks.min() == 750 and high_peaks.max() == 950
    assert low_peaks.min() == 200 and low_peaks.max() == 350


def test_representation_days_form_two_clusters():
    result = tsam.aggregate(
        representation_days(),
        n_clusters=2,
        period_duration="1D",
        cluster=ClusterConfig(method="hierarchical"),
    )
    assignments = result.cluster_assignments
    np.testing.assert_array_equal(assignments[:6], assignments[0])
    np.testing.assert_array_equal(assignments[6:], assignments[6])
    assert assignments[0] != assignments[6]
