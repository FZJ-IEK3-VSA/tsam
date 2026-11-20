"""A one-member cluster under ``Distribution(scope="local")`` must be
represented by its member, not by the member's values sorted ascending.

The local duration representation sorts each cluster's values in place to
build the duration curve and then derives the temporal order from the
cluster's mean profile. For a single member the flattened array is a view of
the cluster data, so the in-place sort also rewrites the data the mean profile
is taken from: the order degenerates to the identity and every attribute comes
back as a monotonically rising ramp, all perfectly correlated with each other.
"""

import numpy as np
import pandas as pd
import pytest

from tsam import ClusterConfig, Distribution, aggregate

T = 24


@pytest.fixture
def three_periods():
    """Two alike periods and one on its own: hierarchical clustering into two
    clusters yields one pair and one singleton (period 2)."""
    base = np.sin(np.linspace(0, 2 * np.pi, T, endpoint=False))
    periods = [10 + base, 10.5 + base, 30 - 5 * base]
    x = np.concatenate(periods)
    return pd.DataFrame(
        {"x": x, "y": 2 * x + 1},
        index=pd.date_range("2025-01-01", periods=3 * T, freq="h"),
    )


def _singleton_representative(frame, scope):
    result = aggregate(
        frame,
        n_clusters=2,
        period_duration=T,
        cluster=ClusterConfig(
            method="hierarchical", representation=Distribution(scope=scope)
        ),
    )
    (singleton,) = [c for c, n in result.cluster_counts.items() if n == 1]
    (member,) = np.where(np.asarray(result.cluster_assignments) == singleton)[0]
    assert member == 2
    rep = result.cluster_representatives.loc[singleton].sort_index()
    return rep, frame.iloc[member * T : (member + 1) * T]


@pytest.mark.parametrize("scope", ["local", "global"])
def test_singleton_cluster_is_represented_by_its_member(three_periods, scope):
    rep, member = _singleton_representative(three_periods, scope)
    for column in three_periods.columns:
        np.testing.assert_allclose(
            rep[column].to_numpy(), member[column].to_numpy(), atol=1e-12
        )


def test_singleton_cluster_is_not_a_sorted_ramp(three_periods):
    rep, member = _singleton_representative(three_periods, "local")
    for column in three_periods.columns:
        got = rep[column].to_numpy()
        # the duration curve is right either way ...
        np.testing.assert_allclose(np.sort(got), np.sort(member[column].to_numpy()))
        # ... but the values must keep the member's temporal order
        assert not np.all(np.diff(got) >= 0), f"{column} came back sorted ascending"


def test_multi_member_cluster_unchanged(three_periods):
    """The pair cluster is a real duration representation: the per-rank mean
    of the two members, ordered by their mean profile."""
    result = aggregate(
        three_periods,
        n_clusters=2,
        period_duration=T,
        cluster=ClusterConfig(
            method="hierarchical", representation=Distribution(scope="local")
        ),
    )
    (pair,) = [c for c, n in result.cluster_counts.items() if n == 2]
    rep = result.cluster_representatives.loc[pair].sort_index()["x"].to_numpy()
    members = three_periods["x"].to_numpy()[: 2 * T]
    expected_sorted = np.sort(members).reshape(T, 2).mean(axis=1)
    np.testing.assert_allclose(np.sort(rep), expected_sorted)
    # ordered like the members' mean profile (ties in the symmetric sine
    # make an exact rank comparison brittle; rank correlation is the claim)
    mean_profile = members.reshape(2, T).mean(axis=0)
    ranks = lambda a: np.argsort(np.argsort(a, kind="stable"), kind="stable")  # noqa: E731
    assert np.corrcoef(ranks(rep), ranks(mean_profile))[0, 1] > 0.99
