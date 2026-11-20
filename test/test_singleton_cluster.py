"""``Distribution(scope="local")`` must keep the temporal order of its clusters.
 
The local duration representation sorts each cluster's values to build the
duration curve and then derives the temporal order from the cluster's data.
If that sort happens in place on a reshape of the cluster data, it rewrites
the data whenever the reshape is a view instead of a copy. That is the case
for
 
* a cluster with a single member, whatever the number of attributes, and
* data with a single attribute, whatever the size of the cluster.
 
The order then degenerates to the identity and every attribute comes back as
a monotonically rising ramp, all perfectly correlated with each other. Only a
cluster with several members in data with several attributes is unaffected.
"""
 
import numpy as np
import pandas as pd
import pytest
 
from tsam import ClusterConfig, Distribution, aggregate
 
T = 24
 
 
@pytest.fixture(params=[["x"], ["x", "y"]], ids=["one_column", "two_columns"])
def three_periods(request):
    """Two alike periods and one on its own: hierarchical clustering into two
    clusters yields one pair and one singleton (period 2).
 
    Every test runs once with one attribute and once with two, because the
    two shapes fail for different clusters.
    """
    base = np.sin(np.linspace(0, 2 * np.pi, T, endpoint=False))
    periods = [10 + base, 10.5 + base, 30 - 5 * base]
    x = np.concatenate(periods)
    frame = pd.DataFrame(
        {"x": x, "y": 2 * x + 1},
        index=pd.date_range("2025-01-01", periods=3 * T, freq="h"),
    )
    return frame[request.param]
 
 
def _aggregate(frame, representation):
    return aggregate(
        frame,
        n_clusters=2,
        period_duration=T,
        cluster=ClusterConfig(method="hierarchical", representation=representation),
    )
 
 
def _singleton_representative(frame, representation):
    result = _aggregate(frame, representation)
    (singleton,) = [c for c, n in result.cluster_counts.items() if n == 1]
    (member,) = np.where(np.asarray(result.cluster_assignments) == singleton)[0]
    assert member == 2
    rep = result.cluster_representatives.loc[singleton].sort_index()
    return rep, frame.iloc[member * T : (member + 1) * T]
 
 
@pytest.mark.parametrize(
    "representation",
    [
        Distribution(scope="local"),
        # "medoid" orders by the cluster data itself, not by the mean profile
        Distribution(scope="local", concurrency="medoid"),
        Distribution(scope="global"),
    ],
    ids=["local", "local_medoid", "global"],
)
def test_singleton_cluster_is_represented_by_its_member(three_periods, representation):
    rep, member = _singleton_representative(three_periods, representation)
    for column in three_periods.columns:
        np.testing.assert_allclose(
            rep[column].to_numpy(), member[column].to_numpy(), atol=1e-12
        )
 
 
def test_singleton_cluster_is_not_a_sorted_ramp(three_periods):
    rep, member = _singleton_representative(three_periods, Distribution(scope="local"))
    for column in three_periods.columns:
        got = rep[column].to_numpy()
        # the duration curve is right either way ...
        np.testing.assert_allclose(np.sort(got), np.sort(member[column].to_numpy()))
        # ... but the values must keep the member's temporal order
        assert not np.all(np.diff(got) >= 0), f"{column} came back sorted ascending"
 
 
def test_multi_member_cluster_follows_mean_profile(three_periods):
    """The pair cluster is a real duration representation: the per-rank mean
    of the two members, ordered by their mean profile.
 
    With two attributes this cluster was never affected. With one attribute it
    came back as a sorted ramp, exactly like the singleton.
    """
    result = _aggregate(three_periods, Distribution(scope="local"))
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