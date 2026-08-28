"""Tests for the ``KMedoids`` clustering-method config object.

Covers issue #365 (solver options stored but never forwarded to the solver) and
the strategy-object API that mirrors ``Distribution`` for representations.
"""

import warnings

import numpy as np
import pytest

from tsam import ClusterConfig, KMedoids
from tsam.config import method_from_dict, method_to_dict


def test_string_and_object_forms_agree_on_name_and_solver():
    assert ClusterConfig(method="kmedoids").method_name == "kmedoids"
    assert ClusterConfig(method=KMedoids()).method_name == "kmedoids"
    assert ClusterConfig(method=KMedoids(solver="gurobi")).solver == "gurobi"
    # non-kmedoids methods report the default solver and keep the raw string
    cfg = ClusterConfig(method="hierarchical")
    assert cfg.method == "hierarchical"
    assert cfg.solver == "highs"


def test_kmedoids_object_is_stored_raw_like_distribution():
    cfg = ClusterConfig(method=KMedoids(options={"time_limit": 42}))
    assert cfg.method == KMedoids(options={"time_limit": 42})
    assert ClusterConfig(method="kmedoids").get_representation() == "medoid"
    assert ClusterConfig(method=KMedoids()).get_representation() == "medoid"


def test_kmedoids_is_hashable_regardless_of_options():
    # options is excluded from __hash__ (dicts are unhashable) but kept in __eq__.
    assert hash(KMedoids(options={"time_limit": 5})) == hash(KMedoids())
    assert KMedoids(options={"time_limit": 5}) != KMedoids()
    # ClusterConfig stays hashable with an options-carrying method.
    assert isinstance(hash(ClusterConfig(method=KMedoids(options={"threads": 2}))), int)


def test_deprecated_solver_kwarg_forwards_into_kmedoids():
    with pytest.warns(DeprecationWarning):
        cfg = ClusterConfig(method="kmedoids", solver="gurobi")
    assert cfg.method == KMedoids(solver="gurobi")
    with pytest.warns(DeprecationWarning):
        cfg2 = ClusterConfig(method=KMedoids(options={"time_limit": 5}), solver="cbc")
    assert cfg2.method == KMedoids(solver="cbc", options={"time_limit": 5})


def test_solver_kwarg_ignored_for_non_kmedoids_but_still_warns():
    with pytest.warns(DeprecationWarning):
        cfg = ClusterConfig(method="hierarchical", solver="gurobi")
    assert cfg.method == "hierarchical"
    assert cfg.solver == "highs"


@pytest.mark.parametrize(
    "cfg",
    [
        ClusterConfig(method="hierarchical"),
        ClusterConfig(method="kmedoids"),
        ClusterConfig(method=KMedoids()),
        ClusterConfig(method=KMedoids(solver="gurobi", options={"time_limit": 42})),
    ],
)
def test_serialization_round_trips(cfg):
    assert ClusterConfig.from_dict(cfg.to_dict()) == cfg


def test_object_serializes_to_typed_dict_string_stays_string():
    assert method_to_dict("kmedoids") == "kmedoids"
    assert method_to_dict(KMedoids(options={"time_limit": 42})) == {
        "type": "kmedoids",
        "options": {"time_limit": 42},
    }
    assert method_from_dict(
        {"type": "kmedoids", "options": {"time_limit": 42}}
    ) == KMedoids(options={"time_limit": 42})


def test_legacy_flat_solver_dict_still_loads():
    # Configs serialized by the pre-KMedoids-object ClusterConfig used a flat
    # {"method": "kmedoids", "solver": ...} form.
    cfg = ClusterConfig.from_dict({"method": "kmedoids", "solver": "gurobi"})
    assert cfg.method == KMedoids(solver="gurobi")


def test_options_are_forwarded_to_the_solver(monkeypatch):
    """Regression for #365: configured solver options must reach the solver."""
    import tsam.algorithms.k_medoids_exact as kme

    captured = {}
    real_highs_cls = kme.appsi.solvers.Highs

    class RecordingHighs(real_highs_cls):
        def solve(self, model, *args, **kwargs):
            captured["highs_options"] = dict(self.highs_options)
            return super().solve(model, *args, **kwargs)

    monkeypatch.setattr(kme.appsi.solvers, "Highs", RecordingHighs)

    X = np.random.RandomState(0).rand(8, 4)
    kme.ExactKMedoids(n_clusters=3, options={"time_limit": 37}).fit(X)

    assert captured["highs_options"] == {"time_limit": 37}


def test_no_options_leaves_solver_defaults(monkeypatch):
    import tsam.algorithms.k_medoids_exact as kme

    captured = {}
    real_highs_cls = kme.appsi.solvers.Highs

    class RecordingHighs(real_highs_cls):
        def solve(self, model, *args, **kwargs):
            captured["highs_options"] = dict(self.highs_options)
            return super().solve(model, *args, **kwargs)

    monkeypatch.setattr(kme.appsi.solvers, "Highs", RecordingHighs)
    X = np.random.RandomState(0).rand(8, 4)
    kme.ExactKMedoids(n_clusters=3).fit(X)
    assert captured["highs_options"] == {}


def test_aggregate_accepts_kmedoids_object():
    import pandas as pd

    from tsam import aggregate

    rng = np.random.RandomState(0)
    raw = pd.DataFrame(rng.rand(96, 3), columns=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = aggregate(
            raw,
            n_clusters=3,
            period_duration=24,
            cluster=ClusterConfig(method=KMedoids(options={"time_limit": 60})),
        )
    assert result.cluster_representatives is not None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
