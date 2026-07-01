"""Tests for the ``concurrency`` ordering strategies of the distribution
representation.

The distribution representation builds each cluster's typical period from
per-attribute duration curves and then re-orders those values along a synthetic
time axis. ``Distribution.concurrency`` selects how that ordering is derived.
All strategies preserve every attribute's marginal distribution (duration
curve); they differ only in how the attributes' concurrency (co-incidence in
time) across a common time axis is preserved:

- ``independent``: each attribute by its own mean profile (no concurrency);
- ``reference``:   one attribute's ordering, broadcast to all;
- ``medoid``:      the cluster medoid's per-attribute ranks (a real period);
- ``consensus``:   first principal component of all mean profiles;
- ``assignment``:  optimal single ordering vs the mean profile.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, Distribution, aggregate
from tsam.algorithms.concurrency import CONCURRENCY_METHODS
from tsam.algorithms.duration_representation import duration_representation

# A small multi-attribute, multi-member cluster: 2 attributes, 3 time steps,
# 3 candidates assigned to a single cluster.
_CANDIDATES = np.array(
    [
        [1.0, 2.0, 3.0, 30.0, 20.0, 10.0],
        [2.0, 4.0, 1.0, 5.0, 25.0, 15.0],
        [3.0, 1.0, 2.0, 12.0, 8.0, 40.0],
    ]
)
_CLUSTER_ORDER = np.array([0, 0, 0])
_TS_PER_PERIOD = 3


def _run(method, **kwargs):
    return np.array(
        duration_representation(
            _CANDIDATES,
            _CLUSTER_ORDER,
            True,
            _TS_PER_PERIOD,
            concurrency_method=method,
            **kwargs,
        )
    )


@pytest.mark.parametrize("method", CONCURRENCY_METHODS)
def test_all_strategies_preserve_marginals(method):
    """Every strategy yields the same per-attribute duration curve (same multiset
    of values per attribute and cluster); only the ordering differs."""
    kwargs = {"reference_attribute_idx": 0} if method == "reference" else {}
    result = _run(method, **kwargs)
    independent = _run("independent")
    for attr in range(2):
        block = slice(attr * _TS_PER_PERIOD, (attr + 1) * _TS_PER_PERIOD)
        np.testing.assert_array_almost_equal(
            np.sort(result[0, block]),
            np.sort(independent[0, block]),
        )


def test_default_matches_independent():
    """No concurrency / no reference == 'independent' (back-compat default)."""
    default = np.array(
        duration_representation(_CANDIDATES, _CLUSTER_ORDER, True, _TS_PER_PERIOD)
    )
    np.testing.assert_array_almost_equal(default, _run("independent"))


def test_reference_default_uses_reference_when_idx_given():
    """No explicit method + a reference index resolves to 'reference'."""
    implicit = np.array(
        duration_representation(
            _CANDIDATES,
            _CLUSTER_ORDER,
            True,
            _TS_PER_PERIOD,
            reference_attribute_idx=1,
        )
    )
    explicit = _run("reference", reference_attribute_idx=1)
    np.testing.assert_array_almost_equal(implicit, explicit)


@pytest.mark.parametrize("method", ["reference", "consensus", "assignment"])
def test_shared_axis_methods_are_comonotone(method):
    """Single-shared-axis strategies place all attributes on one ordering, so
    within the period every attribute is co-monotone with attribute 0."""
    kwargs = {"reference_attribute_idx": 0} if method == "reference" else {}
    result = _run(method, **kwargs)
    order = np.argsort(result[0, 0:_TS_PER_PERIOD], kind="stable")
    for attr in range(2):
        block = slice(attr * _TS_PER_PERIOD, (attr + 1) * _TS_PER_PERIOD)
        ordered = result[0, block][order]
        assert np.all(np.diff(ordered) >= -1e-9)


def test_medoid_method_reproduces_medoid_ordering():
    """The medoid strategy orders each attribute by the medoid period's own
    per-attribute ranks."""
    result = _run("medoid")
    dist = np.linalg.norm(_CANDIDATES[:, None, :] - _CANDIDATES[None, :, :], axis=2)
    medoid = _CANDIDATES[np.argmin(dist.sum(axis=0))]
    for attr in range(2):
        block = slice(attr * _TS_PER_PERIOD, (attr + 1) * _TS_PER_PERIOD)
        medoid_order = np.argsort(medoid[block], kind="stable")
        assert np.all(np.diff(result[0, block][medoid_order]) >= -1e-9)


@pytest.mark.parametrize("method", ["medoid", "consensus"])
def test_svd_and_medoid_are_deterministic(method):
    """medoid (argmin tie-break) and consensus (SVD sign) must be reproducible."""
    a = _run(method)
    b = _run(method)
    np.testing.assert_array_equal(a, b)


def _aggregate(raw, concurrency, *, preserve_minmax=False, rescale=False, **rep_kw):
    return aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(
            method="hierarchical",
            use_duration_curves=False,
            representation=Distribution(
                scope="cluster",
                preserve_minmax=preserve_minmax,
                concurrency=concurrency,
                **rep_kw,
            ),
        ),
        preserve_column_means=rescale,
    )


def test_medoid_improves_concurrency_over_independent():
    """On real, multi-attribute data the medoid strategy preserves the
    cross-attribute correlation structure better than independent ordering,
    while keeping the same marginal (duration-curve) error."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    independent = _aggregate(raw, "independent")
    medoid = _aggregate(raw, "medoid")

    assert (
        medoid.concurrency["corr_frobenius"] < independent.concurrency["corr_frobenius"]
    )
    np.testing.assert_allclose(
        medoid.accuracy.rmse_duration.values,
        independent.accuracy.rmse_duration.values,
        rtol=1e-9,
        atol=1e-12,
    )


@pytest.mark.filterwarnings("ignore:The cluster is too small:UserWarning")
def test_medoid_with_minmax_preserves_envelope_and_integral():
    """medoid + preserve_minmax + rescale stays inside the input envelope and
    preserves the integral."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    predicted = _aggregate(
        raw, "medoid", preserve_minmax=True, rescale=True
    ).reconstructed

    assert (predicted.max() <= raw.max() + 1e-10).all()
    assert (predicted.min() >= raw.min() - 1e-10).all()
    np.testing.assert_allclose(raw.sum(), predicted.sum(), rtol=5e-3)


@pytest.mark.filterwarnings("ignore:The cluster is too small:UserWarning")
@pytest.mark.parametrize("concurrency", ["independent", "medoid"])
def test_periodwise_minmax_preserves_integral_without_rescaling(concurrency):
    """Period-wise distribution + min/max preserves the integral on its own
    (even with rescaling off), while still pinning the peaks. Holds for every
    ordering strategy, since the ordering is only a permutation."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    predicted = _aggregate(
        raw, concurrency, preserve_minmax=True, rescale=False
    ).reconstructed

    np.testing.assert_allclose(raw.sum(), predicted.sum(), rtol=1e-9)
    assert (predicted.max() <= raw.max() + 1e-9).all()
    assert (predicted.min() >= raw.min() - 1e-9).all()


def test_concurrency_validation_errors():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # concurrency (non-independent) requires scope='cluster' — caught in config
    with pytest.raises(ValueError, match="scope='cluster'"):
        Distribution(scope="global", concurrency="medoid")

    # global scope reaches the algorithm guard for an explicit non-independent
    # method only if config validation is bypassed; verify the reference-name
    # check at aggregation time instead.
    with pytest.raises(ValueError, match="not one of the data columns"):
        _aggregate(raw, "reference", reference_attribute="NOPE")


def test_distribution_config_concurrency_roundtrip():
    dist = Distribution(concurrency="medoid")
    assert dist.to_dict()["concurrency"] == "medoid"
    assert Distribution.from_dict(dist.to_dict()).concurrency == "medoid"

    dist_ref = Distribution(reference_attribute="Load")
    assert dist_ref.to_dict()["reference_attribute"] == "Load"
    assert Distribution.from_dict(dist_ref.to_dict()).reference_attribute == "Load"

    with pytest.raises(ValueError, match="scope='cluster'"):
        Distribution(scope="global", concurrency="medoid")

    with pytest.raises(ValueError, match="requires reference_attribute"):
        Distribution(concurrency="reference")


@pytest.mark.parametrize("method", ["medoid", "reference"])
def test_concurrency_rejected_on_segment_representation(method):
    """concurrency / reference_attribute belong on the cluster representation,
    not the segment representation."""
    from tsam import SegmentConfig

    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    kw = {"reference_attribute": "Load"} if method == "reference" else {}
    with pytest.raises(ValueError, match="not supported for segment"):
        aggregate(
            raw,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical", representation="mean"),
            segments=SegmentConfig(
                n_segments=8,
                representation=Distribution(concurrency=method, **kw),
            ),
            preserve_column_means=False,
        )
