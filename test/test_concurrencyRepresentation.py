"""Tests for the concurrencyMethod ordering strategies of the duration
representation.

The distribution representation builds each cluster's typical period from
per-attribute duration curves and then re-orders those values along a synthetic
time axis. ``concurrencyMethod`` selects how that ordering is derived. All
strategies preserve every attribute's marginal distribution (duration curve);
they differ only in how the attributes' concurrency (co-incidence in time)
across a common time axis is preserved:

- ``independent``: each attribute by its own mean profile (no concurrency);
- ``reference``:   one attribute's ordering, broadcast to all;
- ``medoid``:      the cluster medoid's per-attribute ranks (a real period);
- ``consensus``:   first principal component of all mean profiles;
- ``assignment``:  optimal single ordering vs the mean profile.
"""

import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV
from tsam import ClusterConfig, Distribution, aggregate
from tsam.utils.durationRepresentation import (
    CONCURRENCY_METHODS,
    durationRepresentation,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore::tsam.exceptions.LegacyAPIWarning"),
]

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
        durationRepresentation(
            _CANDIDATES,
            _CLUSTER_ORDER,
            True,
            _TS_PER_PERIOD,
            concurrencyMethod=method,
            **kwargs,
        )
    )


@pytest.mark.parametrize("method", CONCURRENCY_METHODS)
def test_all_strategies_preserve_marginals(method):
    """Every strategy yields the same per-attribute duration curve (same
    multiset of values per attribute and cluster); only the ordering differs."""
    kwargs = {"referenceAttributeIdx": 0} if method == "reference" else {}
    result = _run(method, **kwargs)
    independent = _run("independent")
    for attr in range(2):
        block = slice(attr * _TS_PER_PERIOD, (attr + 1) * _TS_PER_PERIOD)
        np.testing.assert_array_almost_equal(
            np.sort(result[0, block]),
            np.sort(independent[0, block]),
        )


def test_default_matches_independent():
    """No concurrencyMethod / no reference == 'independent' (back-compat)."""
    default = np.array(
        durationRepresentation(_CANDIDATES, _CLUSTER_ORDER, True, _TS_PER_PERIOD)
    )
    np.testing.assert_array_almost_equal(default, _run("independent"))


def test_reference_method_matches_referenceAttributeIdx():
    """concurrencyMethod='reference' reproduces the legacy referenceAttributeIdx."""
    legacy = np.array(
        durationRepresentation(
            _CANDIDATES, _CLUSTER_ORDER, True, _TS_PER_PERIOD, referenceAttributeIdx=1
        )
    )
    explicit = _run("reference", referenceAttributeIdx=1)
    np.testing.assert_array_almost_equal(legacy, explicit)


@pytest.mark.parametrize("method", ["reference", "consensus", "assignment"])
def test_shared_axis_methods_are_comonotone(method):
    """Single-shared-axis strategies place all attributes on one ordering, so
    within the period every attribute is co-monotone with attribute 0."""
    kwargs = {"referenceAttributeIdx": 0} if method == "reference" else {}
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
    # medoid = member minimising summed Euclidean distance to the others
    dist = np.linalg.norm(
        _CANDIDATES[:, None, :] - _CANDIDATES[None, :, :], axis=2
    )
    medoid = _CANDIDATES[np.argmin(dist.sum(axis=0))]
    for attr in range(2):
        block = slice(attr * _TS_PER_PERIOD, (attr + 1) * _TS_PER_PERIOD)
        medoid_order = np.argsort(medoid[block], kind="stable")
        # the attribute, ordered by the medoid's ranks, must be non-decreasing
        assert np.all(np.diff(result[0, block][medoid_order]) >= -1e-9)


def test_medoid_improves_concurrency_over_independent():
    """On real, multi-attribute data the medoid strategy preserves the
    cross-attribute correlation structure better than independent ordering,
    while keeping the same marginal (duration-curve) error."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    common = dict(
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        sortValues=False,
        clusterMethod="hierarchical",
        representationMethod="distributionRepresentation",
        rescaleClusterPeriods=False,
    )

    independent = tsam.TimeSeriesAggregation(
        raw, representationConcurrencyMethod="independent", **common
    )
    independent.createTypicalPeriods()

    medoid = tsam.TimeSeriesAggregation(
        raw, representationConcurrencyMethod="medoid", **common
    )
    medoid.createTypicalPeriods()

    # better concurrency
    assert (
        medoid.concurrencyIndicators()["corr_frobenius"]
        < independent.concurrencyIndicators()["corr_frobenius"]
    )
    # same marginal fit
    np.testing.assert_allclose(
        medoid.accuracyIndicators()["RMSE_duration"].values,
        independent.accuracyIndicators()["RMSE_duration"].values,
        rtol=1e-9,
        atol=1e-12,
    )


def test_medoid_with_minmax_preserves_envelope_and_integral():
    """medoid + distributionAndMinMax + rescale stays within the input envelope
    and preserves the integral."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        sortValues=False,
        clusterMethod="hierarchical",
        representationMethod="distributionAndMinMaxRepresentation",
        representationConcurrencyMethod="medoid",
        rescaleClusterPeriods=True,
    )
    aggregation.createTypicalPeriods()
    predicted = aggregation.predictOriginalData()

    assert (predicted.max() <= raw.max() + 1e-10).all()
    assert (predicted.min() >= raw.min() - 1e-10).all()
    np.testing.assert_allclose(raw.sum(), predicted.sum(), rtol=5e-3)


def test_concurrency_validation_errors():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # Unknown strategy name
    with pytest.raises(ValueError, match="one of"):
        tsam.TimeSeriesAggregation(
            raw,
            representationMethod="distributionRepresentation",
            representationConcurrencyMethod="nonsense",
        )

    # Non-distribution representation method
    with pytest.raises(ValueError, match="distribution-based"):
        tsam.TimeSeriesAggregation(
            raw,
            representationMethod="meanRepresentation",
            representationConcurrencyMethod="medoid",
        )

    # distributionPeriodWise must be True
    with pytest.raises(ValueError, match="distributionPeriodWise=True"):
        tsam.TimeSeriesAggregation(
            raw,
            representationMethod="distributionRepresentation",
            distributionPeriodWise=False,
            representationConcurrencyMethod="medoid",
        )

    # reference strategy needs an attribute
    with pytest.raises(ValueError, match="requires representationReferenceAttribute"):
        tsam.TimeSeriesAggregation(
            raw,
            representationMethod="distributionRepresentation",
            representationConcurrencyMethod="reference",
        )


def test_distribution_config_concurrency_roundtrip():
    dist = Distribution(concurrency="medoid")
    assert dist.to_dict()["concurrency"] == "medoid"
    assert Distribution.from_dict(dist.to_dict()).concurrency == "medoid"

    # non-independent concurrency only with scope='cluster'
    with pytest.raises(ValueError, match="scope='cluster'"):
        Distribution(scope="global", concurrency="medoid")

    # reference concurrency requires a reference attribute
    with pytest.raises(ValueError, match="requires reference_attribute"):
        Distribution(concurrency="reference")


def test_high_level_api_concurrency_matches_old_api():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    result = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(
            method="hierarchical",
            representation=Distribution(concurrency="medoid"),
        ),
        preserve_column_means=False,
    )

    old = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod="hierarchical",
        sortValues=False,
        representationMethod="distributionRepresentation",
        representationConcurrencyMethod="medoid",
        rescaleClusterPeriods=False,
    )
    oldTypPeriods = old.createTypicalPeriods()

    np.testing.assert_array_almost_equal(
        result.cluster_representatives.values, oldTypPeriods.values
    )


if __name__ == "__main__":
    for m in CONCURRENCY_METHODS:
        test_all_strategies_preserve_marginals(m)
    test_medoid_improves_concurrency_over_independent()
    print("ok")
