"""Tests for the concurrency-preserving duration representation.

When a ``representationReferenceAttribute`` is given, the distribution-based
representations (``distributionRepresentation`` /
``distributionAndMinMaxRepresentation``) derive a single temporal ordering from
that attribute's mean profile and apply it to every attribute. This keeps all
attributes aligned on a common time axis (concurrency) while each attribute
still takes the values of its own fitted duration curve.
"""

import numpy as np
import pandas as pd
import pytest

import tsam.timeseriesaggregation as tsam
from conftest import TESTDATA_CSV
from tsam import ClusterConfig, Distribution, SegmentConfig, aggregate
from tsam.utils.durationRepresentation import durationRepresentation

pytestmark = [
    pytest.mark.filterwarnings("ignore::tsam.exceptions.LegacyAPIWarning"),
]


def test_durationRepresentation_referenceAttribute_unit():
    """Direct unit test of the shared-ordering logic.

    Two attributes, three time steps, one cluster of two identical candidates.
    Attribute 0 (the reference) rises over time; attribute 1 falls. With the
    reference ordering, attribute 1 must be reordered to co-move with the
    reference instead of keeping its own (opposite) shape.
    """
    timeStepsPerPeriod = 3
    candidates = np.array(
        [
            [1.0, 2.0, 3.0, 30.0, 20.0, 10.0],
            [1.0, 2.0, 3.0, 30.0, 20.0, 10.0],
        ]
    )
    clusterOrder = np.array([0, 0])

    # Default: every attribute keeps its own ordering -> attribute 1 stays
    # decreasing, i.e. concurrency between the two attributes is not enforced.
    default = np.array(
        durationRepresentation(candidates, clusterOrder, True, timeStepsPerPeriod)
    )
    np.testing.assert_array_almost_equal(default[0, 0:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(default[0, 3:6], [30.0, 20.0, 10.0])

    # Reference ordering (attribute 0): both attributes follow the reference's
    # increasing ordering, so attribute 1 is now also increasing.
    referenced = np.array(
        durationRepresentation(
            candidates,
            clusterOrder,
            True,
            timeStepsPerPeriod,
            referenceAttributeIdx=0,
        )
    )
    np.testing.assert_array_almost_equal(referenced[0, 0:3], [1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(referenced[0, 3:6], [10.0, 20.0, 30.0])

    # The reference attribute itself is identical in both cases.
    np.testing.assert_array_almost_equal(default[0, 0:3], referenced[0, 0:3])


def test_durationRepresentation_referenceAttribute_requires_periodWise():
    candidates = np.array([[1.0, 2.0, 3.0, 30.0, 20.0, 10.0]])
    clusterOrder = np.array([0])
    with pytest.raises(ValueError, match="distributionPeriodWise=True"):
        durationRepresentation(
            candidates, clusterOrder, False, 3, referenceAttributeIdx=0
        )


def _typical_periods_by_cluster(aggregation, typPeriods, column):
    """Yield the representative profile of ``column`` for every cluster."""
    for clusterNum in np.unique(aggregation.clusterOrder):
        yield typPeriods.loc[clusterNum, :].loc[:, column].values


def test_referenceAttribute_preserves_concurrency():
    """Every attribute must be co-monotone with the reference attribute.

    Sorting the representative period's time steps by the reference attribute
    must leave every other attribute non-decreasing.
    """
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    reference = "Load"

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        sortValues=False,
        clusterMethod="hierarchical",
        representationMethod="distributionRepresentation",
        representationReferenceAttribute=reference,
        rescaleClusterPeriods=False,
    )
    typPeriods = aggregation.createTypicalPeriods()

    for clusterNum in np.unique(aggregation.clusterOrder):
        ref_profile = typPeriods.loc[clusterNum, :].loc[:, reference].values
        ref_order = np.argsort(ref_profile, kind="stable")
        for column in raw.columns:
            profile = typPeriods.loc[clusterNum, :].loc[:, column].values
            ordered = profile[ref_order]
            # non-decreasing along the reference ordering (tolerant to ties)
            assert np.all(np.diff(ordered) >= -1e-9), (
                f"attribute {column} is not co-monotone with the reference "
                f"in cluster {clusterNum}"
            )


def test_referenceAttribute_keeps_reference_distribution():
    """The reference attribute's profile must match the plain distribution
    representation (it uses its own ordering in both cases), and the overall
    result must differ from it (other attributes get re-aligned)."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    reference = "Load"

    common = {
        "noTypicalPeriods": 8,
        "hoursPerPeriod": 24,
        "sortValues": False,
        "clusterMethod": "hierarchical",
        "representationMethod": "distributionRepresentation",
        "rescaleClusterPeriods": False,
    }

    plain = tsam.TimeSeriesAggregation(raw, **common)
    typPlain = plain.createTypicalPeriods()

    referenced = tsam.TimeSeriesAggregation(
        raw, representationReferenceAttribute=reference, **common
    )
    typRef = referenced.createTypicalPeriods()

    # Same clustering -> same cluster assignment, so profiles are comparable.
    np.testing.assert_array_equal(plain.clusterOrder, referenced.clusterOrder)

    # Reference attribute distribution is unchanged.
    np.testing.assert_array_almost_equal(
        typPlain.loc[:, reference].values,
        typRef.loc[:, reference].values,
    )

    # But the overall representation differs (other attributes are re-aligned).
    assert not np.allclose(typPlain.values, typRef.values)


def test_referenceAttribute_still_fits_distribution_better_than_mean():
    """The concurrency-preserving variant must still fit the duration curves
    markedly better than the plain mean representation."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    common = {
        "noTypicalPeriods": 8,
        "hoursPerPeriod": 24,
        "sortValues": False,
        "clusterMethod": "hierarchical",
        "rescaleClusterPeriods": False,
    }

    mean = tsam.TimeSeriesAggregation(
        raw, representationMethod="meanRepresentation", **common
    )
    mean.createTypicalPeriods()

    referenced = tsam.TimeSeriesAggregation(
        raw,
        representationMethod="distributionRepresentation",
        representationReferenceAttribute="Load",
        **common,
    )
    referenced.createTypicalPeriods()

    np.testing.assert_array_less(
        referenced.accuracyIndicators().loc[:, "RMSE_duration"].sum(),
        mean.accuracyIndicators().loc[:, "RMSE_duration"].sum(),
    )


def test_referenceAttribute_works_with_minmax():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    reference = "Load"

    aggregation = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        sortValues=False,
        clusterMethod="hierarchical",
        representationMethod="distributionAndMinMaxRepresentation",
        representationReferenceAttribute=reference,
        rescaleClusterPeriods=False,
    )
    typPeriods = aggregation.createTypicalPeriods()

    for clusterNum in np.unique(aggregation.clusterOrder):
        ref_profile = typPeriods.loc[clusterNum, :].loc[:, reference].values
        ref_order = np.argsort(ref_profile, kind="stable")
        for column in raw.columns:
            profile = typPeriods.loc[clusterNum, :].loc[:, column].values
            ordered = profile[ref_order]
            assert np.all(np.diff(ordered) >= -1e-9)


def test_referenceAttribute_validation_errors():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # Unknown attribute name
    with pytest.raises(ValueError, match="one of the"):
        tsam.TimeSeriesAggregation(
            raw,
            representationMethod="distributionRepresentation",
            representationReferenceAttribute="DoesNotExist",
        )

    # Non-distribution representation method
    with pytest.raises(ValueError, match="distribution-based"):
        tsam.TimeSeriesAggregation(
            raw,
            representationMethod="meanRepresentation",
            representationReferenceAttribute="Load",
        )

    # distributionPeriodWise must be True
    with pytest.raises(ValueError, match="distributionPeriodWise=True"):
        tsam.TimeSeriesAggregation(
            raw,
            representationMethod="distributionRepresentation",
            distributionPeriodWise=False,
            representationReferenceAttribute="Load",
        )


def test_distribution_config_reference_attribute_roundtrip():
    """The typed Distribution carries reference_attribute through (de)serialization."""
    dist = Distribution(reference_attribute="Load")
    assert dist.to_dict()["reference_attribute"] == "Load"
    assert Distribution.from_dict(dist.to_dict()).reference_attribute == "Load"

    # Only valid with scope="cluster".
    with pytest.raises(ValueError, match="scope='cluster'"):
        Distribution(scope="global", reference_attribute="Load")


def test_high_level_api_reference_attribute_matches_old_api():
    """aggregate(... Distribution(reference_attribute=...)) equals the old API."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    reference = "Load"

    result = aggregate(
        raw,
        n_clusters=8,
        period_duration=24,
        cluster=ClusterConfig(
            method="hierarchical",
            representation=Distribution(reference_attribute=reference),
        ),
        preserve_column_means=False,
    )
    newTypPeriods = result.cluster_representatives

    old = tsam.TimeSeriesAggregation(
        raw,
        noTypicalPeriods=8,
        hoursPerPeriod=24,
        clusterMethod="hierarchical",
        sortValues=False,
        representationMethod="distributionRepresentation",
        representationReferenceAttribute=reference,
        rescaleClusterPeriods=False,
    )
    oldTypPeriods = old.createTypicalPeriods()

    np.testing.assert_array_almost_equal(newTypPeriods.values, oldTypPeriods.values)

    # And the concurrency property holds end-to-end through the high-level API.
    for clusterNum in np.unique(old.clusterOrder):
        ref_order = np.argsort(
            oldTypPeriods.loc[clusterNum, :].loc[:, reference].values, kind="stable"
        )
        for column in raw.columns:
            ordered = oldTypPeriods.loc[clusterNum, :].loc[:, column].values[ref_order]
            assert np.all(np.diff(ordered) >= -1e-9)


def test_high_level_api_reference_attribute_rejected_for_segments():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    with pytest.raises(ValueError, match="segment representations"):
        aggregate(
            raw,
            n_clusters=8,
            period_duration=24,
            cluster=ClusterConfig(method="hierarchical"),
            segments=SegmentConfig(
                n_segments=8,
                representation=Distribution(reference_attribute="Load"),
            ),
        )


if __name__ == "__main__":
    test_durationRepresentation_referenceAttribute_unit()
    test_referenceAttribute_preserves_concurrency()
    test_referenceAttribute_keeps_reference_distribution()
