"""Medoid and maxoid selection must not depend on floating-point rounding.

Both pick a cluster member by the extreme of a summed-distance array. Ties in
that array are common rather than exotic: every member of a two-member group is
equidistant from the others, since both column sums of ``[[0, d], [d, 0]]`` are
``d``. A bare ``argmin``/``argmax`` then resolves the tie by whatever order the
squared differences happened to be accumulated in, which changes with the column
order of the input or with the BLAS build.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, aggregate
from tsam.algorithms.representations import (
    maxoid_representation,
    medoid_representation,
)


def test_medoid_of_a_two_member_group_takes_the_earlier_member():
    """Both members are equidistant, so the earlier index is the stable choice."""
    candidates = np.array([[0.0, 1.0], [1.0, 0.0]])
    _, indices = medoid_representation(candidates, np.array([0, 0]))
    assert indices == [0]


def test_maxoid_of_a_two_member_group_takes_the_earlier_member():
    candidates = np.array([[0.0, 1.0], [1.0, 0.0]])
    _, indices = maxoid_representation(candidates, np.array([0, 0]))
    assert indices == [0]


@pytest.mark.parametrize("representation", ["medoid", "maxoid"])
def test_selection_survives_a_one_ulp_perturbation(representation):
    """A difference at the last bit must not flip the representative.

    This is the failure that made `segmentation_medoid` differ between v3 and
    v4: the sums differed by 5.5e-16 with opposite signs on the two versions.
    """
    represent = {
        "medoid": medoid_representation,
        "maxoid": maxoid_representation,
    }[representation]

    base = np.array([[0.0, 1.0], [1.0, 0.0]])
    order = np.array([0, 0])
    nudged = base.copy()
    nudged[1] = np.nextafter(nudged[1], 1e9)

    assert represent(base, order)[1] == represent(nudged, order)[1] == [0]


@pytest.mark.parametrize("representation", ["medoid", "maxoid"])
def test_selection_is_independent_of_column_order(representation):
    """Reordering columns cannot change which period represents a cluster."""
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)
    cluster = ClusterConfig(method="hierarchical", representation=representation)
    kwargs = {"n_clusters": 8, "period_duration": 24, "cluster": cluster}

    as_given = aggregate(raw, **kwargs)
    alphabetical = aggregate(raw[sorted(raw.columns)], **kwargs)

    assert list(as_given.cluster_assignments) == list(alphabetical.cluster_assignments)
    assert (
        as_given.clustering.cluster_centers == alphabetical.clustering.cluster_centers
    )
