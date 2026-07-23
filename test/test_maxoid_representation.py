"""Regression tests for ``maxoid_representation`` (issue #366).

Pins the *deliberate* global scope of the maxoid: it selects the cluster member
farthest from the whole dataset (a globally extreme period), which is asymmetric
to the within-cluster medoid.
"""

import numpy as np

from tsam.algorithms.representations import maxoid_representation, medoid_representation


def test_maxoid_uses_global_not_within_cluster_scope():
    # Cluster 0 = {0, 1, 2}; a distant cluster 1 = {-100} pulls the "farthest
    # from the whole dataset" member of cluster 0 to value 2, whereas a purely
    # within-cluster maxoid would pick value 0 (tie broken to the first index).
    candidates = np.array([[0.0], [1.0], [2.0], [-100.0]])
    cluster_order = np.array([0, 0, 0, 1])

    centers, indices = maxoid_representation(candidates, cluster_order)

    # cluster 0 representative is the globally extreme member (value 2, index 2)
    assert indices[0] == 2
    np.testing.assert_array_equal(centers[0], [2.0])
    # a within-cluster maxoid would instead have picked index 0 (value 0)
    assert indices[0] != 0


def test_medoid_is_within_cluster_central_member():
    candidates = np.array([[0.0], [1.0], [2.0], [-100.0]])
    cluster_order = np.array([0, 0, 0, 1])

    centers, indices = medoid_representation(candidates, cluster_order)

    # most central member of cluster 0 is value 1 (index 1)
    assert indices[0] == 1
    np.testing.assert_array_equal(centers[0], [1.0])
