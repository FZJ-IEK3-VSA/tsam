import os
import time

import pandas as pd
import numpy as np

from tsam.utils.k_medoids_exact import KMedoids
from tsam.utils.k_medoids_contiguity import k_medoids_contiguity, _contiguity_to_graph

# similarity between node 0, 1 and 2
# 0===1
#  \ /
#   2
DISTANCES = np.array(
    [
        [0, 1, 4],
        [1, 0, 5],
        [4, 5, 0],
    ]
)

# add the adjacency between node 0, 1 and 2
# 0   1
#  \ /
#   2 -
adjacency = np.array(
    [
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
)


import networkx as nx


def test_node_cuts():
    """Tests the cutting of the nodes in the adjacency network"""
    G = _contiguity_to_graph(adjacency)
    cuts = list(nx.all_node_cuts(G))
    # only one cut expected
    assert len(cuts) == 1


def test_k_medoids_simple():
    """
    Checks that k-medoid brings the results as expected without the adjacency constraints
    """

    n_clusters = 2
    cluster_instance = KMedoids(n_clusters=n_clusters)
    r_y, r_x, r_obj = cluster_instance._k_medoids_exact(DISTANCES, n_clusters)

    labels_raw = r_x.argmax(axis=0)

    # check that node 0 and node 1 are in the same cluster
    assert labels_raw[0] == labels_raw[1]
    # check that node 1 and node 2 are in a different cluster
    assert labels_raw[1] != labels_raw[2]
    # check that the error/objective is the distance between node 0 and 1
    assert r_obj == 1


def test_k_medoids_simple_contiguity():
    """
    Checks if the adjacency constraint holds true
    """

    n_clusters = 2
    r_y, r_x, r_obj = k_medoids_contiguity(DISTANCES, n_clusters, adjacency)
    labels_raw = r_x.argmax(axis=0)
    # check that node 0 and node 2 are in the same cluster
    assert labels_raw[0] == labels_raw[2]
    # check that node 0 and node 1 are in a different cluster
    assert labels_raw[0] != labels_raw[1]
    # check that the error/objective is the distance between node 0 and 2
    assert r_obj == 4


if __name__ == "__main__":
    test_k_medoids_simple()
    test_k_medoids_simple_contiguity()
