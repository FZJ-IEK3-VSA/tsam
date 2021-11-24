# -*- coding: utf-8 -*-

import numpy as np
import time
import pyomo.environ as pyomo
import pyomo.opt as opt
import networkx as nx
from tsam.utils.k_medoids_exact import (
    _setup_k_medoids,
    KMedoids,
    _solve_given_pyomo_model,
)


# class KMedoids_contiguity(KMedoids):


def k_medoids_contiguity(distances, n_clusters, adjacency, max_iter=500, solver="cbc"):
    """Declares a k-medoids model and iteratively adds cutting planes to hold on adjacency/contiguity

    The algorithm is based on: Oehrlein and Hauner (2017): A cutting-plane method for adjacency-constrained spatial aggregation
    """
    # First transform the network to a networkx instance which is required for cut generation
    G = _contiguity_to_graph(adjacency, distances=distances)

    # check if inputs are correct
    np.size(distances) == np.size(adjacency)

    # and test for connectivity
    if not nx.is_connected(G):
        raise ValueError("The give adjacency matrix is not connected.")

    # Initial setup of k medoids
    M = _setup_k_medoids(distances, n_clusters)

    M.adjacency = adjacency

    # Add constraintlist for the cuts later added
    M.cuts = pyomo.ConstraintList()

    # Loop over the relaxed k-medoids problem and add cuts until the problem fits
    _all_cluster_connected = False
    _iter = 0
    _cuts_added = []
    while not _all_cluster_connected and _iter < max_iter:
        # first solve instance
        t_presolve = time.time()
        print(str(_iter) + " iteration: Solving instance")
        r_x, r_y, obj = _solve_given_pyomo_model(M, solver=solver)
        t_aftersolve = time.time()
        print(
            "Total distance: "
            + str(obj)
            + " with solving time: "
            + str(t_aftersolve - t_presolve)
        )

        candidates, labels = np.where(r_x == 1)
        # claim that the resulting clusters are connected
        _all_cluster_connected = True
        _new_cuts_added = []
        for label in np.unique(labels):
            # extract the cluster
            cluster = G.subgraph(np.where(labels == label)[0])
            # Identify if the cluster is contineous, instead of validating the constraints such as Validi and Oehrlein.
            if not nx.is_connected(cluster):
                _all_cluster_connected = False
                # if not add contiguity constraints based on c-v (Oehrlein) o a-b (Validi) separators
                for candidate in cluster.nodes:
                    # It is not clear in Validi and Oehrlein, if cuts between all cluster candidates or just the center and the candidates shall be made. The latter one does not converge for the test system wherefore the first one is chosen.
                    for node in cluster.nodes:
                        # different to Validi et al. (2021) and Oehrlein and Haunert (2017), check first and just add continuity constraints for the not connected candidates  to increase performance
                        if nx.node_connectivity(cluster, node, candidate) == 0:
                            # check that the cut was not added so far for the cluster
                            if (label, candidate, node) not in _cuts_added:
                                # include the cut in the cut list
                                _new_cuts_added.append((label, candidate, node))
                                # Cuts to Separators - Appendix A Minimum-weight vertex separators (Oehrlein and Haunert, 2017)
                                # Validi uses an own cut generator and Oehrlein falls back to a Java library, here we use simple max flow cutting
                                # TODO: Check performance for large networks
                                cut_set = nx.minimum_node_cut(G, node, candidate)
                                # (Eq. 13 - Oehrlein and Haunert, 2017)
                                M.cuts.add(
                                    sum(M.z[u, node] for u in cut_set)
                                    >= M.z[candidate, node]
                                )
                            else:
                                raise ValueError(
                                    "Minimal cluster,candidate separation/minimum cut does not seem sufficient. Adding additional separators is could help."
                                )
        # Total cuts
        _cuts_added.extend(_new_cuts_added)
        _iter += 1
        t_afteradding = time.time()

        print(
            str(len(_new_cuts_added))
            + " contiguity constraints/cuts added, adding to a total number of "
            + str(len(_cuts_added))
            + " cuts within time: "
            + str(t_afteradding - t_aftersolve)
        )

    labels = np.where(r_x == 1)

    return (r_y, r_x.T, obj)


def _contiguity_to_graph(adjacency, distances=None):
    """Transforms a adjacency matrix to a networkx.Graph

    Args:
        adjacency (np.ndarray): 2-diimensional adjacency matrix
        distances (np.ndarray, optional): If provided, delivers the distances between the nodes. Defaults to None.

    Returns:
        nx.Graph: Graph with every index as node name.
    """
    rows, cols = np.where(adjacency == 1)
    G = nx.Graph()
    if distances is None:
        edges = zip(rows.tolist(), cols.tolist())
        G.add_edges_from(edges)
    else:
        normed_distances = distances / np.max(distances)
        weights = 1 - normed_distances
        if np.any(weights < 0) or np.any(weights > 1):
            raise ValueError("Weight calculation went wrong.")

        edge_weights = weights[rows, cols]
        edges = zip(rows.tolist(), cols.tolist(), edge_weights.tolist())
        G.add_weighted_edges_from(edges)
    return G
