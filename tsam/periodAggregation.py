# -*- coding: utf-8 -*-

import numpy as np
from tsam.representations import representations

def aggregatePeriods(candidates, n_clusters=8, n_iter=100, clusterMethod='k_means', solver='glpk',
                     representationMethod=None, representationDict=None, timeStepsPerPeriod=None):
    '''
    Clusters the data based on one of the cluster methods:
    'averaging', 'k_means', 'exact k_medoid' or 'hierarchical'

    :param candidates: Dissimilarity matrix where each row represents a candidate. required
    :type candidates: np.ndarray

    :param n_clusters: Number of aggregated cluster. optional (default: 8)
    :type n_clusters: integer

    :param n_iter: Only required for the number of starts of the k-mean algorithm. optional (default: 10)
    :type n_iter: integer

    :param clusterMethod: Chosen clustering algorithm. Possible values are
        'averaging','k_means','exact k_medoid' or 'hierarchical'. optional (default: 'k_means')
    :type clusterMethod: string
    '''

    # cluster the data
    if clusterMethod == 'averaging':
        n_sets = len(candidates)
        if n_sets % n_clusters == 0:
            cluster_size = int(n_sets / n_clusters)
            clusterOrder = [
                [n_cluster] *
                cluster_size for n_cluster in range(n_clusters)]
        else:
            cluster_size = int(n_sets / n_clusters)
            clusterOrder = [
                [n_cluster] *
                cluster_size for n_cluster in range(n_clusters)]
            clusterOrder.append([n_clusters - 1] *
                                int(n_sets - cluster_size * n_clusters))
        clusterOrder = np.hstack(np.array(clusterOrder))
        clusterCenters, clusterCenterIndices = representations(candidates, clusterOrder, default='meanRepresentation',
                                                               representationMethod=representationMethod,
                                                               representationDict=representationDict,
                                                               timeStepsPerPeriod=timeStepsPerPeriod)

    if clusterMethod == 'k_means':
        from sklearn.cluster import KMeans
        k_means = KMeans(
            n_clusters=n_clusters,
            max_iter=1000,
            n_init=n_iter,
            tol=1e-4)

        clusterOrder = k_means.fit_predict(candidates)
        # get with own mean representation to avoid numerical trouble caused by sklearn
        clusterCenters, clusterCenterIndices = representations(candidates, clusterOrder, default='meanRepresentation',
                                                               representationMethod=representationMethod,
                                                               representationDict=representationDict,
                                                               timeStepsPerPeriod=timeStepsPerPeriod)

    if clusterMethod == 'k_medoids':
        from tsam.utils.k_medoids_exact import KMedoids
        k_medoid = KMedoids(n_clusters=n_clusters, solver=solver)

        clusterOrder = k_medoid.fit_predict(candidates)
        clusterCenters, clusterCenterIndices = representations(candidates, clusterOrder, default='medoidRepresentation',
                                                               representationMethod=representationMethod,
                                                               representationDict=representationDict,
                                                               timeStepsPerPeriod=timeStepsPerPeriod)

    if clusterMethod == 'hierarchical' or clusterMethod == 'adjacent_periods':
        if n_clusters==1:
            clusterOrder=np.asarray([0]*len(candidates))
        else:
            from sklearn.cluster import AgglomerativeClustering
            if clusterMethod == 'hierarchical':
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage='ward')
            elif clusterMethod == 'adjacent_periods':
                adjacencyMatrix = np.eye(len(candidates), k=1) + np.eye(len(candidates), k=-1)
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters, linkage='ward', connectivity=adjacencyMatrix)
            clusterOrder = clustering.fit_predict(candidates)
        # represent hierarchical aggregation with medoid
        clusterCenters, clusterCenterIndices = representations(candidates, clusterOrder, default='medoidRepresentation',
                                                               representationMethod=representationMethod,
                                                               representationDict=representationDict,
                                                               timeStepsPerPeriod=timeStepsPerPeriod)

    return clusterCenters, clusterCenterIndices, clusterOrder



