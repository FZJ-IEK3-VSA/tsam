# -*- coding: utf-8 -*-
"""Exact K-medoids clustering"""


import numpy as np

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted
import pyomo.environ as pyomo
import pyomo.opt as opt



class KMedoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    k-medoids class.

    Parameters
    ----------
    n_clusters: int, optional, default: 8
        How many medoids. Must be positive.
    distance_metric : string, optional, default: 'euclidean'
        What distance metric to use.
    timelimit: int, optional, default:  100
        Specify the time limit of the solver.
    threads: int, optional, default: 7
        Threads to use by the optimization solver.
    solver: str, optional, default: 'glpk'
    """

    def __init__(self, n_clusters=8, distance_metric='euclidean',
                 timelimit=100, threads=7, solver='glpk'):

        self.n_clusters = n_clusters

        self.distance_metric = distance_metric

        self.solver = solver

        self.timelimit = timelimit

        self.threads = threads

    def _check_init_args(self):

        # Check n_clusters
        if self.n_clusters is None or self.n_clusters <= 0 or \
                not isinstance(self.n_clusters, int):
            raise ValueError("n_clusters has to be nonnegative integer")

        # Check distance_metric
        if callable(self.distance_metric):
            self.distance_func = self.distance_metric
        elif self.distance_metric in PAIRWISE_DISTANCE_FUNCTIONS:
            self.distance_func = \
                PAIRWISE_DISTANCE_FUNCTIONS[self.distance_metric]
        else:
            raise ValueError("distance_metric needs to be " +
                             "callable or one of the " +
                             "following strings: " +
                             "{}".format(PAIRWISE_DISTANCE_FUNCTIONS.keys()) +
                             ". Instead, '{}' ".format(self.distance_metric) +
                             "was given.")

    def fit(self, X, y=None):
        """Fit K-Medoids to the provided data.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        Returns
        -------
        self
        """

        self._check_init_args()

        # check that the array is good and attempt to convert it to
        # Numpy array if possible
        X = self._check_array(X)

        # apply distance metric to get the distance matrix
        D = self.distance_func(X)

        # run exact optimization
        r_y, r_x, best_inertia = self._k_medoids_exact(D, self.n_clusters)


        labels_raw = r_x.argmax(axis=0)

        count = 0
        translator = {}
        cluster_centers_ = []
        for ix,val in enumerate(r_y):
            if val > 0:
                translator[ix] = count
                cluster_centers_.append( X[ix] )
                count += 1
        labels_ = []
        for label in labels_raw:
            labels_.append(translator[label])

        self.labels_ = labels_
        self.cluster_centers_ = cluster_centers_

        return self

    def _check_array(self, X):

        X = check_array(X)

        # Check that the number of clusters is less than or equal to
        # the number of samples
        if self.n_clusters > X.shape[0]:
            raise ValueError("The number of medoids " +
                             "({}) ".format(self.n_clusters) +
                             "must be larger than the number " +
                             "of samples ({})".format(X.shape[0]))

        return X

    def transform(self, X):
        """Transforms X to cluster-distance space.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Data to transform.
        Returns
        -------
        X_new : array, shape=(n_samples, n_clusters)
            X transformed in the new space.
        """

        check_is_fitted(self, "cluster_centers_")

        # Apply distance metric wrt. cluster centers (medoids),
        # and return these distances
        return self.distance_func(X, Y=self.cluster_centers_)

    def predict(self, X):

        check_is_fitted(self, "cluster_centers_")

        # Check that the array is good and attempt to convert it to
        # Numpy array if possible
        X = check_array(X)

        # Apply distance metric wrt. cluster centers (medoids)
        D = self.distance_func(X, Y=self.cluster_centers_)

        # Assign data points to clusters based on
        # which cluster assignment yields
        # the smallest distance
        labels = np.argmin(D, axis=1)

        return labels

    def inertia(self, X):

        # Map the original X to the distance-space
        Xt = self.transform(X)

        # Define inertia as the sum of the sample-distances
        # to closest cluster centers
        inertia = np.sum(np.min(Xt, axis=1))

        return inertia

    def _k_medoids_exact(self, distances, n_clusters):
        """
        Parameters
        ----------
        distances : int, required
            Pairwise distances between each row.
        n_clusters : int, required
            Number of clusters.
        """
        # Create model
        M=pyomo.ConcreteModel()

        # get distance matrix
        M.d = distances

        # set number of clusters
        M.no_k = n_clusters

        # Distances is a symmetrical matrix, extract its length
        length = distances.shape[0]

        # get indices
        M.i = [j for j in range(length)]
        M.j = [j for j in range(length)]

        # initialize vars
        M.z = pyomo.Var(M.i, M.j, within=pyomo.Binary)
        M.y = pyomo.Var(M.i, within=pyomo.Binary)


        # get objective
        def objRule(M):
            return sum(sum(M.d[i,j] * M.z[i,j] for j in M.j) for i in M.i)
        M.obj=pyomo.Objective(rule=objRule)

        # s.t.
        # Assign all candidates to clusters
        def candToClusterRule(M,j):
            return sum(M.z[i,j] for i in M.i) == 1
        M.candToClusterCon = pyomo.Constraint(M.j, rule=candToClusterRule)

        # no of clusters
        def noClustersRule(M):
            return sum(M.y[i] for i in M.i) == M.no_k
        M.noClustersCon = pyomo.Constraint(rule=noClustersRule)

        # cluster relation
        def clusterRelationRule(M,i,j):
            return M.z[i,j] <= M.y[i]
        M.clusterRelationCon = pyomo.Constraint(M.i,M.j, rule=clusterRelationRule)


        # create optimization problem
        optprob = opt.SolverFactory(self.solver)
        if self.solver =='gurobi':
            optprob.set_options("Threads=" + str(self.threads) +
                    " TimeLimit=" + str(self.timelimit))

        results = optprob.solve(M,tee=False)
        # check that it does not fail
        if self.solver=='gurobi' and results['Solver'][0]['Termination condition'].index == 11:
            print(results['Solver'][0]['Termination message'])
            return False
        elif self.solver=='gurobi' and not results['Solver'][0]['Termination condition'].index in [2,7,8,9,10]: # optimal
            raise ValueError(results['Solver'][0]['Termination message'])

        # Get results
        r_x = np.array([[round(M.z[i,j].value) for i in range(length)]
                                  for j in range(length)])

        r_y = np.array([round(M.y[j].value) for j in range(length)])

        r_obj = pyomo.value(M.obj)

        return (r_y, r_x.T, r_obj)
