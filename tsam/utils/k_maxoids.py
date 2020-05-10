# -*- coding: utf-8 -*-
"""Exact K-maxoids clustering"""


import numpy as np
import numpy.random as rnd

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_is_fitted



class KMaxoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    k-maxoids class.

    :param n_clusters:  How many maxoids. Must be positive. optional, default: 8
    :type n_clusters: integer

    :param distance_metric: What distance metric to use. optional, default: 'euclidean'
    :type distance_metric: string

    :param timelimit: Specify the time limit of the solver. optional, default:  100
    :type timelimit: integer

    :param threads: Threads to use by the optimization solver. optional, default: 7
    :type threads: integer

    :param solver: optional, default: 'glpk'
    :type solver: string
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

        :param X: shape=(n_samples, n_features)
        :type X: array-like or sparse matrix

        :returns: self
        """

        self._check_init_args()

        # check that the array is good and attempt to convert it to
        # Numpy array if possible
        X = self._check_array(X)

        # apply distance metric to get the distance matrix
        D = self.distance_func(X)

        # run mk-maxoids clustering
        self.cluster_centers_, self.labels_ = self.k_maxoids(X, self.n_clusters)

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

        :param X: Data to transform. shape=(n_samples, n_features)
        :type X: array-like or sparse matrix,

        Returns
        -------
        :returns: **X_new** (array) -- X transformed in the new space. shape=(n_samples, n_clusters)
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

        # Apply distance metric wrt. cluster centers (maxoids)
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

    def k_maxoids(self, X, k, doLogarithmic=True):

        X_old=X
        n, m = X.shape
        inds = rnd.permutation(np.arange(n))

        X = X[inds]
        M = np.copy(X[:k])

        for j in range(n):
            x = X[j]
            D = np.sum((M - x) ** 2, axis=1)
            i = np.argmin(D)
            d = np.sum((M - M[i]) ** 2, axis=1)

            if doLogarithmic:
                D[i] = 1.
                d[i] = 1.
                valx = np.prod(D)
                valm = np.prod(d)
            else:
                D[i] = 0.
                d[i] = 0.
                valx = np.sum(D)
                valm = np.sum(d)

            if valx > valm:
                M[i] = x

        D = self.distance_func(X_old, Y=list(M))

        I = np.argmin(D, axis=1)

        return list(M), I