# -*- coding: utf-8 -*-
"""Exact K-maxoids clustering"""


import numpy as np
import numpy.random as rnd

from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.utils import check_array


class KMaxoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    k-maxoids class.

    :param n_clusters:  How many maxoids. Must be positive. optional, default: 8
    :type n_clusters: integer

    :param distance_metric: What distance metric to use. optional, default: 'euclidean'
    :type distance_metric: string
    """

    def __init__(
        self,
        n_clusters=8,
        distance_metric="euclidean",
        timelimit=100,
        threads=7,
        solver="cbc",
    ):

        self.n_clusters = n_clusters

        self.distance_metric = distance_metric

    def _check_init_args(self):

        # Check n_clusters
        if (
            self.n_clusters is None
            or self.n_clusters <= 0
            or not isinstance(self.n_clusters, int)
        ):
            raise ValueError("n_clusters has to be nonnegative integer")

        # Check distance_metric
        if callable(self.distance_metric):
            self.distance_func = self.distance_metric
        elif self.distance_metric in PAIRWISE_DISTANCE_FUNCTIONS:
            self.distance_func = PAIRWISE_DISTANCE_FUNCTIONS[self.distance_metric]
        else:
            raise ValueError(
                "distance_metric needs to be "
                + "callable or one of the "
                + "following strings: "
                + "{}".format(PAIRWISE_DISTANCE_FUNCTIONS.keys())
                + ". Instead, '{}' ".format(self.distance_metric)
                + "was given."
            )

    def fit(self, X, y=None):
        """Fit K-Maxoids to the provided data.

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
            raise ValueError(
                "The number of medoids "
                + "({}) ".format(self.n_clusters)
                + "must be larger than the number "
                + "of samples ({})".format(X.shape[0])
            )

        return X

    def k_maxoids(self, X, k, numpasses=5, doLogarithmic=False, n_init=100):

        X_old = X
        n, m = X.shape
        inertiaTempPrime = None

        for i in range(n_init):
            inds = rnd.permutation(np.arange(n))

            X = X[inds]
            M = np.copy(X[:k])
            for t in range(numpasses):
                for j in range(n):
                    x = X[j]
                    D = np.sum((M - x) ** 2, axis=1)
                    i = np.argmin(D)
                    d = np.sum((M - M[i]) ** 2, axis=1)

                    if doLogarithmic:
                        D[i] = 1.0
                        d[i] = 1.0
                        valx = np.prod(D)
                        valm = np.prod(d)
                    else:
                        D[i] = 0.0
                        d[i] = 0.0
                        valx = np.sum(D)
                        valm = np.sum(d)

                    if valx > valm:
                        M[i] = x

            dTemp = self.distance_func(X_old, Y=list(M))
            inertiaTemp = np.sum(np.min(dTemp, axis=1))

            if inertiaTempPrime is None:
                mFinal = M
                inertiaTempPrime = inertiaTemp
            else:
                if inertiaTemp < inertiaTempPrime:
                    mFinal = M
                    inertiaTempPrime = inertiaTemp

        D = self.distance_func(X_old, Y=list(mFinal))

        I = np.argmin(D, axis=1)

        return list(mFinal), I
