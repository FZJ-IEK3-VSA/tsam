"""Exact K-maxoids clustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.random as rnd
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import PAIRWISE_DISTANCE_FUNCTIONS
from sklearn.utils import check_array

if TYPE_CHECKING:
    from collections.abc import Callable


class KMaxoids(BaseEstimator, ClusterMixin, TransformerMixin):
    """k-maxoids class.

    Args:
        n_clusters: How many maxoids. Must be positive.
        distance_metric: What distance metric to use. It can be either a
            self-defined callable or a string supported by scikit-learn,
            such as ``"euclidean"``.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        distance_metric: str | Callable = "euclidean",
    ) -> None:
        self.n_clusters = n_clusters

        self.distance_metric = distance_metric

    def _check_init_args(self) -> None:
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
                + f"{PAIRWISE_DISTANCE_FUNCTIONS.keys()}"
                + f". Instead, '{self.distance_metric}' "
                + "was given."
            )

    def fit(self, X: np.ndarray, y: object = None) -> KMaxoids:
        """Fit K-Maxoids to the provided data.

        Args:
            X: Data of shape (n_samples, n_features).
            y: Ignored; present for scikit-learn API compatibility.

        Returns:
            self.
        """

        self._check_init_args()

        # check that the array is good and attempt to convert it to
        # Numpy array if possible
        X = self._check_array(X)

        # apply distance metric to get the distance matrix (kept for potential debugging)
        _D = self.distance_func(X)

        # run mk-maxoids clustering
        self.cluster_centers_, self.labels_ = self.k_maxoids(X, self.n_clusters)

        return self

    def _check_array(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)

        # Check that the number of clusters is less than or equal to
        # the number of samples
        if self.n_clusters > X.shape[0]:
            raise ValueError(
                "The number of medoids "
                + f"({self.n_clusters}) "
                + "must be larger than the number "
                + f"of samples ({X.shape[0]})"
            )

        return X

    @staticmethod
    def _run_passes(X: np.ndarray, M: np.ndarray, n_passes: int) -> None:
        """Run the sequential maxoid-replacement passes, updating ``M`` in place.

        Same loop and per-sample decision as the naive implementation, but the
        three decision inputs (nearest maxoid, sample value, maxoid value) are
        read from arrays that only need recomputing when a replacement modifies
        ``M`` — a rare event. The recomputation uses the same
        ``np.sum((A - b) ** 2, axis=1)`` reduction as the naive per-sample
        code, so the decisions are bit-identical.
        """
        n = X.shape[0]
        rows = np.arange(n)
        D_all = np.stack([np.sum((X - m) ** 2, axis=1) for m in M], axis=1)
        d_med = np.stack([np.sum((M - m) ** 2, axis=1) for m in M], axis=1)
        np.fill_diagonal(d_med, 0.0)

        def refresh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            nearest = np.argmin(D_all, axis=1)
            zeroed = D_all.copy()
            zeroed[rows, nearest] = 0.0
            return nearest, zeroed.sum(axis=1), d_med.sum(axis=1)

        nearest_all, valx_all, valm_all = refresh()
        for _ in range(n_passes):
            for j in range(n):
                nearest = nearest_all[j]
                if valx_all[j] > valm_all[nearest]:
                    M[nearest] = X[j]
                    D_all[:, nearest] = np.sum((X - X[j]) ** 2, axis=1)
                    d = np.sum((M - X[j]) ** 2, axis=1)
                    d[nearest] = 0.0
                    d_med[nearest, :] = d
                    d_med[:, nearest] = d
                    nearest_all, valx_all, valm_all = refresh()

    def k_maxoids(
        self,
        X: np.ndarray,
        k: int,
        n_passes: int = 5,
        do_logarithmic: bool = False,
        n_init: int = 100,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        x_old = X
        n, _m = X.shape
        inertia_best = None

        for _ in range(n_init):
            inds = rnd.permutation(np.arange(n))

            X = X[inds]
            M = np.copy(X[:k])
            if not do_logarithmic:
                self._run_passes(X, M, n_passes)
            else:
                for _ in range(n_passes):
                    for j in range(n):
                        x = X[j]
                        D = np.sum((M - x) ** 2, axis=1)
                        nearest = np.argmin(D)  # type: ignore[assignment]
                        d = np.sum((M - M[nearest]) ** 2, axis=1)

                        D[nearest] = 1.0
                        d[nearest] = 1.0
                        valx = np.prod(D)
                        valm = np.prod(d)

                        if valx > valm:
                            M[nearest] = x

            d_temp = self.distance_func(x_old, Y=list(M))
            inertia_temp = np.sum(np.min(d_temp, axis=1))

            if inertia_best is None:
                m_final = M
                inertia_best = inertia_temp
            else:
                if inertia_temp < inertia_best:
                    m_final = M
                    inertia_best = inertia_temp

        D = self.distance_func(x_old, Y=list(m_final))

        I = np.argmin(D, axis=1)

        return list(m_final), I
