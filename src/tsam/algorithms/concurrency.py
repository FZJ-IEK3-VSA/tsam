"""Time-axis ordering strategies for the distribution representation.

The distribution (duration-curve) representation reproduces each attribute's set
of values but must decide *how to lay those values out in time*. Every strategy
here produces the same per-attribute duration curve — and therefore the same
marginal distribution — and differs only in how much cross-attribute
**concurrency** (co-incidence in time, i.e. the empirical copula) it preserves.

Each strategy returns an ``order`` array of shape ``(n_attrs, n_timesteps)``:
the timestep position that each attribute's ascending duration-curve value is
placed at. ``final_repr[attr, order[attr]] = repr_values[attr]``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tsam.algorithms.selection import deterministic_argmin


def _first_principal_component(matrix: np.ndarray) -> np.ndarray:
    """First right singular vector of ``matrix``, with a canonical sign.

    A singular vector's sign is arbitrary and platform-dependent, and a flip
    would reverse any ordering derived from it. Forcing the largest-magnitude
    component to be positive makes the result deterministic.
    """
    _, _, vt = np.linalg.svd(matrix, full_matrices=False)
    pc: np.ndarray = vt[0]
    if pc[np.argmax(np.abs(pc))] < 0:
        pc = -pc
    return np.asarray(pc)


def _independent(
    cluster_data: np.ndarray,
    means: np.ndarray,
    repr_values: np.ndarray,
    reference_attribute_idx: int | None,
) -> np.ndarray:
    """Order each attribute by its own mean profile.

    Fits every attribute's distribution best, but the resulting time axes differ
    between attributes, so concurrency across attributes is lost. This is the
    default and matches the historical behaviour.
    """
    return means.argsort(axis=1, kind="stable")


def _reference(
    cluster_data: np.ndarray,
    means: np.ndarray,
    repr_values: np.ndarray,
    reference_attribute_idx: int | None,
) -> np.ndarray:
    """Broadcast the reference attribute's ordering to all attributes.

    A single temporal ordering is derived from the reference attribute's mean
    profile and applied to every attribute, so their co-incidence with the
    reference attribute is preserved.
    """
    if reference_attribute_idx is None:
        raise ValueError("The 'reference' strategy requires a reference attribute.")
    n_attrs, n_timesteps = means.shape
    ref_order = means[reference_attribute_idx].argsort(kind="stable")
    return np.broadcast_to(ref_order, (n_attrs, n_timesteps))


def _medoid(
    cluster_data: np.ndarray,
    means: np.ndarray,
    repr_values: np.ndarray,
    reference_attribute_idx: int | None,
) -> np.ndarray:
    """Order every attribute by the cluster medoid's own per-attribute ranks.

    The medoid is a real, physically consistent multivariate period, so
    reproducing its per-attribute rank pattern preserves its joint co-occurrence
    across all attribute pairs, while each attribute still takes the values of
    its own duration curve.
    """
    n_cands = cluster_data.shape[1]
    members = cluster_data.transpose(1, 0, 2).reshape(n_cands, -1)
    distances = np.linalg.norm(members[:, None, :] - members[None, :, :], axis=2)
    medoid_profile = cluster_data[:, deterministic_argmin(distances.sum(axis=0)), :]
    return np.round(medoid_profile, 10).argsort(axis=1, kind="stable")


def _consensus(
    cluster_data: np.ndarray,
    means: np.ndarray,
    repr_values: np.ndarray,
    reference_attribute_idx: int | None,
) -> np.ndarray:
    """Derive one shared ordering from the first principal component of the means.

    Standardises every attribute's mean profile, takes the first principal
    component of the profile matrix as a consensus time axis, and broadcasts it
    to all attributes. Balances concurrency across all attributes without needing
    a manual reference choice.
    """
    n_attrs, n_timesteps = means.shape
    std = means.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    z = (means - means.mean(axis=1, keepdims=True)) / std
    scores = z.T @ _first_principal_component(z.T)
    shared_order = np.round(scores, 10).argsort(kind="stable")
    return np.broadcast_to(shared_order, (n_attrs, n_timesteps))


def _assignment(
    cluster_data: np.ndarray,
    means: np.ndarray,
    repr_values: np.ndarray,
    reference_attribute_idx: int | None,
) -> np.ndarray:
    """Solve for the single shared ordering closest to the cluster mean profile.

    Assigns duration-curve ranks to timesteps to minimise the total squared
    deviation from the cluster's mean profile across all attributes
    simultaneously (a linear assignment problem). The cost of placing rank ``r``
    at timestep ``t`` is that squared deviation summed over attributes.
    """
    from scipy.optimize import linear_sum_assignment

    n_attrs, n_timesteps = means.shape
    cost = ((repr_values[:, :, None] - means[:, None, :]) ** 2).sum(axis=0)
    rank_idx, time_idx = linear_sum_assignment(cost)
    shared_order = np.empty(n_timesteps, dtype=int)
    shared_order[rank_idx] = time_idx
    return np.broadcast_to(shared_order, (n_attrs, n_timesteps))


_Strategy = Callable[[np.ndarray, np.ndarray, np.ndarray, int | None], np.ndarray]

_STRATEGIES: dict[str, _Strategy] = {
    "independent": _independent,
    "reference": _reference,
    "medoid": _medoid,
    "consensus": _consensus,
    "assignment": _assignment,
}

#: The ordering strategies supported by the distribution representation.
CONCURRENCY_METHODS: tuple[str, ...] = tuple(_STRATEGIES)


def compute_ordering(
    cluster_data: np.ndarray,
    means: np.ndarray,
    repr_values: np.ndarray,
    method: str | None,
    reference_attribute_idx: int | None,
) -> np.ndarray:
    """Derive the per-attribute time-axis ordering for one cluster.

    Args:
        cluster_data: Cluster members, shape ``(n_attrs, n_cands, n_timesteps)``.
        means: Per-attribute mean profile over the members, shape
            ``(n_attrs, n_timesteps)``, already rounded for stable tie-breaking.
        repr_values: Per-attribute ascending duration curve, shape
            ``(n_attrs, n_timesteps)``.
        method: One of :data:`CONCURRENCY_METHODS`. ``None`` falls back to
            ``"reference"`` when ``reference_attribute_idx`` is given, otherwise
            ``"independent"``.
        reference_attribute_idx: Attribute index used by the ``"reference"``
            strategy; ignored otherwise.

    Returns:
        The ordering array of shape ``(n_attrs, n_timesteps)``.
    """
    if method is None:
        method = "reference" if reference_attribute_idx is not None else "independent"
    try:
        strategy = _STRATEGIES[method]
    except KeyError:
        raise ValueError(
            f"Unknown concurrency method {method!r}. "
            f"Expected one of {CONCURRENCY_METHODS}."
        ) from None
    return strategy(cluster_data, means, repr_values, reference_attribute_idx)
