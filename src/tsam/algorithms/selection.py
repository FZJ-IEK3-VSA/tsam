"""Picking one member out of a group by a summed-distance score.

Kept apart from the representations that use it so that
`tsam.algorithms.concurrency` can share it without closing the
``representations -> duration_representation -> concurrency`` import loop.
"""

from __future__ import annotations

import numpy as np

# Distances are compared at this many decimals. Everything closer than that
# counts as a tie, which is what makes the choice reproducible.
_TIE_DECIMALS = 10


def deterministic_argmin(values: np.ndarray) -> int:
    """Index of the minimum, with a platform-stable tie-break.

    Rounds before comparing, so values equal in exact arithmetic but differing
    by floating-point noise — across BLAS builds, array layouts or the column
    order of the input — collapse to one value; ``argmin`` then returns the
    lowest such index identically everywhere.

    Ties are the normal case rather than an edge case wherever a group member
    is chosen by summed distance: every member of a two-member group is
    equidistant from the others, since both column sums of ``[[0, d], [d, 0]]``
    are ``d``. Left to a bare ``argmin`` the winner is decided by the order the
    squared differences happened to be accumulated in.
    """
    return int(np.argmin(np.round(values, _TIE_DECIMALS)))


def deterministic_argmax(values: np.ndarray) -> int:
    """Index of the maximum, with the tie-break of `deterministic_argmin`."""
    return int(np.argmax(np.round(values, _TIE_DECIMALS)))
