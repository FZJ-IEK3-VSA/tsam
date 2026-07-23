"""Reconstruction and accuracy computation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tsam.pipeline.normalize import denormalize

if TYPE_CHECKING:
    from tsam.pipeline.types import NormalizedData, PeriodProfiles


def reconstruct(
    typical_periods: pd.DataFrame,
    cluster_order: np.ndarray,
    period_profiles: PeriodProfiles,
    norm_data: NormalizedData,
    original_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand typical periods back into a full-length time series.

    Replaces each original period with its assigned cluster representative,
    producing a series the same shape as the input. Returns both the
    denormalized series (original units) and the normalized series (kept for
    accuracy computation).

    A bounds check runs just before this stage in the orchestrator: any column
    whose typical-period max or min exceeds the original data's range beyond
    ``numerical_tolerance`` triggers a warning. That can happen with
    distribution representations or aggressive rescaling.

    Args:
        typical_periods: The representative periods (normalized, optionally
            segmented).
        cluster_order: Per-period cluster assignment mapping originals to
            representatives.
        period_profiles: Profile metadata used to reshape back to the original
            layout.
        norm_data: Normalization state, for converting back to original units.
        original_data: Original input, defining the output length, index, and
            columns.

    Returns:
        ``(denormalized_predicted, normalized_predicted)`` — the reconstructed
        series in original units and in normalized units.

    Note:
        See `compute_accuracy`, which scores the normalized reconstruction
        against the input, and `denormalize`, used internally to return values
        to original units.
    """
    # Unstack once, then use vectorized indexing to select periods by cluster order
    typical_unstacked = typical_periods.unstack()
    reconstructed = typical_unstacked.loc[list(cluster_order)].values

    # Back in matrix form
    clustered_data_df = pd.DataFrame(
        reconstructed,
        columns=period_profiles.column_index,
        index=period_profiles.profiles_dataframe.index,
    )
    clustered_data_df = clustered_data_df.stack(future_stack=True, level="TimeStep")  # type: ignore[assignment]

    # Trim to original data length
    original_len = len(original_data)
    normalized_predicted = pd.DataFrame(
        clustered_data_df.values[:original_len],
        index=original_data.index,
        columns=original_data.columns,
    )

    denormalized = denormalize(normalized_predicted, norm_data)

    return denormalized, normalized_predicted


def compute_accuracy(
    normalized_original: pd.DataFrame,
    normalized_predicted: pd.DataFrame,
) -> pd.DataFrame:
    """Score the reconstruction against the original, per column.

    Both inputs are unweighted normalized data, so they are compared directly.
    On the user-facing result these metrics are computed lazily (a
    ``cached_property``) when first accessed.

    Metrics per column:

    | Metric | Description |
    |---|---|
    | RMSE | Root mean square error. |
    | MAE | Mean absolute error. |
    | RMSE (duration) | RMSE on sorted (duration-curve) values — measures distribution fit. |

    Args:
        normalized_original: The original series in normalized units.
        normalized_predicted: The reconstructed series in normalized units
            (from `reconstruct`).

    Returns:
        One row per column with ``RMSE``, ``RMSE_duration`` and ``MAE``.

    Note:
        `reconstruct` produces the predicted series scored here.
    """
    indicator_raw: dict[str, dict] = {
        "RMSE": {},
        "RMSE_duration": {},
        "MAE": {},
    }

    for column in normalized_original.columns:
        orig_ts = normalized_original[column]
        pred_ts = normalized_predicted[column]

        indicator_raw["RMSE"][column] = np.sqrt(mean_squared_error(orig_ts, pred_ts))
        indicator_raw["RMSE_duration"][column] = np.sqrt(
            mean_squared_error(
                orig_ts.sort_values(ascending=False).reset_index(drop=True),
                pred_ts.sort_values(ascending=False).reset_index(drop=True),
            )
        )
        indicator_raw["MAE"][column] = mean_absolute_error(orig_ts, pred_ts)

    return cast("pd.DataFrame", pd.DataFrame(indicator_raw))


def compute_concurrency(
    normalized_original: pd.DataFrame,
    normalized_predicted: pd.DataFrame,
) -> pd.Series:
    """Score how well cross-attribute concurrency (co-incidence) is preserved.

    While `compute_accuracy` measures per-attribute error (including the
    duration-curve / marginal-distribution fit), this measures the *between*
    -attribute structure that the distribution representation can destroy: it
    compares the cross-attribute correlation matrix of the reconstructed series
    with that of the original. Lower is better.

    Parameters
    ----------
    normalized_original
        The original series in normalized units.
    normalized_predicted
        The reconstructed series in normalized units (from `reconstruct`).

    Returns
    -------
    pd.Series
        ``correlation_error`` — Frobenius norm of the difference between the
        Pearson correlation matrices of the original and reconstructed
        attributes; ``rank_correlation_error`` — the same for the Spearman
        rank-correlation matrices (a copula proxy, invariant to monotone marginal
        changes). Lower is better; both are ``NaN`` when fewer than two
        attributes are present.

    See Also
    --------
    compute_accuracy : Per-attribute (marginal) error metrics.
    """
    pred = normalized_predicted[normalized_original.columns]

    if len(normalized_original.columns) < 2:
        return cast(
            "pd.Series",
            pd.Series({"correlation_error": np.nan, "rank_correlation_error": np.nan}),
        )

    def _frob(method: Literal["pearson", "spearman"]) -> float:
        diff = normalized_original.corr(method=method) - pred.corr(method=method)
        return float(np.sqrt(np.square(diff.values).sum()))

    return cast(
        "pd.Series",
        pd.Series(
            {
                "correlation_error": _frob("pearson"),
                "rank_correlation_error": _frob("spearman"),
            }
        ),
    )
