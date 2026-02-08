"""Reconstruction and accuracy computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tsam.pipeline.normalize import denormalize

if TYPE_CHECKING:
    from tsam.pipeline.types import NormalizedData, PeriodProfiles


def reconstruct(
    typical_periods: pd.DataFrame,
    cluster_order: list | np.ndarray,
    period_profiles: PeriodProfiles,
    norm_data: NormalizedData,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand typical periods via assignments, denormalize.

    Replicates predict_original_data (monolith lines 1394-1436).

    Returns (denormalized_predicted, normalized_predicted).
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
    original_len = len(norm_data.original_data)
    normalized_predicted = pd.DataFrame(
        clustered_data_df.values[:original_len],
        index=norm_data.original_data.index,
        columns=norm_data.original_data.columns,
    )

    # In the monolith, _post_process_time_series modified normalized_typical_periods
    # IN-PLACE: dividing out weights and multiplying by _normalized_mean.
    # Then predict_original_data used the already-modified data, so it called
    # _post_process_time_series with apply_weighting=False (weights already gone).
    #
    # In our pipeline there is no in-place modification, so normalized_predicted
    # still has weights baked in. We undo them here so that:
    # 1) denormalize gets weight-free data (matching the monolith's path)
    # 2) the returned normalized_predicted is in the same space as norm_data.values
    #    with weights removed, which compute_accuracy expects.
    if norm_data.weights:
        normalized_predicted = normalized_predicted.copy()
        for column in norm_data.weights:
            normalized_predicted[column] /= norm_data.weights[column]

    denormalized = denormalize(normalized_predicted, norm_data, apply_weights=False)

    return denormalized, normalized_predicted


def compute_accuracy(
    normalized_original: pd.DataFrame,
    normalized_predicted: pd.DataFrame,
    norm_data: NormalizedData,
) -> pd.DataFrame:
    """Compute RMSE, MAE, duration RMSE per column.

    Replicates accuracy_indicators (monolith lines 1485-1517).
    """
    weights = norm_data.weights

    indicator_raw: dict[str, dict] = {
        "RMSE": {},
        "RMSE_duration": {},
        "MAE": {},
    }

    for column in normalized_original.columns:
        if weights:
            orig_ts = normalized_original[column] / weights[column]
        else:
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

    return pd.DataFrame(indicator_raw)
