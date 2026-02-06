"""Normalization and denormalization of time series data."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tsam.pipeline.types import NormalizedData

MIN_WEIGHT = 1e-6


def normalize(
    data: pd.DataFrame,
    weights: dict[str, float] | None,
    normalize_column_means: bool,
) -> NormalizedData:
    """Sort columns, cast float, fit MinMaxScaler, normalize, normalize_column_means, apply weights.

    Replicates _preProcessTimeSeries lines 635-654 from the monolith.
    """
    # Sort columns alphabetically (monolith line 636)
    data = data.sort_index(axis=1)

    # Convert to float (monolith line 639)
    data = data.astype(float)

    # Fit MinMaxScaler and normalize (monolith lines 588-593)
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index,
    )

    # Store mean before normalize_column_means division (monolith line 595)
    normalized_mean = normalized.mean()

    # Divide by mean if normalize_column_means (monolith lines 596-597)
    if normalize_column_means:
        normalized = normalized / normalized_mean

    # Apply weights (monolith lines 644-654)
    if weights:
        for column in weights:
            w = weights[column]
            if w < MIN_WEIGHT:
                print(f'weight of "{column}" set to the minimal tolerable weighting')
                w = MIN_WEIGHT
            normalized[column] = normalized[column] * w

    return NormalizedData(
        values=normalized,
        scaler=scaler,
        normalized_mean=normalized_mean,
        original_data=data,
        weights=weights,
        normalize_column_means=normalize_column_means,
    )


def denormalize(
    df: pd.DataFrame,
    norm_data: NormalizedData,
    *,
    apply_weights: bool = True,
) -> pd.DataFrame:
    """Undo weights, undo normalization using stored scaler.

    Replicates _post_process_time_series lines 668-687 from the monolith.
    """
    result = df.copy()

    # Undo weights (monolith lines 672-676)
    if apply_weights and norm_data.weights:
        for column in norm_data.weights:
            result[column] = result[column] / norm_data.weights[column]

    # Undo normalize_column_means (monolith lines 619-620)
    if norm_data.normalize_column_means:
        result = result * norm_data.normalized_mean

    # Inverse transform using stored scaler (monolith lines 622-626)
    unnormalized = pd.DataFrame(
        norm_data.scaler.inverse_transform(result),
        columns=result.columns,
        index=result.index,
    )

    return unnormalized
