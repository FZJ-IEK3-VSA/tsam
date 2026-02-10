"""Normalization and denormalization of time series data."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tsam.pipeline.types import NormalizedData


def normalize(
    data: pd.DataFrame,
    normalize_column_means: bool,
) -> NormalizedData:
    """Sort columns, cast float, fit MinMaxScaler, normalize, normalize_column_means.

    Weights are NOT applied here — they are used only for clustering distance.
    """
    # Sort columns alphabetically
    data = data.sort_index(axis=1)

    data = data.astype(float)

    # Fit MinMaxScaler and normalize
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index,
    )

    # Store mean before normalize_column_means division
    normalized_mean = normalized.mean()

    if normalize_column_means:
        normalized = normalized / normalized_mean

    return NormalizedData(
        values=normalized,
        scaler=scaler,
        normalized_mean=normalized_mean,
        original_data=data,
        normalize_column_means=normalize_column_means,
    )


def denormalize(
    df: pd.DataFrame,
    norm_data: NormalizedData,
) -> pd.DataFrame:
    """Undo normalization using stored scaler.

    No weight logic — weights are only used for clustering distance.
    """
    result = df.copy()

    if norm_data.normalize_column_means:
        result = result * norm_data.normalized_mean

    # Inverse transform using stored scaler
    unnormalized = pd.DataFrame(
        norm_data.scaler.inverse_transform(result),
        columns=result.columns,
        index=result.index,
    )

    return unnormalized
