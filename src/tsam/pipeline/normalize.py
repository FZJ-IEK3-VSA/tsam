"""Normalization and denormalization of time series data."""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tsam.pipeline.types import NormalizedData


def normalize(
    data: pd.DataFrame,
    scale_by_column_means: bool,
) -> NormalizedData:
    """Scale every column to ``[0, 1]`` so no column dominates the distance.

    This is the first stage of the pipeline. It removes scale differences
    between columns before clustering and stores everything needed to invert
    the transformation later in `denormalize`.

    What happens:

    1. **Cast to float** (integer columns would otherwise truncate).
    2. **Min-max scale** each column to ``[0, 1]`` with scikit-learn's
       ``MinMaxScaler``. The fitted scaler is kept on the returned
       `NormalizedData` for inversion.
    3. **Column-mean normalization** (optional, ``scale_by_column_means=True``):
       divide each column by its post-scaling mean so columns contribute
       equally regardless of how full their ``[0, 1]`` range typically is.

    **Weights are NOT applied here.** Per-column weights affect only the
    clustering distance and are baked into a separate candidate copy during
    data preparation. The values produced here are unweighted and flow through
    every downstream stage (rescale, denormalize, reconstruct) untouched by
    weight compensation.

    Why it matters: without normalization, columns with larger numeric ranges
    dominate the clustering distance — a temperature column ranging 0–40 would
    overshadow a solar capacity factor ranging 0–1.

    Parameters
    ----------
    data
        Raw input time series, one column per attribute.
    scale_by_column_means
        If ``True``, additionally divide each scaled column by its mean so all
        columns carry equal weight irrespective of their typical level.

    Returns
    -------
    NormalizedData
        The normalized values plus the fitted ``MinMaxScaler``, the stored
        per-column mean, and the ``scale_by_column_means`` flag — the most
        widely read intermediate in the pipeline.

    See Also
    --------
    denormalize : Invert this transformation back to original units.
    """
    data = data.astype(float)

    # Fit MinMaxScaler and normalize
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(data),
        columns=data.columns,
        index=data.index,
    )

    # Store mean before scale_by_column_means division
    normalized_mean = normalized.mean()

    if scale_by_column_means:
        normalized = normalized / normalized_mean

    return NormalizedData(
        values=normalized,
        scaler=scaler,
        normalized_mean=normalized_mean,
        scale_by_column_means=scale_by_column_means,
    )


def denormalize(
    df: pd.DataFrame,
    norm_data: NormalizedData,
) -> pd.DataFrame:
    """Return values to original units by inverting `normalize`.

    Applied during the format-and-reconstruct phase to turn normalized typical
    periods back into the user's units. It undoes the two transformations from
    `normalize` in reverse order:

    1. Undo column-mean normalization (multiply by the stored per-column mean),
       if ``scale_by_column_means`` was set.
    2. Inverse the min-max scaling via the stored ``MinMaxScaler``.

    No weight removal is needed because weights were never baked into the data
    that reaches this stage.

    Parameters
    ----------
    df
        Normalized values to convert back to original units (typical periods,
        optionally segmented).
    norm_data
        The object returned by `normalize`, carrying the fitted scaler and
        stored mean.

    Returns
    -------
    pd.DataFrame
        The same data expressed in the original input units.

    See Also
    --------
    normalize : The forward transformation this inverts.
    """
    result = df.copy()

    if norm_data.scale_by_column_means:
        result = result * norm_data.normalized_mean

    # Inverse transform using stored scaler
    unnormalized = pd.DataFrame(
        norm_data.scaler.inverse_transform(result),
        columns=result.columns,
        index=result.index,
    )

    return unnormalized
