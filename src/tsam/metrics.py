"""Comparison tables for time series and aggregation results.

``series_statistics`` reports values in the input's physical units.
``aggregation_summary`` collects the normalized accuracy and correlation
metrics already provided by aggregation results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from tsam.result import AggregationResult


def _comparison_columns(
    series: dict[str, pd.DataFrame], columns: list[str] | None
) -> list[str]:
    """Validate finite numeric data and a common set of comparison columns."""
    if not series:
        raise ValueError("series is empty — pass at least one named data frame.")
    columns = list(next(iter(series.values())).columns) if columns is None else columns
    if not columns:
        raise ValueError("Select at least one column to compare.")
    for name, frame in series.items():
        missing = [column for column in columns if column not in frame]
        if frame.empty or missing:
            raise ValueError(f"Series {name!r} is empty or missing columns: {missing}.")
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"Series {name!r} contains non-finite values.")
    return columns


def series_statistics(
    series: dict[str, pd.DataFrame], *, columns: list[str] | None = None
) -> pd.DataFrame:
    """Compare minima, means and maxima in the input's physical units.

    Args:
        series: Named time series or profiles. Each row has equal weight;
            lengths may differ. For typical periods with unequal occurrence
            counts, pass ``result.reconstructed`` to include those weights.
        columns: Attributes to include, taken from the first frame if omitted.

    Returns:
        One row per named series, with two-level columns (attribute, statistic)
        for ``min``, ``mean`` and ``max``. Input order is retained.

    Raises:
        ValueError: If the mapping or a frame is empty, columns are missing,
            or the selected data contain non-finite values.
    """
    columns = _comparison_columns(series, columns)
    return pd.DataFrame.from_dict(
        {
            name: {
                (column, statistic): getattr(frame[column], statistic)()
                for column in columns
                for statistic in ("min", "mean", "max")
            }
            for name, frame in series.items()
        },
        orient="index",
    ).rename_axis(index="series", columns=["attribute", "statistic"])


def aggregation_summary(results: dict[str, AggregationResult]) -> pd.DataFrame:
    """Compare normalized reconstruction accuracy and cross-attribute errors.

    This collects the existing result metrics without recomputing or rounding
    them. Compare runs using the same original data and attribute weights:
    accuracy is evaluated in normalized units, not in physical units.

    Args:
        results: Named aggregation results in the desired display order.

    Returns:
        One row per run, with ``rmse``, ``rmse_duration``, ``correlation_error``
        and ``rank_correlation_error``. The RMSE columns use the result's
        weighted metrics. Correlation errors compare the original and
        reconstructed Pearson/Spearman matrices using the Frobenius norm;
        they are NaN for single-attribute data. Lower errors are better.

    Raises:
        ValueError: If no results are supplied.
    """
    if not results:
        raise ValueError("results is empty — pass at least one aggregation result.")
    return pd.DataFrame.from_dict(
        {
            name: {
                "rmse": result.accuracy.weighted_rmse,
                "rmse_duration": result.accuracy.weighted_rmse_duration,
                "correlation_error": result.concurrency.correlation_error,
                "rank_correlation_error": result.concurrency.rank_correlation_error,
            }
            for name, result in results.items()
        },
        orient="index",
    ).rename_axis("configuration")
