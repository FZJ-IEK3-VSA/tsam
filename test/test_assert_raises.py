import copy

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, ExtremeConfig, SegmentConfig, aggregate

# NOTE: The legacy ``TimeSeriesAggregation`` constructor performed runtime
# type-checks on a number of scalar/boolean keyword arguments (e.g.
# ``evalSumPeriods``/``sortValues``/``sameMean``/``rescaleClusterPeriods`` "has
# to be boolean", and ``predefClusterOrder``/``predefClusterCenterIndices`` "has
# to be an array or list").  The new ``aggregate`` API expresses these through
# typed config objects / dataclass fields, so there is no equivalent runtime
# validation to assert against — those cases are intentionally not ported here.
#
# The legacy "non-datetime index requires temporal_resolution" error is also not
# asserted: v4 intentionally defaults to hourly resolution in that case (the
# common test data has a plain string index), documented in the v3->v4 migration
# guide.


def test_assert_raises():
    raw = pd.read_csv(TESTDATA_CSV, index_col=0)

    # wrong time series type
    with pytest.raises(TypeError, match="data must be a pandas DataFrame"):
        aggregate("erroneousTimeSeries", n_clusters=8)

    # extreme-period columns that do not occur in the data
    for cfg in (
        ExtremeConfig(method="append", min_value=["erroneousAttribute"]),
        ExtremeConfig(method="append", max_value=["erroneousAttribute"]),
        ExtremeConfig(method="append", min_period=["erroneousAttribute"]),
        ExtremeConfig(method="append", max_period=["erroneousAttribute"]),
    ):
        with pytest.raises(
            ValueError, match="Extreme period columns not found in data"
        ):
            aggregate(raw, n_clusters=8, extremes=cfg)

    # erroneous temporal_resolution argument
    with pytest.raises(ValueError, match="temporal_resolution"):
        aggregate(raw, n_clusters=8, temporal_resolution="erroneousResolution")

    # erroneous period_duration argument
    with pytest.raises(TypeError, match="period_duration must be"):
        aggregate(raw, n_clusters=8, period_duration=None)

    # erroneous n_clusters argument
    with pytest.raises(ValueError, match="n_clusters must be a positive integer"):
        aggregate(raw, n_clusters=None)

    # non-integer number of time steps per period
    with pytest.raises(
        ValueError,
        match="does not result in an integer number of time steps per period",
    ):
        aggregate(raw, n_clusters=8, period_duration=23, temporal_resolution=2)

    # number of segments per period higher than the number of time steps per period
    with pytest.raises(ValueError, match="cannot exceed timesteps per period"):
        aggregate(raw, n_clusters=8, segments=SegmentConfig(n_segments=25))

    # erroneous cluster method
    with pytest.raises(ValueError, match="Unknown cluster_method"):
        aggregate(
            raw, n_clusters=8, cluster=ClusterConfig(method="erroneousClusterMethod")
        )

    # erroneous representation method
    with pytest.raises(ValueError, match="Unknown representation method"):
        aggregate(
            raw,
            n_clusters=8,
            cluster=ClusterConfig(representation="erroneousRepresentationMethod"),
        )

    # erroneous extreme period method
    with pytest.raises(ValueError, match="Unknown extreme period method"):
        aggregate(
            raw,
            n_clusters=8,
            extremes=ExtremeConfig(
                method="erroneousExtremePeriodMethod", max_value=["Load"]
            ),
        )

    # data frame containing NaN values
    rawNan = copy.deepcopy(raw)
    rawNan.iloc[10, :] = np.nan
    with pytest.raises(
        ValueError,
        match="Pre processed data includes NaN",
    ):
        aggregate(rawNan, n_clusters=8)


if __name__ == "__main__":
    test_assert_raises()
