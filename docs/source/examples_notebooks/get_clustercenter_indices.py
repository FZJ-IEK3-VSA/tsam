import pandas as pd

import tsam
from tsam import ClusterConfig

raw = pd.read_csv("testdata.csv", index_col=0, parse_dates=True)

result = tsam.aggregate(
    raw,
    n_periods=8,
    period_hours=24,
    cluster=ClusterConfig(method="hierarchical"),
)

typical_periods = result.typical_periods
weights = result.cluster_weights

timesteps = [
    i for i in range(0, len(typical_periods.index.get_level_values("TimeStep")))
]

print(result.cluster_center_indices)

# get all index for every hour that in day of the clusterCenterIndices
days = [d for d in raw.index.dayofyear if d in result.cluster_center_indices]  # type: ignore[attr-defined]

# select the dates based on this
dates = raw.iloc[days].index

# TODO: Check index start (1 in dataframe, 0 in numpy -> should fit, otherwise
# days are offset one day
