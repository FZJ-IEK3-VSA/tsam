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

print(result.cluster_center_indices)

# Get the start datetime for each cluster center period
# cluster_center_indices are 0-based period indices
period_hours = 24
cluster_center_start_indices = [
    idx * period_hours for idx in result.cluster_center_indices
]

# Select the dates for each cluster center period
cluster_center_dates = [
    raw.index[start_idx] for start_idx in cluster_center_start_indices
]
