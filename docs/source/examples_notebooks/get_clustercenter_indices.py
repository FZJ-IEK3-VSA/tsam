import pandas as pd

import tsam.timeseriesaggregation as tsam

raw = pd.read_csv("testdata.csv", index_col=0, parse_dates=True)

aggregation = tsam.TimeSeriesAggregation(
    raw, noTypicalPeriods=8, hoursPerPeriod=24, clusterMethod="hierarchical"
)
df = aggregation.createTypicalPeriods()
weights = aggregation.clusterPeriodNoOccur

timesteps = [i for i in range(0, len(df.index.get_level_values("TimeStep")))]

print(aggregation.clusterCenterIndices)

# get all index for every hour that in day of the clusterCenterIndices
days = [d for d in raw.index.dayofyear if d in aggregation.clusterCenterIndices]  # type: ignore[attr-defined]

# select the dates based on this
dates = raw.iloc[days].index

# TODO: Check index start (1 in dataframe, 0 in numpy -> should fit, otherwise
# days are offset one day
