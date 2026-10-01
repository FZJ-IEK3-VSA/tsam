# Benchmark datasets

## fine_multiregional.csv.gz

Hourly year (8760 rows) x 40 columns: 5 profiles (wind onshore/offshore and
PV operation rates, electricity and hydrogen demand) x 8 regions, assembled
from the [FINE](https://github.com/FZJ-IEK3-VSA/FINE) (MIT license) example
`examples/03_Multi-regional_Energy_System_Workflow/InputData/SpatialData`
(files `maxOperationRateOnshore_el.xlsx`, `maxOperationRateOffshore_el.xlsx`,
`maxOperationRatePV_el.xlsx`, `electricityDemand_GWh_el.xlsx`,
`hydrogenDemand_GWh_hydrogen.xlsx`; columns prefixed with the profile name,
values rounded to 6 significant digits, synthetic 2020 datetime index).

Used by the `headline` benchmarks, which mirror FINE's
`aggregateTemporally()` defaults — the canonical downstream tsam workload.
