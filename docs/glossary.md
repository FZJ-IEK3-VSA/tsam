# Glossary

Key concepts used in the ETHOS.TSAM API:

| Concept | Description |
|---------|-------------|
| **Period** | A fixed-length time window (e.g., 24 hours = 1 day). The original time series is divided into periods for clustering. |
| **Typical Period** | A representative period selected or computed to represent a cluster of similar periods. |
| **Cluster** | A group of similar original periods. Each cluster is represented by one typical period. |
| **Segment** | A subdivision within a period. Consecutive timesteps are grouped into segments to reduce temporal resolution. |
| **Timestep** | A single time point within a period (e.g., one hour in a 24-hour period). |
| **Duration Curve** | A sorted representation of values within a period (highest to lowest). Used with `use_duration_curves=True` to cluster by value distribution rather than temporal pattern. |
| `n_clusters` | Number of clusters to create. Each cluster is represented by one typical period. |
| `n_segments` | Number of segments per period. If not specified, equals timesteps per period (no segmentation). |
| `period_duration` | Length of each period. Accepts int/float (hours) or pandas Timedelta strings (e.g., `24`, `'24h'`, `'1d'`). |
| `temporal_resolution` | Time resolution of input data. Accepts float (hours) or pandas Timedelta strings (e.g., `1.0`, `'1h'`, `'15min'`). If not provided, inferred from the datetime index. |
| `cluster_assignments` | Array mapping each original period to its cluster index (0 to n_clusters-1). |
| `cluster_weights` | Dictionary mapping cluster index to occurrence count (how many original periods each cluster represents). |
| `segment_durations` | Nested tuple with duration (in timesteps) for each segment in each typical period. |
| `cluster_representatives` | MultiIndex DataFrame with aggregated data. Index levels are (cluster, timestep) or (cluster, segment) if segmented. |
