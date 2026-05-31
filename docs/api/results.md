# Results

What [`aggregate`][tsam.aggregate] returns, and the transfer object used to
re-apply a clustering to new data.

::: tsam.result
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 2

`ClusteringResult` captures a fitted clustering. It is serializable
(`to_json` / `from_json`) and can be re-applied to new data via `apply()`.

::: tsam.config.ClusteringResult
    options:
      heading_level: 2
