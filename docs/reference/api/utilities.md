# Utilities

Supporting modules used across the package.

## Options

::: tsam.options
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Weights

::: tsam.weights
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Shared helpers

Generic helpers shared across the package: resolution inference, DatetimeIndex
(de)serialization for JSON round-trips, duration-string parsing, and weighted
aggregation of per-column metrics.

::: tsam.commons
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Plotting

Use `AttributeSpace` to overlay chronological paths, or `path_panels` to compare
one highlighted path per panel with shared axes. Pass a `colors` mapping keyed
by path name to keep reference days identifiable across figures.
Use `compare_series` to compare named time series or duration curves with a
separate physical scale for each attribute.

::: tsam.plot
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Comparison statistics

`series_statistics` reports minima, means, and maxima in physical units.
`aggregation_summary` compares the normalized accuracy and correlation errors
already calculated by aggregation results.

::: tsam.metrics
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Low-level aggregation

The clustering and representation primitives the pipeline dispatches to.

::: tsam.algorithms.clustering
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.algorithms.selection
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.algorithms.representations
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Algorithm backends

The concrete clustering, representation, and segmentation algorithms the
primitives above dispatch to — k-medoids/k-maxoids solvers, the duration-curve
representation, and constrained agglomerative segmentation.

::: tsam.algorithms.k_medoids_exact
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.algorithms.k_maxoids
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.algorithms.duration_representation
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.algorithms.concurrency
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.algorithms.segmentation
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
