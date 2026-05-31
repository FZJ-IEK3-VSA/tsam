# Utilities

Supporting modules used across the package.

## Options

::: tsam.options
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Weights

::: tsam.weights
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Plotting

::: tsam.plot
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Low-level aggregation

The clustering and representation primitives the pipeline dispatches to.

::: tsam.period_aggregation
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.representations
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Algorithm backends

The concrete clustering, representation, and segmentation algorithms the
primitives above dispatch to — k-medoids/k-maxoids solvers, the duration-curve
representation, and constrained agglomerative segmentation.

::: tsam.utils.k_medoids_exact
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.utils.k_maxoids
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.utils.duration_representation
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.utils.segmentation
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
