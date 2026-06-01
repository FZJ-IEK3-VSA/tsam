# Pipeline internals

The aggregation runs in four phases orchestrated by `run_pipeline`. For the
conceptual walkthrough — what each phase does and how the stages fit together —
see the [Pipeline Guide](../background/architecture/pipeline_guide.md). This
page is the reference for the functions themselves.

## Orchestrator

`run_pipeline` and the four phase functions it calls.

::: tsam.pipeline.orchestrator
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3

## Stages

The pure transforms each phase calls, one module per concern.

::: tsam.pipeline.normalize
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.pipeline.periods
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.pipeline.clustering
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.pipeline.extremes
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.pipeline.rescale
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.pipeline.segmentation
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }
::: tsam.pipeline.accuracy
    options: { show_root_heading: true, show_root_toc_entry: false, heading_level: 3 }

## Internal types

The dataclasses passed between phases — each phase's milestone output.

::: tsam.pipeline.types
    options:
      show_root_heading: true
      show_root_toc_entry: false
      heading_level: 3
