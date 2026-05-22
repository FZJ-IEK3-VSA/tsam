# Architecture

tsam is a Python library that reduces a long, multi-attribute time series to a small number of typical periods, so downstream optimization frameworks ([ETHOS.FINE](https://github.com/FZJ-IEK3-VSA/FINE), [flixopt](https://github.com/flixOpt/flixopt), [oemof](https://oemof.org/), ...) can stay tractable. Its architecture is documented at three altitudes — each page below contains *one* diagram and explains only that one.

| Page | What it shows | When to read it |
|------|---------------|----------------|
| [System Context](context.md) | tsam in its environment: users, inputs, outputs, external systems | First read; to see what tsam does and does not own. |
| [Components](components.md) | Modules inside the `tsam` package and how they depend on each other | Onboarding as a contributor; deciding where new code belongs. |
| [Pipeline Data Flow](pipeline-dataflow.md) | What flows between aggregation stages, and what shape the data has at each boundary | Debugging or extending the aggregation pipeline. |

The three views follow Simon Brown's [C4 model](https://c4model.com/) (Context → Containers → Components → Code), with two levels deliberately skipped:

- **Containers** is omitted — tsam is a single Python package, so there is only one container.
- **Code** is omitted — the auto-generated [API Reference](../../api/tsam/api.md) covers individual classes and functions in more detail than a hand-drawn diagram ever could.

For the *reasoning* behind architectural choices (rather than the shape of them), see [Decisions](../decisions/index.md).
