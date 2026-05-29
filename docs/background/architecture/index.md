# Architecture

tsam is a Python library that reduces a long, multi-attribute time series to a small number of typical periods, so downstream optimization frameworks ([ETHOS.FINE](https://github.com/FZJ-IEK3-VSA/FINE), [flixopt](https://github.com/flixOpt/flixopt), [oemof](https://oemof.org/), ...) can stay tractable. Its architecture is documented as two C4 diagrams — one per page.

| Page | What it shows | When to read it |
|------|---------------|----------------|
| [System Context](context.md) | tsam in its environment: users, inputs, outputs, external systems | First read; to see what tsam does and does not own. |
| [Components](components.md) | Modules inside the `tsam` package and how they depend on each other | Onboarding as a contributor; deciding where new code belongs. |

These follow Simon Brown's [C4 model](https://c4model.com/) (Context → Containers → Components → Code), with two levels deliberately skipped:

- **Containers** is omitted — tsam is a single Python package, so there is only one container.
- **Code** is omitted — the auto-generated [API Reference](../../api/index.md) covers individual classes and functions in more detail than a hand-drawn diagram ever could.

For the *reasoning* behind architectural choices (rather than the shape of them), see [Decisions](../decisions/index.md). For how the aggregation pipeline processes data internally, see the [Pipeline Guide](pipeline_guide.md).
