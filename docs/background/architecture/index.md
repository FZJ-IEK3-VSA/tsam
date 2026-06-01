# Architecture

tsam is a Python library that reduces a long, multi-attribute time series to a small number of typical periods, so downstream optimization frameworks ([ETHOS.FINE](https://github.com/FZJ-IEK3-VSA/FINE), [flixopt](https://github.com/flixOpt/flixopt), [oemof](https://oemof.org/), ...) can stay tractable. Its architecture is documented across two pages.

| Page | What it shows | When to read it |
|------|---------------|----------------|
| [System Context](context.md) | tsam in its environment: users, inputs, outputs, external systems | First read; to see what tsam does and does not own. |
| [Pipeline Guide](pipeline_guide.md) | One diagram of the modules, the public/internal split, and the four-phase `aggregate()` data flow, plus a phase-by-phase walkthrough and module map | Onboarding as a contributor; understanding or modifying the aggregation flow. |

These follow Simon Brown's [C4 model](https://c4model.com/) (Context → Containers → Components → Code):

- **System Context** is the [System Context](context.md) page.
- **Containers** is omitted — tsam is a single Python package, so there is only one container.
- **Components** — the module structure — is the architecture diagram on the [Pipeline Guide](pipeline_guide.md), which also overlays the data flow.
- **Code** is omitted — the auto-generated [API Reference](../../api/index.md) covers individual classes and functions in more detail than a hand-drawn diagram ever could.

For the *reasoning* behind architectural choices (rather than the shape of them), see [Decisions](../decisions/index.md).
