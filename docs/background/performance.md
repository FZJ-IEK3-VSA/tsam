# Performance & Scaling

`tsam.aggregate` is fast: a full year of hourly data becomes a handful of typical
periods in about **0.02 s**, and it stays quick at fine resolution or with
thousands of parallel series. The one decision that actually costs you is the
**clustering method** — and even that depends on how you have set things up.

<iframe src="../../assets/benchmarks/method_scaling.html" width="100%" height="470" frameborder="0"></iframe>

One figure, one year of data, aggregated from 4-hourly down to 5-minute
resolution — for every clustering method, at daily (365 periods) and weekly
(52 periods):

- **Fast, even at scale.** `averaging`, `contiguous` and `hierarchical` stay well
  under a second — even at 5-minute resolution (~100 000 timesteps), and ~0.75 s
  at 1000 series.
- **Methods differ by orders of magnitude.** `kmeans` runs ~10× the cheap methods;
  the exact-MILP `kmedoids` and heuristic `kmaxoids` are heavier still.
- **And that gap depends on your setup.** `kmedoids` is only tractable on 365
  *daily* periods right at hourly resolution, yet comfortable across every
  *weekly* one (just 52 periods). Which method is affordable moves with the period
  length and resolution — it is not fixed. (A method's line stops where the
  benchmark suite dropped it — see the note below.)

!!! note "A missing point is a benchmark cutoff, not a usability limit"
    To keep the suite quick to re-run, it only measures cases that finish in
    roughly a second and skips the ones that would take tens of seconds to
    minutes — so those methods simply have no dot here. That threshold is ours,
    not yours. An aggregation that takes 20 s, or even a few minutes, is often
    perfectly fine for a workflow you run occasionally: if it is a one-off or a
    monthly job, judge it against *your* time budget, not against whether it
    appears on this chart. The gaps mark where a method stops being *cheap*, not
    where it stops being *usable*.

**Why it stays fast:** cost tracks the number of **periods** — the clustering
candidates — which for a one-year horizon is ~365 (daily) or ~52 (weekly),
*fixed* regardless of resolution or column count. Refining the resolution or
adding series only widens each period's feature vector, adding work roughly
linearly, never a clustering blow-up. The remaining knobs — extremes, weights,
segmentation — stay within ~2× of the baseline.

**Rule of thumb: default to `hierarchical`.** It is deterministic and stays fast
at every scale here. Reach for `kmedoids` or `kmaxoids` only when the period count
is small and their exact representativeness matters.

## Representation is cheap

Given hierarchical clustering, the *representation* you pick — plain `mean` /
`medoid`, duration-preserving `distribution`, concurrency-preserving orderings, or
`minmax_mean` — barely changes the runtime: all of them scale together, within a
small factor of each other.

<iframe src="../../assets/benchmarks/representation_scaling.html" width="100%" height="470" frameborder="0"></iframe>

## Reproducing

```bash
uv pip install -e ".[develop]"
pytest benchmarks/ --benchmark-only --benchmark-save=headline   # a few minutes
python benchmarks/make_docs_figures.py                          # renders the figure
```

The scaling dimensions are native pytest parameters, so any slice is one `-k`
away. See `benchmarks/README.md` for the full matrix and the `benchmem` workflow.
