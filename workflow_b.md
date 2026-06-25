Workstream B — TSAM v4 validation harness vs v3.4.1 + PR #282 characterization.
This is testable Python modules + tests, NOT docs. First read the memory files
(pr282-bug-v4-inherited, project-docs-rework, tsam-environments), the PR #282 diff
(`gh pr diff 282 --repo FZJ-IEK3-VSA/tsam`), and test/_golden_cases.py +
test/test_golden_regression.py. Then propose the harness layout before building.

Goals:
1. Validate v4 (this worktree, env `tsam_improve_reworked_notebooks`) against the last
   3.x release, tsam 3.4.1, on identical configs/data: diff `reconstructed` within
   tolerance and flag any case where the aggregated series exceeds the input min/max
   envelope. Create a side env: `mamba create -n tsam_3_4_1 python=3.12 -y` then
   `mamba run -n tsam_3_4_1 pip install tsam==3.4.1` (v3.4.1 still ships the legacy
   `tsam.timeseriesaggregation.TimeSeriesAggregation`). Drive it via subprocess into
   that env.
2. Provide pipeline tracers I can step through stage by stage
   (normalize → unstack → cluster → represent → rescale → reconstruct).
3. Characterize PR #282 ("fix integral and min max preservation", open). v4 inherited
   the bug in src/tsam/pipeline/rescale.py (~L119-139, iterated multiplicative rescale
   loop) and src/tsam/algorithms/duration_representation.py::_represent_min_max
   (~L194-212, single multiplicative correction_factor, no clip/redistribute). Show:
   (a) v4 reproduces the envelope overshoot on the `wide` dataset with
   distribution+minmax+rescale (the `_EXPECT_MAXVAL_WARNING` cases), (b) the PR's
   clip-and-water-fill removes it, (c) the mean-preservation vs envelope trade-off
   (PR relaxes atol 1e-4 → rtol 5e-3).

Decision already made: CHARACTERIZE FIRST — do NOT change v4 behavior or regenerate
goldens. The harness documents the deviation. Confirm the architecture verdict: no
structural rework needed; the fix ports into those two functions; adopting it is a
behavioral change that would require regenerating the affected golden baselines.

Deliverable: a `validation/` package with (1) a side-by-side v4-vs-v3.4.1 runner over
the golden configs/datasets (test/data/golden/*/{testdata,wide,constant,
with_zero_column}.csv), (2) stage tracers, (3) a focused PR #282 characterization test
plus a short markdown report with the recommendation.
