"""Performance benchmarks for :func:`tsam.aggregate`.

Star design: one baseline configuration (a hourly year, 8 typical days,
hierarchical clustering, library-default representation) with exactly one
dimension varied per test function. Every parametrize value is a scalar, so
each becomes a pytest-benchmem dim and ``benchmem plot --x <dim>`` slices out
that function's scaling curve.

On top of its own varied axis, every benchmark records its whole shape and
configuration through :func:`_stamp_dims` — timesteps, columns, clusters,
segments, method, representation, scope, extremes, rescaling. That is what
makes the suite one queryable dataset rather than a set of unrelated curves:
cost can be plotted against any axis and filtered on any other, across
functions.

Usage::

    pytest benchmarks/ --benchmark-only                       # quick tier, timing
    pytest benchmarks/ --benchmark-only --slow                # + k-medoids, 96/400 columns
    pytest benchmarks/ --benchmark-only --large               # + production-sized cases
    pytest benchmarks/ --benchmark-only --benchmark-memory    # + memray peak memory
    pytest benchmarks/ --benchmark-only --benchmark-save=dev  # snapshot to .benchmarks/

    benchmem compare '*base*' '*dev*' --columns time --diff
    benchmem plot  .benchmarks/*/0001_dev.json --x n_columns --color method
    benchmem plot  .benchmarks/*/0001_dev.json --x n_clusters --where n_segments=12
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest

from tsam import ClusterConfig, Distribution, ExtremeConfig, SegmentConfig, aggregate

ROOT = Path(__file__).resolve().parent.parent
TESTDATA_CSV = ROOT / "docs" / "data" / "testdata.csv"
WIDE_CSV = ROOT / "test" / "data" / "wide.csv"

N_CLUSTERS = 8
ROUNDS = 10
#: Width the scaling sweeps run at. Not the 4-column example: a whole
#: aggregation of it takes ~6 ms, of which ~4 ms is fixed per-call overhead, so
#: a curve measured there describes the overhead rather than the algorithm —
#: ward's quadratic in the period count is invisible until the candidate matrix
#: is wide enough to matter. At 48 columns a year costs ~34 ms and three years
#: ~183 ms, and the shape of the curve is the thing being measured.
SCALE_COLUMNS = 48
#: Production-sized cases run few rounds: one is already seconds long.
LARGE_OPTS = {"rounds": 3, "iterations": 1, "warmup_rounds": 0}


@lru_cache
def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _testdata() -> pd.DataFrame:
    """One year of hourly GHI/T/Wind/Load data (8760 x 4)."""
    return _load(str(TESTDATA_CSV))


def _wide_columns(n_columns: int) -> pd.DataFrame:
    """wide.csv tiled with unique column names up to ``n_columns`` (8760 rows)."""
    base = _load(str(WIDE_CSV))
    repeats = -(-n_columns // base.shape[1])
    tiled = pd.concat([base.add_suffix(f"_{i}") for i in range(repeats)], axis=1)
    return tiled.iloc[:, :n_columns]


def _tile_years(data: pd.DataFrame, n_years: int) -> pd.DataFrame:
    """``data`` repeated ``n_years`` times on one continuous hourly index."""
    tiled = pd.concat([data] * n_years, ignore_index=True)
    tiled.index = pd.date_range("2020-01-01", periods=len(tiled), freq="h")
    return tiled


def _quarter_hourly() -> pd.DataFrame:
    """testdata linearly interpolated to 15-min resolution (35,040 rows)."""
    base = _testdata()
    idx = pd.date_range(base.index[0], periods=len(base) * 4, freq="15min")
    return base.reindex(idx).interpolate(method="linear").ffill()


def _stamp_dims(benchmark, data: pd.DataFrame, n_clusters: int, kwargs: dict) -> None:
    """Record the full shape and configuration as pytest-benchmem dims.

    Dims are the parametrize params plus any scalar ``extra_info``, so a
    benchmark that only carries its own varied axis cannot be compared with
    any other: ``test_method[kmeans]`` knowing nothing about its column count
    means no plot can put method cost against input size, and ``--where
    n_columns=4`` silently drops it. Stamping every run with its whole shape
    turns the suite into one queryable dataset instead of a set of unrelated
    curves. The varied axis stays a real parametrize param, so nothing
    collapses.
    """
    cluster = kwargs.get("cluster")
    segments = kwargs.get("segments")
    extremes = kwargs.get("extremes")
    representation = getattr(cluster, "representation", None)

    if representation is None:
        representation_name = "default"
    elif isinstance(representation, str):
        representation_name = representation
    else:
        representation_name = type(representation).__name__.lower()

    benchmark.extra_info.update(
        {
            "n_timesteps": len(data),
            "n_columns": data.shape[1],
            "n_clusters": n_clusters,
            "n_segments": getattr(segments, "n_segments", 0),
            "method": getattr(cluster, "method", "hierarchical"),
            "representation": representation_name,
            # "local" / "global" is the library's own vocabulary; only the
            # duration representations have a scope at all.
            "scope": getattr(representation, "scope", "n/a"),
            "extremes": getattr(extremes, "method", "none"),
            "rescale": kwargs.get("preserve_column_means", True),
        }
    )


def _bench(
    benchmark, data: pd.DataFrame, n_clusters: int = N_CLUSTERS, **kwargs
) -> None:
    _stamp_dims(benchmark, data, n_clusters, kwargs)
    benchmark.pedantic(
        lambda: aggregate(data, n_clusters, **kwargs),
        rounds=ROUNDS,
        warmup_rounds=1,
    )


@pytest.mark.benchmark(group="scale-timesteps")
@pytest.mark.parametrize("n_years", [1, 2, 3, pytest.param(5, marks=pytest.mark.slow)])
def test_scale_timesteps(benchmark, n_years):
    """Baseline config over a growing number of years.

    The period count grows with this axis and the linkage is quadratic in it,
    so this is the curve that bends: one year to three is 3x the periods and
    roughly 5x the time.
    """
    _bench(benchmark, _tile_years(_wide_columns(SCALE_COLUMNS), n_years))


@pytest.mark.benchmark(group="scale-columns")
@pytest.mark.parametrize(
    "n_columns",
    [
        4,
        12,
        48,
        pytest.param(96, marks=pytest.mark.slow),
        pytest.param(400, marks=pytest.mark.slow),
    ],
)
def test_scale_columns(benchmark, n_columns):
    """Baseline config over a growing number of columns (full year)."""
    _bench(benchmark, _wide_columns(n_columns))


@pytest.mark.benchmark(group="method")
@pytest.mark.parametrize(
    "method", ["averaging", "kmeans", "kmaxoids", "hierarchical", "contiguous"]
)
def test_method(benchmark, method):
    """Clustering methods on the full year, default representation each."""
    _bench(benchmark, _testdata(), cluster=ClusterConfig(method=method))


@pytest.mark.slow
@pytest.mark.benchmark(group="method")
def test_method_kmedoids(benchmark):
    """Exact k-medoids (MILP): truncated to 8 weeks to keep the solve tractable.

    Not size-comparable with :func:`test_method` — its dims record the reduced
    input, so plots and tables place it at its own size rather than alongside
    the full-year methods.
    """
    _bench(
        benchmark,
        _testdata().iloc[: 8 * 168],
        cluster=ClusterConfig(method="kmedoids"),
    )


#: Representation name -> what `ClusterConfig` takes. Named separately so the
#: parametrize value stays a scalar dim while global scope can be an object.
REPRESENTATIONS = {
    "mean": "mean",
    "medoid": "medoid",
    "maxoid": "maxoid",
    "distribution": "distribution",
    "distribution_global": Distribution(scope="global"),
    "minmax_mean": "minmax_mean",
}


@pytest.mark.benchmark(group="representation")
@pytest.mark.parametrize("representation", list(REPRESENTATIONS))
def test_representation(benchmark, representation):
    """Representations on hierarchical clustering, full year."""
    _bench(
        benchmark,
        _testdata(),
        cluster=ClusterConfig(
            method="hierarchical", representation=REPRESENTATIONS[representation]
        ),
    )


@pytest.mark.benchmark(group="feature")
@pytest.mark.parametrize(
    "feature", ["baseline", "segmentation", "extremes", "no_rescale"]
)
def test_feature(benchmark, feature):
    """Cost of each pipeline feature toggled onto the baseline, one at a time."""
    kwargs = {}
    if feature == "segmentation":
        kwargs["segments"] = SegmentConfig(n_segments=12)
    elif feature == "extremes":
        kwargs["extremes"] = ExtremeConfig(
            method="new_cluster", max_value=["Load"], min_value=["T"]
        )
    elif feature == "no_rescale":
        kwargs["preserve_column_means"] = False
    _bench(benchmark, _testdata(), **kwargs)


@pytest.mark.benchmark(group="resolution")
def test_resolution_15min(benchmark):
    """One year at 15-min resolution: 96 steps per daily period stress the reshapes."""
    benchmark.extra_info["steps_per_period"] = 96
    _bench(benchmark, _quarter_hourly())


@pytest.mark.benchmark(group="extremes-scale")
def test_extremes_3y(benchmark):
    """new_cluster extreme handling over 3 years; reassignment scales with period count."""
    _bench(
        benchmark,
        _tile_years(_testdata(), 3),
        extremes=ExtremeConfig(
            method="new_cluster", max_value=["Load"], min_value=["T"]
        ),
    )


@pytest.mark.large
@pytest.mark.benchmark(group="large")
def test_large_wide(benchmark):
    """Production-sized single frame: two years hourly x 256 columns."""
    data = _tile_years(_wide_columns(256), 2)
    _stamp_dims(benchmark, data, N_CLUSTERS, {})
    benchmark.pedantic(lambda: aggregate(data, N_CLUSTERS), **LARGE_OPTS)


@pytest.mark.large
@pytest.mark.benchmark(group="large")
def test_large_scenarios(benchmark):
    """Eight sequential year x 64-column aggregations (multi-scenario workload)."""
    data = _wide_columns(64)
    _stamp_dims(benchmark, data, N_CLUSTERS, {})
    benchmark.extra_info["n_slices"] = 8
    benchmark.pedantic(
        lambda: [aggregate(data, N_CLUSTERS) for _ in range(8)], **LARGE_OPTS
    )


def _fine_data() -> pd.DataFrame:
    """FINE 8-region example: 5 profiles x 8 regions, hourly year (8760 x 40)."""
    return _load(str(ROOT / "benchmarks" / "data" / "fine_multiregional.csv.gz"))


@pytest.mark.benchmark(group="headline")
def test_fine_default(benchmark):
    """FINE's aggregateTemporally() defaults: 40 typical days, hierarchical +
    duration representation, 12 segments, no rescale."""
    _bench(
        benchmark,
        _fine_data(),
        n_clusters=40,
        cluster=ClusterConfig(method="hierarchical", representation="distribution"),
        segments=SegmentConfig(n_segments=12),
        preserve_column_means=False,
    )


@pytest.mark.benchmark(group="headline")
def test_fine_extremes(benchmark):
    """FINE defaults plus appended peak-demand extreme periods per region."""
    data = _fine_data()
    peak_columns = [c for c in data.columns if c.startswith("ElecDemand")]
    _bench(
        benchmark,
        data,
        n_clusters=40,
        cluster=ClusterConfig(method="hierarchical", representation="distribution"),
        segments=SegmentConfig(n_segments=12),
        extremes=ExtremeConfig(method="append", max_value=peak_columns),
        preserve_column_means=False,
    )


@pytest.mark.large
@pytest.mark.benchmark(group="headline")
def test_fine_production(benchmark):
    """Production-scale FINE workload (the issue #49 shape): two years x 400
    columns of tiled FINE profiles, 40 typical days, global-scope
    distribution, 12 segments, peak-demand extremes. ~6 s on tsam 3.4.2."""
    base = _fine_data()
    wide = pd.concat([base.add_suffix(f"_{i}") for i in range(10)], axis=1)
    data = pd.concat([wide] * 2, ignore_index=True)
    data.index = pd.date_range("2020-01-01", periods=len(data), freq="h")
    kwargs = {
        "cluster": ClusterConfig(
            method="hierarchical", representation=Distribution(scope="global")
        ),
        "segments": SegmentConfig(n_segments=12),
        "extremes": ExtremeConfig(
            method="append",
            max_value=[c for c in data.columns if c.startswith("ElecDemand")],
        ),
        "preserve_column_means": False,
    }
    _stamp_dims(benchmark, data, 40, kwargs)
    benchmark.pedantic(lambda: aggregate(data, 40, **kwargs), **LARGE_OPTS)


@pytest.mark.benchmark(group="accuracy")
@pytest.mark.parametrize(
    "n_columns", [4, 48, pytest.param(400, marks=pytest.mark.slow)]
)
def test_accuracy(benchmark, n_columns):
    """Lazy accuracy metrics on a fresh result each round.

    The per-column work dominates, so the column count is the axis that
    matters — at four columns the metrics are nearly free and at 400 they
    cost more than the aggregation that produced them.
    """
    data = _wide_columns(n_columns)
    _stamp_dims(benchmark, data, N_CLUSTERS, {})
    benchmark.extra_info["phase"] = "accuracy"
    benchmark.pedantic(
        lambda result: result.accuracy,
        setup=lambda: ((aggregate(data, N_CLUSTERS),), {}),
        rounds=ROUNDS,
        warmup_rounds=1,
    )


@pytest.mark.benchmark(group="representation-scale")
@pytest.mark.parametrize(
    "n_columns", [4, 48, pytest.param(400, marks=pytest.mark.slow)]
)
def test_representation_global_scale(benchmark, n_columns):
    """Global-scope duration representation over a growing number of columns.

    Global scope fits one duration curve across all periods rather than one
    per period, so its cost scales with columns differently from the
    period-wise default that `test_representation` covers.
    """
    _bench(
        benchmark,
        _wide_columns(n_columns),
        cluster=ClusterConfig(
            method="hierarchical", representation=Distribution(scope="global")
        ),
    )


@pytest.mark.benchmark(group="scale-clusters")
@pytest.mark.parametrize("n_clusters", [8, 40, 200])
def test_scale_clusters(benchmark, n_clusters):
    """Baseline config over a growing number of typical periods.

    The least intuitive axis: asking for more typical periods costs more even
    though the clustering itself is unchanged — the linkage is cut at a
    different height, but representation, rescaling and extreme handling all
    scale with the number of clusters they produce. The slope is shallow, and
    that is the finding.
    """
    _bench(benchmark, _wide_columns(SCALE_COLUMNS), n_clusters=n_clusters)


@pytest.mark.benchmark(group="scale-segments")
@pytest.mark.parametrize("n_segments", [3, 6, 12, 24])
def test_scale_segments(benchmark, n_segments):
    """Segments per typical period, at a fixed typical-period count.

    Separates the two factors `test_segmentation_scale` multiplies together:
    this one holds the period count still and varies how finely each period is
    cut. 24 is the ceiling for daily periods — one segment per hour, i.e. no
    reduction at all.
    """
    _bench(
        benchmark,
        _wide_columns(SCALE_COLUMNS),
        segments=SegmentConfig(n_segments=n_segments),
    )


@pytest.mark.benchmark(group="segmentation-scale")
@pytest.mark.parametrize("n_clusters", [10, 40, 80])
def test_segmentation_scale(benchmark, n_clusters):
    """Segmentation cost against the number of typical periods.

    Segments are clustered within each typical period independently, so this
    axis multiplies: doubling the typical periods doubles the segmentation
    work while leaving the period clustering roughly flat.
    """
    _bench(
        benchmark,
        _wide_columns(SCALE_COLUMNS),
        n_clusters=n_clusters,
        segments=SegmentConfig(n_segments=12),
    )


@pytest.mark.benchmark(group="disaggregate")
@pytest.mark.parametrize("shape", ["small", "wide", "wide_segmented"])
def test_disaggregate(benchmark, shape):
    """Expanding typical-period data back to the full datetime index.

    Reconstruction of the input is eager inside ``aggregate()`` and therefore
    already timed by every other benchmark; this times the standalone
    expansion of external (e.g. optimization) results. Segmented results
    expand through a different path, and the cost only becomes visible at a
    realistic cluster and column count — hence the three shapes.
    """
    if shape == "small":
        data, n_clusters, segments = _testdata(), N_CLUSTERS, None
    else:
        data, n_clusters = _wide_columns(20), 36
        segments = SegmentConfig(n_segments=12) if shape == "wide_segmented" else None

    _stamp_dims(benchmark, data, n_clusters, {"segments": segments})
    benchmark.extra_info["phase"] = "disaggregate"
    result = aggregate(data, n_clusters, segments=segments)
    benchmark.pedantic(
        lambda: result.disaggregate(result.cluster_representatives),
        rounds=ROUNDS,
        warmup_rounds=1,
    )
