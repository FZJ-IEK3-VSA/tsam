"""Script to edit how_it_works notebooks (Task A and Task B)."""
from __future__ import annotations

import json
import pathlib
import sys


NOTEBOOKS_DIR = pathlib.Path(r"C:\Programming\tsam\docs\notebooks\how_it_works")


def read_nb(path: pathlib.Path) -> dict:
    raw = path.read_text(encoding="utf-8-sig")
    return json.loads(raw)


def write_nb(path: pathlib.Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def find_cell(nb: dict, cell_id: str) -> dict:
    for cell in nb["cells"]:
        if cell.get("id") == cell_id:
            return cell
    raise KeyError(f"Cell {cell_id!r} not found")


def cell_after(nb: dict, cell_id: str) -> int:
    """Return index of cell AFTER the given cell_id."""
    for i, cell in enumerate(nb["cells"]):
        if cell.get("id") == cell_id:
            return i + 1
    raise KeyError(f"Cell {cell_id!r} not found")


def insert_cell(nb: dict, after_id: str, new_cell: dict) -> None:
    idx = cell_after(nb, after_id)
    nb["cells"].insert(idx, new_cell)


def make_markdown(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source,
    }


def make_code(cell_id: str, source: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "source": source,
        "outputs": [],
        "execution_count": None,
    }


# =========================================================================
# TASK A — Rewrite 00_overview.ipynb intro
# =========================================================================
def task_a_overview():
    path = NOTEBOOKS_DIR / "00_overview.ipynb"
    nb = read_nb(path)

    new_title = (
        "# How aggregation works — Overview\n"
        "\n"
        "---\n"
        "\n"
        "## 1  The TSA taxonomy: two axes, four cells\n"
        "\n"
        "Time-series aggregation (TSA) methods can be classified along two independent axes\n"
        "(Hoffmann et al. 2020, [Table 3](https://www.mdpi.com/1996-1073/13/3/641)):\n"
        "\n"
        "* **How** periods or timesteps are grouped — by **time position** (consecutive\n"
        "  blocks, calendar position) or by **feature similarity** (clustering by value).\n"
        "* **What the result looks like** — **resolution variation** (fewer or coarser\n"
        "  timesteps, same calendar extent) or **typical periods** (a small set of\n"
        "  representative day-shapes with occurrence weights).\n"
        "\n"
        "These two axes are independent, so methods from both columns can be\n"
        "combined in a single pipeline (e.g., cluster periods *and* segment within each period).\n"
        "\n"
        "|                   | **Resolution variation** | **Typical periods**          |\n"
        "|-------------------|--------------------------|------------------------------|\n"
        "| **Time-based**    | Downsampling             | Time slices / averaging      |\n"
        "| **Feature-based** | Segmentation             | Clustering                   |\n"
        "\n"
        "---\n"
        "\n"
        "## 2  What tsam implements — and where it sits in the table\n"
        "\n"
        "tsam's primary focus is the **feature-based / typical periods** cell: given a\n"
        "time series split into equal-length periods (e.g. days), it finds $k$ representative\n"
        "periods using clustering.\n"
        "\n"
        "| Taxonomy cell | Method | tsam name | Notebook |\n"
        "|---|---|---|---|\n"
        "| Feature-based → Typical periods | K-means | `kmeans` | [01](01_partitional_clustering.ipynb) |\n"
        "| Feature-based → Typical periods | K-medoids | `kmedoids` | [01](01_partitional_clustering.ipynb) |\n"
        "| Feature-based → Typical periods | K-maxoids | `kmaxoids` | [01](01_partitional_clustering.ipynb) |\n"
        "| Feature-based → Typical periods | Hierarchical Ward | `hierarchical` | [02](02_agglomerative_clustering.ipynb) |\n"
        "| Feature-based → Typical periods | Contiguous Ward\\* | `contiguous` | [02](02_agglomerative_clustering.ipynb) |\n"
        "| Time-based → Typical periods | Block averaging | `averaging` | [03](03_averaging.ipynb) |\n"
        "| Feature-based → Resolution variation | Segmentation | `SegmentConfig` | [04](04_segmentation.ipynb) |\n"
        "| Cross-cutting | Representation & rescaling | `ClusterConfig(representation=...)` | [05](05_representation_rescaling.ipynb) |\n"
        "| Cross-cutting | Extreme periods | `ExtremeConfig` | [06](06_extreme_periods.ipynb) |\n"
        "\n"
        "**Nuances worth noting:**\n"
        "\n"
        "* `contiguous`\\* is Ward agglomerative clustering with a temporal-adjacency\n"
        "  constraint — it is **feature-based** (driven by value similarity), not\n"
        "  time-based. The adjacency constraint prevents non-neighbouring periods from\n"
        "  merging, but the merge cost is still the Ward variance criterion.\n"
        "* `averaging` produces **consecutive equal-size blocks** only. Full calendar\n"
        "  time-slices (e.g. \"all winter weekdays\") are not built in; build the\n"
        "  assignment vector externally if you need them.\n"
        "* **Downsampling** (coarser timesteps, same calendar) is intentionally\n"
        "  delegated to pandas: `df.resample(rule).mean()`. tsam does not duplicate this.\n"
        "\n"
        "---\n"
        "\n"
        "## 3  Further methods not in tsam\n"
        "\n"
        "Several methods surveyed in [Hoffmann et al. (2020)](https://www.mdpi.com/1996-1073/13/3/641)\n"
        "are not implemented in tsam. Downsampling is a one-liner in pandas and needs no\n"
        "dedicated support. Full calendar time-slices (grouping by season and weekday type)\n"
        "require an external assignment vector. Alternative centroid variants such as\n"
        "k-medians and k-centers, shape-based and time-shift-tolerant methods (DTW,\n"
        "k-shape), and dimensionality-reduction pre-processing (PCA, autoencoders) all\n"
        "fall outside tsam's scope and are surveyed with references in Hoffmann et al.\n"
        "(2020). Random period sampling and multiple-time-grid schemes (different\n"
        "resolutions per season) are likewise not supported within a single aggregation call.\n"
        "\n"
        "---\n"
        "\n"
        "## The tiny synthetic dataset\n"
        "\n"
        "Six days × 4 timesteps/day (6-hourly), two attributes: **solar** irradiance proxy and\n"
        "**load** proxy. Day shapes are deliberately distinct:\n"
        "\n"
        "* Days 0–1: bright sunny days (high solar peak)\n"
        "* Days 2–3: overcast / moderate days\n"
        "* Days 4–5: cloudy + one extreme-load day\n"
        "\n"
        "This is small enough to verify every number by hand."
    )

    cell = find_cell(nb, "md-title")
    cell["source"] = new_title
    # Clear outputs on all code cells (they will be re-executed)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    write_nb(path, nb)
    print(f"[DONE] Task A: rewrote {path.name}")


# =========================================================================
# TASK B — Add config cells to deep-dive notebooks
# =========================================================================

def task_b_01_partitional():
    path = NOTEBOOKS_DIR / "01_partitional_clustering.ipynb"
    nb = read_nb(path)

    # Clear all code cell outputs
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # --- K-means config cell (insert after md-kmeans, before lloyd-iteration) ---
    cfg_kmeans_md = make_markdown(
        "cfg-kmeans-md",
        "**TSAM configuration for k-means:**"
    )
    cfg_kmeans_code = make_code(
        "cfg-kmeans-config",
        "from tsam import ClusterConfig\nimport tsam\n\n"
        "# K-means: feature-based clustering using Lloyd's algorithm.\n"
        "# representation defaults to 'mean' (centroid) for kmeans.\n"
        "cfg_kmeans = ClusterConfig(method=\"kmeans\", representation=\"mean\")\n"
        "print(cfg_kmeans)\n"
        "\n"
        "# Full aggregate call:\n"
        "# result = tsam.aggregate(\n"
        "#     df,\n"
        "#     n_clusters=k,\n"
        "#     period_duration=\"1D\",\n"
        "#     cluster=cfg_kmeans,\n"
        "# )"
    )
    insert_cell(nb, "md-kmeans", cfg_kmeans_md)
    insert_cell(nb, "cfg-kmeans-md", cfg_kmeans_code)

    # --- K-medoids config cell (insert after md-kmeans-note) ---
    cfg_kmed_md = make_markdown(
        "cfg-kmedoids-md",
        "**TSAM configuration for k-medoids:**"
    )
    cfg_kmed_code = make_code(
        "cfg-kmedoids-config",
        "# K-medoids: each representative is an actual observed period (medoid).\n"
        "# Uses MILP optimization — solver='highs' (default, open-source).\n"
        "cfg_kmedoids = ClusterConfig(method=\"kmedoids\", representation=\"medoid\", solver=\"highs\")\n"
        "print(cfg_kmedoids)\n"
        "\n"
        "# Full aggregate call:\n"
        "# result = tsam.aggregate(\n"
        "#     df,\n"
        "#     n_clusters=k,\n"
        "#     period_duration=\"1D\",\n"
        "#     cluster=cfg_kmedoids,\n"
        "# )"
    )
    insert_cell(nb, "md-kmeans-note", cfg_kmed_md)
    insert_cell(nb, "cfg-kmedoids-md", cfg_kmed_code)

    # --- K-maxoids config cell (insert after md-kmaxoids) ---
    cfg_kmx_md = make_markdown(
        "cfg-kmaxoids-md",
        "**TSAM configuration for k-maxoids:**"
    )
    cfg_kmx_code = make_code(
        "cfg-kmaxoids-config",
        "# K-maxoids: selects the k most mutually dissimilar periods.\n"
        "# representation defaults to 'maxoid' for kmaxoids.\n"
        "cfg_kmaxoids = ClusterConfig(method=\"kmaxoids\", representation=\"maxoid\")\n"
        "print(cfg_kmaxoids)\n"
        "\n"
        "# Full aggregate call:\n"
        "# result = tsam.aggregate(\n"
        "#     df,\n"
        "#     n_clusters=k,\n"
        "#     period_duration=\"1D\",\n"
        "#     cluster=cfg_kmaxoids,\n"
        "# )"
    )
    insert_cell(nb, "md-kmaxoids", cfg_kmx_md)
    insert_cell(nb, "cfg-kmaxoids-md", cfg_kmx_code)

    write_nb(path, nb)
    print(f"[DONE] Task B: added config cells to {path.name}")


def task_b_02_agglomerative():
    path = NOTEBOOKS_DIR / "02_agglomerative_clustering.ipynb"
    nb = read_nb(path)

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # Hierarchical config — insert after md-hierarchical
    cfg_hier_md = make_markdown(
        "cfg-hierarchical-md",
        "**TSAM configuration for hierarchical clustering:**"
    )
    cfg_hier_code = make_code(
        "cfg-hierarchical-config",
        "from tsam import ClusterConfig\nimport tsam\n\n"
        "# Hierarchical: bottom-up Ward agglomerative clustering, unconstrained.\n"
        "# representation defaults to 'medoid' for hierarchical.\n"
        "cfg_hierarchical = ClusterConfig(method=\"hierarchical\", representation=\"medoid\")\n"
        "print(cfg_hierarchical)\n"
        "\n"
        "# Full aggregate call:\n"
        "# result = tsam.aggregate(\n"
        "#     df,\n"
        "#     n_clusters=k,\n"
        "#     period_duration=\"1D\",\n"
        "#     cluster=cfg_hierarchical,\n"
        "# )"
    )
    insert_cell(nb, "md-hierarchical", cfg_hier_md)
    insert_cell(nb, "cfg-hierarchical-md", cfg_hier_code)

    # Contiguous config — insert after md-contiguous
    cfg_cont_md = make_markdown(
        "cfg-contiguous-md",
        "**TSAM configuration for contiguous clustering:**"
    )
    cfg_cont_code = make_code(
        "cfg-contiguous-config",
        "# Contiguous: Ward agglomerative clustering with temporal-adjacency constraint.\n"
        "# Feature-based (Ward variance criterion), not time-based.\n"
        "# representation defaults to 'medoid' for contiguous.\n"
        "cfg_contiguous = ClusterConfig(method=\"contiguous\", representation=\"medoid\")\n"
        "print(cfg_contiguous)\n"
        "\n"
        "# Full aggregate call:\n"
        "# result = tsam.aggregate(\n"
        "#     df,\n"
        "#     n_clusters=k,\n"
        "#     period_duration=\"1D\",\n"
        "#     cluster=cfg_contiguous,\n"
        "# )"
    )
    insert_cell(nb, "md-contiguous", cfg_cont_md)
    insert_cell(nb, "cfg-contiguous-md", cfg_cont_code)

    write_nb(path, nb)
    print(f"[DONE] Task B: added config cells to {path.name}")


def task_b_03_averaging():
    path = NOTEBOOKS_DIR / "03_averaging.ipynb"
    nb = read_nb(path)

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # Averaging config — insert after md-mechanism
    cfg_avg_md = make_markdown(
        "cfg-averaging-md",
        "**TSAM configuration for averaging:**"
    )
    cfg_avg_code = make_code(
        "cfg-averaging-config",
        "from tsam import ClusterConfig\nimport tsam\n\n"
        "# Averaging: purely positional consecutive-block grouping.\n"
        "# Time-based / typical periods — no value similarity involved.\n"
        "# representation defaults to 'mean' (block average) for averaging.\n"
        "cfg_averaging = ClusterConfig(method=\"averaging\", representation=\"mean\")\n"
        "print(cfg_averaging)\n"
        "\n"
        "# Full aggregate call:\n"
        "# result = tsam.aggregate(\n"
        "#     df,\n"
        "#     n_clusters=k,\n"
        "#     period_duration=\"1D\",\n"
        "#     cluster=cfg_averaging,\n"
        "# )"
    )
    insert_cell(nb, "md-mechanism", cfg_avg_md)
    insert_cell(nb, "cfg-averaging-md", cfg_avg_code)

    write_nb(path, nb)
    print(f"[DONE] Task B: added config cells to {path.name}")


def task_b_04_segmentation():
    path = NOTEBOOKS_DIR / "04_segmentation.ipynb"
    nb = read_nb(path)

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # Segmentation config — insert after md-mechanism
    cfg_seg_md = make_markdown(
        "cfg-segmentation-md",
        "**TSAM configuration for segmentation:**"
    )
    cfg_seg_code = make_code(
        "cfg-segmentation-config",
        "from tsam import ClusterConfig, SegmentConfig\nimport tsam\n\n"
        "# SegmentConfig reduces the number of timesteps within each period.\n"
        "# n_segments: how many segments per period after merging.\n"
        "# representation: how each segment is represented (default 'mean').\n"
        "cfg_seg = SegmentConfig(n_segments=6, representation=\"mean\")\n"
        "print(cfg_seg)\n"
        "\n"
        "# Segmentation is passed separately to tsam.aggregate — it can be combined\n"
        "# with any clustering method:\n"
        "# result = tsam.aggregate(\n"
        "#     df,\n"
        "#     n_clusters=k,\n"
        "#     period_duration=\"1D\",\n"
        "#     cluster=ClusterConfig(method=\"hierarchical\"),\n"
        "#     segments=SegmentConfig(n_segments=6),\n"
        "# )\n"
        "\n"
        "# Segmentation alone (no clustering — n_clusters=1, all periods in one cluster):\n"
        "# result = tsam.aggregate(\n"
        "#     df,\n"
        "#     n_clusters=len(df) // timesteps_per_period,\n"
        "#     period_duration=\"1D\",\n"
        "#     segments=SegmentConfig(n_segments=6),\n"
        "# )"
    )
    insert_cell(nb, "md-mechanism", cfg_seg_md)
    insert_cell(nb, "cfg-segmentation-md", cfg_seg_code)

    write_nb(path, nb)
    print(f"[DONE] Task B: added config cells to {path.name}")


def task_b_05_representation():
    path = NOTEBOOKS_DIR / "05_representation_rescaling.ipynb"
    nb = read_nb(path)

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # Representation config — insert after md-representations
    cfg_rep_md = make_markdown(
        "cfg-representation-md",
        "**TSAM configuration for representation strategies:**"
    )
    cfg_rep_code = make_code(
        "cfg-representation-config",
        "from tsam import ClusterConfig, SegmentConfig\n"
        "from tsam.config import Distribution, MinMaxMean\n"
        "import tsam\n"
        "\n"
        "# The representation= parameter is set on ClusterConfig (or SegmentConfig).\n"
        "# String shortcuts:\n"
        "cfg_mean    = ClusterConfig(method=\"hierarchical\", representation=\"mean\")\n"
        "cfg_medoid  = ClusterConfig(method=\"hierarchical\", representation=\"medoid\")\n"
        "cfg_maxoid  = ClusterConfig(method=\"kmaxoids\",    representation=\"maxoid\")\n"
        "\n"
        "# Typed objects (additional options):\n"
        "cfg_dist_cluster = ClusterConfig(\n"
        "    method=\"hierarchical\",\n"
        "    representation=Distribution(scope=\"cluster\"),   # per-cluster duration curve\n"
        ")\n"
        "cfg_dist_global = ClusterConfig(\n"
        "    method=\"hierarchical\",\n"
        "    representation=Distribution(scope=\"global\"),    # overall duration curve\n"
        ")\n"
        "cfg_minmaxmean = ClusterConfig(\n"
        "    method=\"hierarchical\",\n"
        "    representation=MinMaxMean(max_columns=[\"Load\"], min_columns=[]),\n"
        ")\n"
        "\n"
        "# SegmentConfig also accepts representation=:\n"
        "cfg_seg_medoid = SegmentConfig(n_segments=6, representation=\"medoid\")\n"
        "\n"
        "print('mean:        ', cfg_mean)\n"
        "print('medoid:      ', cfg_medoid)\n"
        "print('dist_cluster:', cfg_dist_cluster)\n"
        "print('minmaxmean:  ', cfg_minmaxmean)\n"
        "print('seg_medoid:  ', cfg_seg_medoid)"
    )
    insert_cell(nb, "md-representations", cfg_rep_md)
    insert_cell(nb, "cfg-representation-md", cfg_rep_code)

    # Rescaling config — insert after md-rescaling
    cfg_rescale_md = make_markdown(
        "cfg-rescaling-md",
        "**TSAM configuration for rescaling:**"
    )
    cfg_rescale_code = make_code(
        "cfg-rescaling-config",
        "# Rescaling is controlled by preserve_column_means (top-level aggregate param).\n"
        "# It is NOT part of ClusterConfig — it applies after all clustering is done.\n"
        "\n"
        "# Default (preserve_column_means=True): rescale so weighted mean of\n"
        "# representatives matches original column means.\n"
        "result_rescaled = tsam.aggregate(\n"
        "    data,\n"
        "    n_clusters=6,\n"
        "    period_duration=\"1D\",\n"
        "    cluster=ClusterConfig(method=\"hierarchical\"),\n"
        "    preserve_column_means=True,   # default — recommended\n"
        ")\n"
        "\n"
        "# To disable rescaling:\n"
        "# result_no_rescale = tsam.aggregate(\n"
        "#     data, n_clusters=6, period_duration=\"1D\",\n"
        "#     cluster=ClusterConfig(method=\"hierarchical\"),\n"
        "#     preserve_column_means=False,\n"
        "# )\n"
        "\n"
        "# scale_by_column_means (on ClusterConfig) is a *clustering* pre-step:\n"
        "# it divides each column by its mean before clustering, so all columns\n"
        "# contribute equally even if they differ in magnitude.\n"
        "cfg_scaled = ClusterConfig(method=\"hierarchical\", scale_by_column_means=True)\n"
        "print('scale_by_column_means config:', cfg_scaled)\n"
        "print('preserve_column_means result RMSE:', round(result_rescaled.accuracy.weighted_rmse, 4))"
    )
    insert_cell(nb, "md-rescaling", cfg_rescale_md)
    insert_cell(nb, "cfg-rescaling-md", cfg_rescale_code)

    write_nb(path, nb)
    print(f"[DONE] Task B: added config cells to {path.name}")


def task_b_06_extremes():
    path = NOTEBOOKS_DIR / "06_extreme_periods.ipynb"
    nb = read_nb(path)

    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None

    # ExtremeConfig overview — insert after md-title
    cfg_ext_md = make_markdown(
        "cfg-extremes-md",
        "**TSAM configuration for extreme periods:**"
    )
    cfg_ext_code = make_code(
        "cfg-extremes-config",
        "from tsam import ClusterConfig, ExtremeConfig\nimport tsam\n\n"
        "# ExtremeConfig is passed as the extremes= argument to tsam.aggregate.\n"
        "# method controls how the extreme period is incorporated:\n"
        "#   'append'      — add as extra cluster (total count increases by 1 per extreme)\n"
        "#   'replace'     — replace nearest existing cluster center (count stays same)\n"
        "#   'new_cluster' — add as new cluster and reassign affected periods (count +1)\n"
        "\n"
        "# Selection criteria (at least one must be non-empty):\n"
        "#   max_value  — period containing the single highest timestep value\n"
        "#   min_value  — period containing the single lowest timestep value\n"
        "#   max_period — period with the highest column sum (e.g. peak solar day)\n"
        "#   min_period — period with the lowest column sum (e.g. lowest wind day)\n"
        "\n"
        "cfg_append = ExtremeConfig(\n"
        "    method=\"append\",\n"
        "    max_value=[\"load\"],   # preserve the period with the peak load timestep\n"
        ")\n"
        "cfg_replace = ExtremeConfig(method=\"replace\", max_value=[\"load\"])\n"
        "cfg_new_cluster = ExtremeConfig(method=\"new_cluster\", max_value=[\"load\"])\n"
        "\n"
        "# Example with multiple criteria:\n"
        "cfg_multi = ExtremeConfig(\n"
        "    method=\"append\",\n"
        "    max_value=[\"load\"],\n"
        "    min_period=[\"solar\"],\n"
        ")\n"
        "\n"
        "print('append config:     ', cfg_append)\n"
        "print('replace config:    ', cfg_replace)\n"
        "print('new_cluster config:', cfg_new_cluster)\n"
        "print('multi config:      ', cfg_multi)\n"
        "\n"
        "# Full aggregate call:\n"
        "# result = tsam.aggregate(\n"
        "#     df,\n"
        "#     n_clusters=k,\n"
        "#     period_duration=\"1D\",\n"
        "#     cluster=ClusterConfig(method=\"hierarchical\"),\n"
        "#     extremes=ExtremeConfig(method=\"append\", max_value=[\"load\"]),\n"
        "# )"
    )
    insert_cell(nb, "md-title", cfg_ext_md)
    insert_cell(nb, "cfg-extremes-md", cfg_ext_code)

    write_nb(path, nb)
    print(f"[DONE] Task B: added config cells to {path.name}")


if __name__ == "__main__":
    task_a_overview()
    task_b_01_partitional()
    task_b_02_agglomerative()
    task_b_03_averaging()
    task_b_04_segmentation()
    task_b_05_representation()
    task_b_06_extremes()
    print("\nAll done.")
