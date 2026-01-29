"""End-to-end clustering tests with CSV fixtures.

This module tests clustering from input data to clustered output,
verifying results against expected CSV files and JSON metadata.
"""

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, ExtremeConfig, SegmentConfig, aggregate

# Fixed seed for reproducible results with stochastic methods (kmeans)
RANDOM_SEED = 42


def set_random_seed():
    """Set random seed for reproducibility of stochastic clustering methods."""
    np.random.seed(RANDOM_SEED)
    # Also set sklearn's random state if available
    try:
        import sklearn

        sklearn.utils.check_random_state(RANDOM_SEED)
    except ImportError:
        pass


class ClusteringTestCase(NamedTuple):
    """Configuration for a clustering test case."""

    id: str
    method: str
    representation: str
    n_segments: int | None = None
    extreme_method: str | None = None
    extreme_columns: list[str] | None = None


# Define all test configurations
TEST_CASES = [
    # Basic clustering methods
    ClusteringTestCase(
        id="hierarchical_medoid_8clusters",
        method="hierarchical",
        representation="medoid",
    ),
    ClusteringTestCase(
        id="kmeans_mean_8clusters",
        method="kmeans",
        representation="mean",
    ),
    ClusteringTestCase(
        id="kmedoids_medoid_8clusters",
        method="kmedoids",
        representation="medoid",
    ),
    ClusteringTestCase(
        id="kmaxoids_maxoid_8clusters",
        method="kmaxoids",
        representation="maxoid",
    ),
    ClusteringTestCase(
        id="contiguous_medoid_8clusters",
        method="contiguous",
        representation="medoid",
    ),
    # Different representations with hierarchical
    ClusteringTestCase(
        id="hierarchical_mean_8clusters",
        method="hierarchical",
        representation="mean",
    ),
    ClusteringTestCase(
        id="hierarchical_distribution_8clusters",
        method="hierarchical",
        representation="distribution",
    ),
    # With segmentation
    ClusteringTestCase(
        id="hierarchical_medoid_8clusters_12segments",
        method="hierarchical",
        representation="medoid",
        n_segments=12,
    ),
    ClusteringTestCase(
        id="hierarchical_medoid_8clusters_6segments",
        method="hierarchical",
        representation="medoid",
        n_segments=6,
    ),
    # With extremes
    ClusteringTestCase(
        id="hierarchical_medoid_8clusters_extremes_append",
        method="hierarchical",
        representation="medoid",
        extreme_method="append",
        extreme_columns=["Load"],
    ),
    ClusteringTestCase(
        id="hierarchical_medoid_8clusters_extremes_newcluster",
        method="hierarchical",
        representation="medoid",
        extreme_method="new_cluster",
        extreme_columns=["Load"],
    ),
    # Replace extreme method
    ClusteringTestCase(
        id="hierarchical_medoid_8clusters_extremes_replace",
        method="hierarchical",
        representation="medoid",
        extreme_method="replace",
        extreme_columns=["Load"],
    ),
    # Combined features
    ClusteringTestCase(
        id="hierarchical_medoid_8clusters_12segments_extremes_append",
        method="hierarchical",
        representation="medoid",
        n_segments=12,
        extreme_method="append",
        extreme_columns=["Load"],
    ),
    ClusteringTestCase(
        id="hierarchical_medoid_8clusters_6segments_extremes_newcluster",
        method="hierarchical",
        representation="medoid",
        n_segments=6,
        extreme_method="new_cluster",
        extreme_columns=["Load"],
    ),
    ClusteringTestCase(
        id="hierarchical_medoid_8clusters_12segments_extremes_replace",
        method="hierarchical",
        representation="medoid",
        n_segments=12,
        extreme_method="replace",
        extreme_columns=["Load"],
    ),
]


def get_test_ids():
    """Get test IDs for parametrization."""
    return [tc.id for tc in TEST_CASES]


@pytest.fixture(scope="module")
def input_data():
    """Load the input data used for all tests."""
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)


@pytest.fixture(scope="module")
def fixtures_dir():
    """Path to the clustering e2e fixtures directory."""
    return Path(__file__).parent / "data" / "clustering_e2e"


def run_aggregation(data: pd.DataFrame, test_case: ClusteringTestCase):
    """Run aggregation with the specified test case configuration."""
    # Set seed for reproducibility with stochastic methods (kmeans, kmaxoids)
    if test_case.method in ("kmeans", "kmaxoids"):
        set_random_seed()

    # Build cluster config
    cluster_config = ClusterConfig(
        method=test_case.method,
        representation=test_case.representation,
    )

    # Build segment config if needed
    segment_config = None
    if test_case.n_segments is not None:
        segment_config = SegmentConfig(n_segments=test_case.n_segments)

    # Build extreme config if needed
    extreme_config = None
    if test_case.extreme_method is not None:
        extreme_config = ExtremeConfig(
            method=test_case.extreme_method,
            max_value=test_case.extreme_columns,
        )

    return aggregate(
        data,
        n_clusters=8,
        cluster=cluster_config,
        segments=segment_config,
        extremes=extreme_config,
    )


class TestClusteringE2E:
    """End-to-end tests for clustering configurations."""

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=get_test_ids())
    def test_cluster_representatives(
        self, test_case: ClusteringTestCase, input_data, fixtures_dir
    ):
        """Test that cluster representatives match expected values."""
        # Load expected output
        expected_path = fixtures_dir / f"expected_{test_case.id}.csv"
        if not expected_path.exists():
            pytest.skip(f"Expected file not found: {expected_path}")

        # Determine index columns based on segmentation
        if test_case.n_segments is not None:
            index_col = [0, 1, 2]  # cluster, segment_step, segment_duration
        else:
            index_col = [0, 1]  # cluster, timestep

        expected = pd.read_csv(expected_path, index_col=index_col)

        # Run aggregation
        result = run_aggregation(input_data, test_case)
        actual = result.cluster_representatives

        # Compare values directly (cluster order should be deterministic)
        np.testing.assert_array_almost_equal(
            expected[actual.columns].values,
            actual.values,
            decimal=4,
            err_msg=f"Cluster representatives mismatch for {test_case.id}",
        )

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=get_test_ids())
    def test_cluster_weights(
        self, test_case: ClusteringTestCase, input_data, fixtures_dir
    ):
        """Test that cluster weights match expected values."""
        # Load expected metadata
        meta_path = fixtures_dir / f"meta_{test_case.id}.json"
        if not meta_path.exists():
            pytest.skip(f"Metadata file not found: {meta_path}")

        with open(meta_path) as f:
            metadata = json.load(f)

        # Run aggregation
        result = run_aggregation(input_data, test_case)

        # Compare cluster weights (sum should match)
        expected_weights = metadata["cluster_weights"]
        actual_weights = result.cluster_weights

        # Total weight should match number of original periods
        expected_total = sum(expected_weights.values())
        actual_total = sum(actual_weights.values())
        assert actual_total == expected_total, (
            f"Total weights mismatch: expected {expected_total}, got {actual_total}"
        )

        # Number of clusters should match
        assert len(actual_weights) == len(expected_weights), (
            f"Number of clusters mismatch: expected {len(expected_weights)}, "
            f"got {len(actual_weights)}"
        )

    @pytest.mark.parametrize("test_case", TEST_CASES, ids=get_test_ids())
    def test_accuracy_metrics(
        self, test_case: ClusteringTestCase, input_data, fixtures_dir
    ):
        """Test that accuracy metrics are within expected bounds."""
        # Load expected metadata
        meta_path = fixtures_dir / f"meta_{test_case.id}.json"
        if not meta_path.exists():
            pytest.skip(f"Metadata file not found: {meta_path}")

        with open(meta_path) as f:
            metadata = json.load(f)

        # Run aggregation
        result = run_aggregation(input_data, test_case)

        # Check RMSE is within tolerance of expected
        expected_rmse = metadata["accuracy"]["rmse"]
        for col, expected_val in expected_rmse.items():
            actual_val = result.accuracy.rmse[col]
            # Allow 1% relative tolerance for floating point comparisons
            np.testing.assert_allclose(
                actual_val,
                expected_val,
                rtol=0.01,
                err_msg=f"RMSE mismatch for column {col} in {test_case.id}",
            )

        # Check MAE is within tolerance of expected
        expected_mae = metadata["accuracy"]["mae"]
        for col, expected_val in expected_mae.items():
            actual_val = result.accuracy.mae[col]
            np.testing.assert_allclose(
                actual_val,
                expected_val,
                rtol=0.01,
                err_msg=f"MAE mismatch for column {col} in {test_case.id}",
            )


# Subset of test cases for transfer tests (skip slow or unsupported ones)
# - kmedoids/contiguous: too slow for repeated runs
# - replace extreme method: creates hybrid representation (some columns from medoid,
#   some from extreme period) that cannot be perfectly reproduced during transfer
TRANSFER_TEST_CASES = [
    tc
    for tc in TEST_CASES
    if tc.method not in ("kmedoids", "contiguous") and tc.extreme_method != "replace"
]


def get_transfer_test_ids():
    """Get test IDs for transfer test parametrization."""
    return [tc.id for tc in TRANSFER_TEST_CASES]


class TestClusteringTransfer:
    """Tests for clustering transfer and reproducibility."""

    @pytest.mark.parametrize(
        "test_case", TRANSFER_TEST_CASES, ids=get_transfer_test_ids()
    )
    def test_apply_produces_identical_results(
        self, test_case: ClusteringTestCase, input_data
    ):
        """Test that applying clustering to same data produces identical results."""
        # Run initial aggregation
        result1 = run_aggregation(input_data, test_case)

        # Apply clustering to same data
        result2 = result1.clustering.apply(input_data)

        # Results should be identical
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
            check_exact=False,
            atol=1e-10,
        )

    @pytest.mark.parametrize(
        "test_case", TRANSFER_TEST_CASES, ids=get_transfer_test_ids()
    )
    def test_json_roundtrip_produces_identical_results(
        self, test_case: ClusteringTestCase, input_data, tmp_path
    ):
        """Test that JSON save/load/apply produces identical results."""
        from tsam import ClusteringResult

        # Run initial aggregation
        result1 = run_aggregation(input_data, test_case)

        # Save to JSON
        json_path = tmp_path / f"clustering_{test_case.id}.json"
        result1.clustering.to_json(str(json_path))

        # Load and apply
        clustering = ClusteringResult.from_json(str(json_path))
        result2 = clustering.apply(input_data)

        # Results should be identical
        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
            check_exact=False,
            atol=1e-10,
        )

    @pytest.mark.parametrize(
        "test_case", TRANSFER_TEST_CASES, ids=get_transfer_test_ids()
    )
    def test_reconstruction_shape(self, test_case: ClusteringTestCase, input_data):
        """Test that reconstructed data has same shape as input."""
        result = run_aggregation(input_data, test_case)
        reconstructed = result.reconstructed

        assert reconstructed.shape == input_data.shape, (
            f"Reconstruction shape mismatch: expected {input_data.shape}, "
            f"got {reconstructed.shape}"
        )
        assert list(reconstructed.columns) == list(input_data.columns)

    def test_apply_to_different_columns(self, input_data):
        """Test applying clustering from subset to full data."""
        # Cluster on single column
        wind_only = input_data[["Wind"]]
        result_wind = aggregate(wind_only, n_clusters=8)

        # Apply to full data
        result_full = result_wind.clustering.apply(input_data)

        # Cluster assignments should be identical
        assert list(result_wind.cluster_assignments) == list(
            result_full.cluster_assignments
        )

        # Full result should have all columns
        assert list(result_full.cluster_representatives.columns) == list(
            input_data.columns
        )

    def test_segmentation_preserved_in_transfer(self, input_data, tmp_path):
        """Test that segmentation info is preserved through JSON roundtrip."""
        from tsam import ClusteringResult

        # Run with segmentation
        result1 = aggregate(
            input_data,
            n_clusters=8,
            segments=SegmentConfig(n_segments=6),
        )

        # Save and load
        json_path = tmp_path / "clustering_seg.json"
        result1.clustering.to_json(str(json_path))
        clustering = ClusteringResult.from_json(str(json_path))

        # Apply to same data
        result2 = clustering.apply(input_data)

        # Segmentation should be preserved
        assert result2.n_segments == result1.n_segments
        assert result2.segment_durations == result1.segment_durations

        pd.testing.assert_frame_equal(
            result1.cluster_representatives,
            result2.cluster_representatives,
            check_exact=False,
            atol=1e-10,
        )


def generate_fixtures(output_dir: Path | None = None):
    """Generate expected fixture files for all test cases.

    This function is used to create the initial fixture files.
    Run manually when setting up tests or updating expected values.

    Usage:
        python -c "from test_clustering_e2e import generate_fixtures; generate_fixtures()"
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "data" / "clustering_e2e"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load input data
    input_data = pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True)

    for test_case in TEST_CASES:
        print(f"Generating fixtures for {test_case.id}...")

        # Run aggregation
        result = run_aggregation(input_data, test_case)

        # Save cluster representatives CSV
        csv_path = output_dir / f"expected_{test_case.id}.csv"
        result.cluster_representatives.to_csv(csv_path)

        # Build and save metadata JSON
        metadata = {
            "config": {
                "method": test_case.method,
                "representation": test_case.representation,
                "n_clusters": 8,
            },
            "cluster_weights": {
                str(k): int(v) for k, v in result.cluster_weights.items()
            },
            "accuracy": {
                "rmse": {col: float(val) for col, val in result.accuracy.rmse.items()},
                "mae": {col: float(val) for col, val in result.accuracy.mae.items()},
            },
            "n_original_periods": len(result.cluster_assignments),
        }

        # Add segmentation info if applicable
        if test_case.n_segments is not None:
            metadata["config"]["n_segments"] = test_case.n_segments

        # Add extremes info if applicable
        if test_case.extreme_method is not None:
            metadata["config"]["extreme_method"] = test_case.extreme_method
            metadata["config"]["extreme_columns"] = test_case.extreme_columns

        meta_path = output_dir / f"meta_{test_case.id}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"Generated fixtures in {output_dir}")


if __name__ == "__main__":
    generate_fixtures()
