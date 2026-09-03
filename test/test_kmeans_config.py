"""Tests for the KMeans method config."""

import pickle
from unittest.mock import patch

import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, KMeans, aggregate


@pytest.fixture
def sample_data():
    return pd.read_csv(TESTDATA_CSV, index_col=0, parse_dates=True).iloc[: 24 * 30]


def _n_init_used(data, cluster):
    """The n_init scikit-learn was actually constructed with."""
    import sklearn.cluster

    real = sklearn.cluster.KMeans
    captured = {}

    def spy(*args, **kwargs):
        captured["n_init"] = kwargs.get("n_init")
        return real(*args, **kwargs)

    with patch("sklearn.cluster.KMeans", side_effect=spy):
        aggregate(data, 4, cluster=cluster)
    return captured["n_init"]


class TestDefaultsUnchanged:
    """The point of the config is to expose the knob, not to move it."""

    def test_bare_string_uses_library_default(self, sample_data):
        assert _n_init_used(sample_data, ClusterConfig(method="kmeans")) == 100

    def test_empty_config_matches_bare_string(self, sample_data):
        assert _n_init_used(sample_data, ClusterConfig(method=KMeans())) == 100

    def test_duration_curves_keep_their_own_default(self, sample_data):
        cluster = ClusterConfig(method=KMeans(), use_duration_curves=True)
        assert _n_init_used(sample_data, cluster) == 30

    def test_config_form_clusters_the_same_way_as_the_string(self, sample_data):
        """Same code path, so the same shape of result.

        Not asserted bit-for-bit: tsam passes no ``random_state`` to
        scikit-learn, so k-means picks its initial centers from an unseeded
        RNG and two runs of *either* form can land on different local optima.
        What this pins is that the config form takes the same path as the
        string, which the n_init assertions above establish exactly.
        """
        from_string = aggregate(sample_data, 4, cluster=ClusterConfig(method="kmeans"))
        from_config = aggregate(sample_data, 4, cluster=ClusterConfig(method=KMeans()))
        assert from_string.n_clusters == from_config.n_clusters
        assert len(from_string.cluster_assignments) == len(
            from_config.cluster_assignments
        )


class TestNInitForwarding:
    def test_explicit_value_reaches_sklearn(self, sample_data):
        cluster = ClusterConfig(method=KMeans(n_init=3))
        assert _n_init_used(sample_data, cluster) == 3

    def test_explicit_value_overrides_duration_curve_default(self, sample_data):
        cluster = ClusterConfig(method=KMeans(n_init=5), use_duration_curves=True)
        assert _n_init_used(sample_data, cluster) == 5

    def test_lower_n_init_still_produces_a_valid_clustering(self, sample_data):
        result = aggregate(
            sample_data, 4, cluster=ClusterConfig(method=KMeans(n_init=1))
        )
        assert result.n_clusters == 4
        assert len(result.cluster_assignments) == 30


class TestRandomState:
    """The seed is what makes a k-means run repeatable at all."""

    def test_same_seed_gives_identical_clusters(self, sample_data):
        cluster = ClusterConfig(method=KMeans(random_state=42))
        first = aggregate(sample_data, 4, cluster=cluster)
        second = aggregate(sample_data, 4, cluster=cluster)
        assert list(first.cluster_assignments) == list(second.cluster_assignments)

    def test_seed_survives_a_lowered_n_init(self, sample_data):
        cluster = ClusterConfig(method=KMeans(n_init=1, random_state=7))
        first = aggregate(sample_data, 4, cluster=cluster)
        second = aggregate(sample_data, 4, cluster=cluster)
        assert list(first.cluster_assignments) == list(second.cluster_assignments)

    def test_seed_reaches_sklearn(self, sample_data):
        import sklearn.cluster

        real, captured = sklearn.cluster.KMeans, {}

        def spy(*args, **kwargs):
            captured["random_state"] = kwargs.get("random_state")
            return real(*args, **kwargs)

        with patch("sklearn.cluster.KMeans", side_effect=spy):
            aggregate(
                sample_data, 4, cluster=ClusterConfig(method=KMeans(random_state=3))
            )
        assert captured["random_state"] == 3

    def test_default_passes_none_as_before(self, sample_data):
        import sklearn.cluster

        real, captured = sklearn.cluster.KMeans, {}

        def spy(*args, **kwargs):
            captured["random_state"] = kwargs.get("random_state")
            return real(*args, **kwargs)

        with patch("sklearn.cluster.KMeans", side_effect=spy):
            aggregate(sample_data, 4, cluster=ClusterConfig(method="kmeans"))
        assert captured["random_state"] is None


class TestConfigObject:
    def test_method_name_is_the_canonical_string(self):
        assert ClusterConfig(method=KMeans(n_init=7)).method_name == "kmeans"

    def test_default_representation_matches_the_string_form(self):
        assert ClusterConfig(method=KMeans()).get_representation() == (
            ClusterConfig(method="kmeans").get_representation()
        )

    def test_is_hashable(self):
        assert hash(ClusterConfig(method=KMeans(n_init=3))) is not None

    def test_equality(self):
        assert KMeans(n_init=3) == KMeans(n_init=3)
        assert KMeans(n_init=3) != KMeans(n_init=4)
        assert KMeans() != KMeans(n_init=100)
        assert KMeans(random_state=1) != KMeans(random_state=2)

    def test_is_frozen(self):
        with pytest.raises(Exception):
            KMeans().n_init = 5


class TestSerialization:
    def test_roundtrip_with_explicit_value(self):
        cluster = ClusterConfig(method=KMeans(n_init=12))
        assert ClusterConfig.from_dict(cluster.to_dict()) == cluster

    def test_roundtrip_with_default(self):
        cluster = ClusterConfig(method=KMeans())
        assert ClusterConfig.from_dict(cluster.to_dict()) == cluster

    def test_dict_omits_unset_fields(self):
        assert KMeans().to_dict() == {"type": "kmeans"}

    def test_dict_carries_seed(self):
        assert KMeans(random_state=4).to_dict() == {"type": "kmeans", "random_state": 4}

    def test_roundtrip_with_seed(self):
        cluster = ClusterConfig(method=KMeans(n_init=5, random_state=11))
        assert ClusterConfig.from_dict(cluster.to_dict()) == cluster

    def test_dict_carries_explicit_n_init(self):
        assert KMeans(n_init=9).to_dict() == {"type": "kmeans", "n_init": 9}

    def test_bare_string_still_deserializes(self):
        assert ClusterConfig.from_dict({"method": "kmeans"}).method == "kmeans"

    def test_pickles(self):
        cluster = ClusterConfig(method=KMeans(n_init=6))
        assert pickle.loads(pickle.dumps(cluster)) == cluster
