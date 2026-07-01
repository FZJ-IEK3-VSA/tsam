"""Tests for the ``Distribution.scope`` rename and its segment semantics.

``scope="cluster"`` was renamed to ``scope="local"`` because "cluster" is
misleading for segment representations, where the group whose distribution is
preserved is a segment, not a cluster. ``"cluster"`` remains accepted as a
deprecated, behaviourally-identical alias.

For a segment (a single value per attribute) the ``Distribution`` options
collapse: ``scope="local"`` equals ``"mean"`` and ``preserve_minmax`` cannot
take effect (issues #378, #382).
"""

import warnings

import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, Distribution, SegmentConfig, aggregate

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:KMeans is known to have a memory leak on Windows with MKL.*:UserWarning"
    ),
]


class TestScopeRename:
    """The ``"cluster"`` -> ``"local"`` rename and its deprecated alias."""

    def test_default_scope_is_local(self):
        assert Distribution().scope == "local"

    def test_local_and_global_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert Distribution(scope="local").scope == "local"
            assert Distribution(scope="global").scope == "global"

    def test_cluster_alias_warns_and_normalizes_to_local(self):
        with pytest.warns(DeprecationWarning, match='scope="cluster" is deprecated'):
            dist = Distribution(scope="cluster")
        assert dist.scope == "local"

    def test_to_dict_omits_default_and_normalized_scope(self):
        # "local" is the default, so it is omitted from the serialized form.
        assert "scope" not in Distribution(scope="local").to_dict()
        with pytest.warns(DeprecationWarning):
            assert "scope" not in Distribution(scope="cluster").to_dict()
        # "global" is non-default and is serialized.
        assert Distribution(scope="global").to_dict()["scope"] == "global"

    def test_from_dict_accepts_deprecated_and_missing_scope(self):
        assert Distribution.from_dict({}).scope == "local"
        assert Distribution.from_dict({"scope": "global"}).scope == "global"
        with pytest.warns(DeprecationWarning):
            assert Distribution.from_dict({"scope": "cluster"}).scope == "local"

    def test_cluster_alias_is_behaviourally_identical_to_local(self):
        """The deprecated alias produces bit-for-bit identical aggregation output."""
        raw = pd.read_csv(TESTDATA_CSV, index_col=0)

        def run(scope):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return aggregate(
                    raw,
                    n_clusters=8,
                    period_duration=24,
                    cluster=ClusterConfig(
                        method="hierarchical",
                        use_duration_curves=False,
                        representation=Distribution(scope=scope),
                    ),
                    preserve_column_means=False,
                ).reconstructed

        pd.testing.assert_frame_equal(run("cluster"), run("local"))


class TestSegmentDistribution:
    """Distribution options collapse for segments (single value per attribute)."""

    @staticmethod
    def _reconstructed(seg_rep):
        raw = pd.read_csv(TESTDATA_CSV, index_col=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return aggregate(
                raw,
                n_clusters=8,
                period_duration=24,
                cluster=ClusterConfig(method="hierarchical", use_duration_curves=False),
                segments=SegmentConfig(n_segments=6, representation=seg_rep),
                preserve_column_means=False,
            ).reconstructed

    def test_preserve_minmax_warns_on_local_segment(self):
        with pytest.warns(UserWarning, match="preserve_minmax has no effect"):
            SegmentConfig(
                n_segments=4,
                representation=Distribution(scope="local", preserve_minmax=True),
            )

    def test_preserve_minmax_does_not_warn_on_global_segment(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # scope="global" pins segment min/max, so preserve_minmax is honoured.
            SegmentConfig(
                n_segments=4,
                representation=Distribution(scope="global", preserve_minmax=True),
            )
            # scope="local" without preserve_minmax is a plain (mean-equivalent) rep.
            SegmentConfig(n_segments=4, representation=Distribution(scope="local"))

    def test_deprecated_cluster_alias_also_warns_on_segment(self):
        # "cluster" normalizes to "local" (DeprecationWarning), then the segment
        # preserve_minmax no-op warning fires too.
        with pytest.warns((DeprecationWarning, UserWarning)):
            SegmentConfig(
                n_segments=4,
                representation=Distribution(scope="cluster", preserve_minmax=True),
            )

    def test_local_segment_is_mean(self):
        """scope="local" segment representation is equivalent to "mean" (#382)."""
        pd.testing.assert_frame_equal(
            self._reconstructed(Distribution(scope="local")),
            self._reconstructed("mean"),
        )

    def test_local_minmax_segment_is_mean(self):
        """preserve_minmax is a silent no-op for a scope="local" segment (#378)."""
        pd.testing.assert_frame_equal(
            self._reconstructed(Distribution(scope="local", preserve_minmax=True)),
            self._reconstructed("mean"),
        )

    def test_global_segment_differs_from_mean(self):
        """scope="global" is the only Distribution segment option distinct from mean."""
        assert not self._reconstructed(Distribution(scope="global")).equals(
            self._reconstructed("mean")
        )
