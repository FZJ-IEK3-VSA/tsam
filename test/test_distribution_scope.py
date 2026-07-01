"""Tests for the ``Distribution.scope`` rename ("cluster" -> "local").

``scope="cluster"`` was renamed to ``scope="local"`` because "cluster" is
misleading for segment representations, where the group whose distribution is
preserved is a segment, not a cluster. ``"cluster"`` remains accepted as a
deprecated, behaviourally-identical alias.
"""

import warnings

import pandas as pd
import pytest

from conftest import TESTDATA_CSV
from tsam import ClusterConfig, Distribution, aggregate

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:KMeans is known to have a memory leak on Windows with MKL.*:UserWarning"
    ),
]


def test_default_scope_is_local():
    assert Distribution().scope == "local"


def test_local_and_global_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert Distribution(scope="local").scope == "local"
        assert Distribution(scope="global").scope == "global"


def test_cluster_alias_warns_and_normalizes_to_local():
    with pytest.warns(DeprecationWarning, match='scope="cluster" is deprecated'):
        dist = Distribution(scope="cluster")
    assert dist.scope == "local"


def test_to_dict_omits_default_and_normalized_scope():
    # "local" is the default, so it is omitted from the serialized form.
    assert "scope" not in Distribution(scope="local").to_dict()
    with pytest.warns(DeprecationWarning):
        assert "scope" not in Distribution(scope="cluster").to_dict()
    # "global" is non-default and is serialized.
    assert Distribution(scope="global").to_dict()["scope"] == "global"


def test_from_dict_accepts_deprecated_and_missing_scope():
    assert Distribution.from_dict({}).scope == "local"
    assert Distribution.from_dict({"scope": "global"}).scope == "global"
    with pytest.warns(DeprecationWarning):
        assert Distribution.from_dict({"scope": "cluster"}).scope == "local"


def test_cluster_alias_is_behaviourally_identical_to_local():
    """The deprecated alias must produce bit-for-bit identical aggregation output."""
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
