"""Local pytest config for the benchmark suite (not collected by default CI)."""

import pytest

TIERS = (
    (
        "slow",
        "--slow",
        "Run slow-tier benchmarks (MILP k-medoids, large column counts).",
    ),
    (
        "large",
        "--large",
        "Run production-sized benchmarks (multi-year x hundreds of columns).",
    ),
)


def pytest_addoption(parser):
    for _marker, flag, help_text in TIERS:
        parser.addoption(flag, action="store_true", default=False, help=help_text)


def pytest_configure(config):
    for marker, flag, _help_text in TIERS:
        config.addinivalue_line(
            "markers", f"{marker}: {marker}-tier benchmark, only runs with {flag}"
        )


def pytest_collection_modifyitems(config, items):
    for marker, flag, _help_text in TIERS:
        if config.getoption(flag):
            continue
        skip = pytest.mark.skip(reason=f"{marker} tier: pass {flag} to run")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)
