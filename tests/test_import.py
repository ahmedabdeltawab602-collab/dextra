"""Smoke tests: the package must import and expose its public API.

This is the single most important test — the previous version of dextra
raised ``NameError`` on import because ``List`` was used in annotations
without being imported.
"""

from __future__ import annotations


def test_package_imports_cleanly():
    import dextra  # noqa: F401


def test_public_api_exports():
    import dextra

    for name in [
        "__version__",
        "DEFAULT_BOX_COLORS",
        "describe_numeric",
        "plot_histograms",
        "plot_boxplots",
        "numdesc",
        "hister",
        "boxpl",
    ]:
        assert hasattr(dextra, name), f"dextra should expose {name!r}"


def test_aliases_point_to_new_names():
    import dextra

    assert dextra.numdesc is dextra.describe_numeric
    assert dextra.hister is dextra.plot_histograms
    assert dextra.boxpl is dextra.plot_boxplots
