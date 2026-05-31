"""Tests for the Phase 5.5 consolidation fixes (IMPROVEMENTS.md items)."""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import dextra as dx


def test_modes_dash_for_all_unique():
    # continuous column: every value unique -> modes must collapse to "-"
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cont": rng.normal(0, 1, 200)})
    out = dx.describe_numeric(df, show=False, return_df=True)
    assert out.loc["modes", "cont"] == "-"


def test_modes_capped_when_many():
    df = pd.DataFrame({"x": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]})
    out = dx.describe_numeric(df, show=False, return_df=True)
    assert "more)" in out.loc["modes", "x"]


def test_ddof_changes_std():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(0, 1, 100)})
    s1 = dx.describe_numeric(df, ddof=1, raw=True, show=False, return_df=True)
    s0 = dx.describe_numeric(df, ddof=0, raw=True, show=False, return_df=True)
    assert float(s1.loc["std", "a"]) != float(s0.loc["std", "a"])


def test_negative_ddof_raises():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        dx.describe_numeric(df, ddof=-1, show=False)


def test_plot_histograms_string_bins():
    pytest.importorskip("seaborn")
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(0, 1, 120)})
    fig = dx.plot_histograms(df, bins="auto", show=False, return_fig=True)
    assert fig is not None


def test_functions_index_lists_api():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        dx.functions()
    text = buf.getvalue()
    assert "describe_numeric" in text
    assert "redundancy" in text
