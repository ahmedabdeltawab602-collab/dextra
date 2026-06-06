"""dextra Phase 8 -- time series (tsdecomp / tsstat / tsfcast).

Stage 8.1 covers tsdecomp: both input modes (series and artifact), classical
additive + multiplicative decomposition, period inference, the JSON-safe
descriptor (no estimator), figure rendering, immutability of the input
DataFrame, and the guard-error paths. The classical path is
dependency-free (numpy / pandas / matplotlib); the STL path is skipped if
statsmodels is absent.
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import dextra as dx

KW = dict(show=False, plot=False)


def _json_ok(report):
    json.loads(json.dumps(report))


@pytest.fixture
def ts_df():
    """Monthly series: linear trend + 12-period seasonality + small noise."""
    n = 72
    rng = np.random.default_rng(0)
    t = np.arange(n)
    y = 10 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.4, n)
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    return pd.DataFrame({"sales": y}, index=idx)


@pytest.fixture
def ts_pos_df():
    """Strictly-positive monthly series for the multiplicative model."""
    n = 72
    t = np.arange(n)
    y = (10 + 0.5 * t) * (1 + 0.3 * np.sin(2 * np.pi * t / 12))
    idx = pd.date_range("2018-01-01", periods=n, freq="MS")
    return pd.DataFrame({"v": y}, index=idx)


# ---------------------------------------------------------------------------
# exports / aliases
# ---------------------------------------------------------------------------

def test_exports_and_no_underscore_names():
    assert callable(dx.tsdecomp)
    assert not hasattr(dx, "ts_decompose")  # underscore-free names only


# ---------------------------------------------------------------------------
# tsdecomp -- series mode
# ---------------------------------------------------------------------------

def test_decompose_series_mode_infers_period(ts_df):
    comp, rep = dx.tsdecomp(ts_df, value="sales", return_params=True, **KW)
    assert list(comp.columns) == ["observed", "trend", "seasonal", "resid"]
    assert len(comp) == len(ts_df)
    assert rep["period"] == 12             # inferred from the monthly index
    assert rep["model"] == "additive"
    assert rep["metadata"]["freq"] == "MS"
    assert rep["metadata"]["input_mode"] == "series"


def test_decompose_additive_reconstruction(ts_df):
    comp = dx.tsdecomp(ts_df, value="sales", **KW)
    m = comp.dropna()
    recon = m["trend"] + m["seasonal"] + m["resid"]
    assert np.allclose(recon, m["observed"])


def test_decompose_strengths_high_for_strong_signal(ts_df):
    _, rep = dx.tsdecomp(ts_df, value="sales", return_params=True, **KW)
    assert rep["metrics"]["strength_trend"] > 0.95
    assert rep["metrics"]["strength_seasonal"] > 0.90


def test_decompose_explicit_period_without_time():
    n = 48
    t = np.arange(n)
    y = 1 + 0.2 * t + 3 * np.sin(2 * np.pi * t / 12)
    df = pd.DataFrame({"sales": y})           # no datetime index
    comp = dx.tsdecomp(df, value="sales", period=12, **KW)
    assert len(comp) == n


def test_decompose_array_like_value(ts_df):
    comp = dx.tsdecomp(ts_df, value=ts_df["sales"].to_numpy(),
                           period=12, **KW)
    assert len(comp) == len(ts_df)


def test_decompose_multiplicative_reconstruction(ts_pos_df):
    comp, rep = dx.tsdecomp(ts_pos_df, value="v", model="multiplicative",
                                return_params=True, **KW)
    assert rep["model"] == "multiplicative"
    m = comp.dropna()
    recon = m["trend"] * m["seasonal"] * m["resid"]
    assert np.allclose(recon, m["observed"])


# ---------------------------------------------------------------------------
# descriptor / immutability / idempotency / artifact mode
# ---------------------------------------------------------------------------

def test_decompose_report_is_json_safe(ts_df):
    _, rep = dx.tsdecomp(ts_df, value="sales", return_params=True, **KW)
    assert "estimator" not in rep
    _json_ok(rep)
    assert rep["function"] == "tsdecomp"
    assert rep["task"] == "timeseries"


def test_decompose_does_not_mutate_input(ts_df):
    before = ts_df.copy(deep=True)
    dx.tsdecomp(ts_df, value="sales", **KW)
    pd.testing.assert_frame_equal(ts_df, before)
    assert "dextra_audit" not in ts_df.attrs


def test_decompose_idempotent(ts_df):
    c1 = dx.tsdecomp(ts_df, value="sales", **KW)
    c2 = dx.tsdecomp(ts_df, value="sales", **KW)
    pd.testing.assert_frame_equal(c1, c2)


def test_decompose_artifact_mode_reproduces(ts_df):
    comp, rep = dx.tsdecomp(ts_df, value="sales", return_params=True, **KW)
    comp2 = dx.tsdecomp(ts_df, params=rep, **KW)
    pd.testing.assert_frame_equal(comp, comp2)


def test_decompose_artifact_mode_bad_params(ts_df):
    with pytest.raises(ValueError):
        dx.tsdecomp(ts_df, params=["not", "a", "dict"], **KW)


# ---------------------------------------------------------------------------
# return-flag combinations / figure
# ---------------------------------------------------------------------------

def test_decompose_return_fig(ts_df):
    comp, rep, fig = dx.tsdecomp(
        ts_df, value="sales", return_params=True, return_fig=True,
        show=False, plot=True)
    assert fig is not None
    assert len(fig.axes) == 4
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_decompose_return_df_false_returns_report(ts_df):
    rep = dx.tsdecomp(ts_df, value="sales", return_df=False,
                          return_params=True, **KW)
    assert isinstance(rep, dict)


# ---------------------------------------------------------------------------
# guard / error paths
# ---------------------------------------------------------------------------

def test_decompose_requires_value(ts_df):
    with pytest.raises(ValueError):
        dx.tsdecomp(ts_df, **KW)


def test_decompose_value_not_found(ts_df):
    with pytest.raises(KeyError):
        dx.tsdecomp(ts_df, value="missing", **KW)


def test_decompose_period_not_inferrable():
    df = pd.DataFrame({"sales": np.arange(40.0)})   # no datetime index
    with pytest.raises(ValueError):
        dx.tsdecomp(df, value="sales", **KW)


def test_decompose_missing_values_rejected(ts_df):
    df = ts_df.copy()
    arr = df["sales"].to_numpy().copy()
    arr[5] = np.nan
    df["sales"] = arr
    with pytest.raises(ValueError):
        dx.tsdecomp(df, value="sales", **KW)


def test_decompose_too_few_periods():
    df = pd.DataFrame({"sales": np.arange(18.0)})
    with pytest.raises(ValueError):
        dx.tsdecomp(df, value="sales", period=12, **KW)


def test_decompose_multiplicative_requires_positive(ts_df):
    df = ts_df.copy()
    arr = df["sales"].to_numpy().copy()
    arr[0] = -1.0
    df["sales"] = arr
    with pytest.raises(ValueError):
        dx.tsdecomp(df, value="sales", model="multiplicative", **KW)


def test_decompose_bad_model(ts_df):
    with pytest.raises(ValueError):
        dx.tsdecomp(ts_df, value="sales", model="nonsense", **KW)


def test_decompose_bad_method(ts_df):
    with pytest.raises(ValueError):
        dx.tsdecomp(ts_df, value="sales", method="nonsense", **KW)


def test_decompose_bad_period_value(ts_df):
    with pytest.raises(ValueError):
        dx.tsdecomp(ts_df, value="sales", period=1, **KW)


# ---------------------------------------------------------------------------
# STL path (requires statsmodels)
# ---------------------------------------------------------------------------

def test_decompose_stl_path(ts_df):
    pytest.importorskip("statsmodels")
    comp, rep = dx.tsdecomp(ts_df, value="sales", method="stl",
                                return_params=True, **KW)
    assert rep["method"] == "stl"
    assert list(comp.columns) == ["observed", "trend", "seasonal", "resid"]
    assert len(comp) == len(ts_df)
