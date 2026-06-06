"""dextra Phase 8 -- time series (tsdecomp / tsstat / tsfcast).

Covers tsdecomp (8.1), tsstat (8.2) and tsfcast (8.3) across both input modes
(series and artifact): decomposition reconstruction and strengths; ADF/KPSS
stationarity with suggested differencing; baseline forecasting validated on a
held-out tail. Each checks the JSON-safe descriptor, figure rendering,
immutability and the guard-error paths. Dependency-free paths run everywhere;
the STL and ADF/KPSS paths are skipped when statsmodels is absent.
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


# ===========================================================================
# 8.2  tsstat  --  stationarity (ADF / KPSS), needs statsmodels for the tests
# ===========================================================================

def test_tsstat_exports_no_underscore():
    assert callable(dx.tsstat)
    assert not hasattr(dx, "ts_stationarity")   # underscore-free names only


def test_tsstat_bad_regression(ts_df):
    # validated before the statsmodels gate, so runs without statsmodels
    with pytest.raises(ValueError):
        dx.tsstat(ts_df, value="sales", regression="bad", **KW)


def test_tsstat_bad_alpha(ts_df):
    with pytest.raises(ValueError):
        dx.tsstat(ts_df, value="sales", alpha=2.0, **KW)


def test_tsstat_bad_max_diff(ts_df):
    with pytest.raises(ValueError):
        dx.tsstat(ts_df, value="sales", max_diff=-1, **KW)


def test_tsstat_white_noise_is_stationary():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(3)
    n = 200
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    df = pd.DataFrame({"x": rng.normal(0, 1, n)}, index=idx)
    tbl, rep = dx.tsstat(df, value="x", return_params=True, **KW)
    assert list(tbl.index) == ["ADF", "KPSS"]
    assert rep["metrics"]["suggested_d"] == 0
    assert rep["metrics"]["adf"]["pvalue"] < 0.05      # rejects unit root


def test_tsstat_random_walk_needs_differencing():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(4)
    n = 200
    rw = np.cumsum(rng.normal(0, 1, n))
    idx = pd.date_range("2010-01-01", periods=n, freq="MS")
    df = pd.DataFrame({"x": rw}, index=idx)
    _, rep = dx.tsstat(df, value="x", return_params=True, **KW)
    assert rep["metrics"]["suggested_d"] >= 1
    assert rep["metrics"]["adf"]["pvalue"] > 0.05      # cannot reject unit root


def test_tsstat_descriptor_json_safe():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(5)
    n = 120
    df = pd.DataFrame({"x": rng.normal(0, 1, n)},
                      index=pd.date_range("2010-01-01", periods=n, freq="MS"))
    _, rep = dx.tsstat(df, value="x", return_params=True, **KW)
    assert rep["function"] == "tsstat"
    assert rep["task"] == "timeseries"
    assert "estimator" not in rep
    _json_ok(rep)
    assert {"adf", "kpss", "verdict", "suggested_d", "differencing_path"} \
        <= set(rep["metrics"])


def test_tsstat_does_not_mutate_input():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(6)
    n = 120
    df = pd.DataFrame({"x": rng.normal(0, 1, n)},
                      index=pd.date_range("2010-01-01", periods=n, freq="MS"))
    before = df.copy(deep=True)
    dx.tsstat(df, value="x", **KW)
    pd.testing.assert_frame_equal(df, before)
    assert "dextra_audit" not in df.attrs


def test_tsstat_artifact_mode_reproduces():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame({"x": rng.normal(0, 1, n)},
                      index=pd.date_range("2010-01-01", periods=n, freq="MS"))
    tbl, rep = dx.tsstat(df, value="x", return_params=True, **KW)
    tbl2 = dx.tsstat(df, params=rep, **KW)
    pd.testing.assert_frame_equal(tbl, tbl2)


def test_tsstat_return_fig_three_panels():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(8)
    n = 120
    df = pd.DataFrame({"x": rng.normal(0, 1, n)},
                      index=pd.date_range("2010-01-01", periods=n, freq="MS"))
    tbl, rep, fig = dx.tsstat(df, value="x", return_params=True,
                              return_fig=True, show=False, plot=True)
    assert fig is not None
    assert len(fig.axes) == 3
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_tsstat_requires_value(ts_df):
    pytest.importorskip("statsmodels")
    with pytest.raises(ValueError):
        dx.tsstat(ts_df, **KW)


def test_tsstat_value_not_found(ts_df):
    pytest.importorskip("statsmodels")
    with pytest.raises(KeyError):
        dx.tsstat(ts_df, value="missing", **KW)


def test_tsstat_too_short():
    pytest.importorskip("statsmodels")
    df = pd.DataFrame({"x": np.arange(6.0)})
    with pytest.raises(ValueError):
        dx.tsstat(df, value="x", **KW)


def test_tsstat_constant_series():
    pytest.importorskip("statsmodels")
    df = pd.DataFrame({"x": np.ones(30)})
    with pytest.raises(ValueError):
        dx.tsstat(df, value="x", **KW)


def test_tsstat_missing_values_rejected():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(9)
    n = 40
    arr = rng.normal(0, 1, n)
    arr[5] = np.nan
    df = pd.DataFrame({"x": arr})
    with pytest.raises(ValueError):
        dx.tsstat(df, value="x", **KW)


# ===========================================================================
# 8.3  tsfcast  --  baseline forecast (dependency-free)
# ===========================================================================

def test_tsfcast_exports_no_underscore():
    assert callable(dx.tsfcast)
    assert not hasattr(dx, "ts_quick_forecast")   # underscore-free names only


def test_tsfcast_auto_picks_snaive_when_seasonal(ts_df):
    fc, rep = dx.tsfcast(ts_df, value="sales", horizon=12,
                         return_params=True, **KW)
    assert rep["method"] == "snaive"
    assert rep["metadata"]["method_requested"] == "auto"
    assert list(fc.columns) == ["forecast", "lower", "upper"]
    assert len(fc) == 12
    assert (fc["upper"] >= fc["lower"]).all()


def test_tsfcast_auto_picks_naive_without_season():
    df = pd.DataFrame({"x": np.cumsum(np.random.default_rng(1).normal(0, 1, 50))})
    fc, rep = dx.tsfcast(df, value="x", horizon=4, return_params=True, **KW)
    assert rep["method"] == "naive"
    assert isinstance(fc.index, pd.RangeIndex)


def test_tsfcast_snaive_repeats_last_season(ts_df):
    fc = dx.tsfcast(ts_df, value="sales", method="snaive", horizon=12, **KW)
    assert np.allclose(fc["forecast"].to_numpy(),
                       ts_df["sales"].to_numpy()[-12:])


def test_tsfcast_naive_mean_drift_formulas(ts_df):
    y = ts_df["sales"].to_numpy()
    n = len(y)
    assert np.allclose(dx.tsfcast(ts_df, value="sales", method="naive",
                                  horizon=3, **KW)["forecast"], y[-1])
    assert np.allclose(dx.tsfcast(ts_df, value="sales", method="mean",
                                  horizon=3, **KW)["forecast"], y.mean())
    slope = (y[-1] - y[0]) / (n - 1)
    fd = dx.tsfcast(ts_df, value="sales", method="drift", horizon=2, **KW)
    assert np.allclose(fd["forecast"].to_numpy(), y[-1] + slope * np.array([1, 2]))


def test_tsfcast_future_datetime_index(ts_df):
    fc = dx.tsfcast(ts_df, value="sales", method="naive", horizon=3, **KW)
    assert isinstance(fc.index, pd.DatetimeIndex)
    assert fc.index[0] == ts_df.index[-1] + pd.offsets.MonthBegin(1)


def test_tsfcast_descriptor_json_safe(ts_df):
    fc, rep = dx.tsfcast(ts_df, value="sales", method="drift",
                         return_params=True, **KW)
    assert rep["function"] == "tsfcast"
    assert rep["task"] == "timeseries"
    assert "estimator" not in rep
    _json_ok(rep)
    assert set(rep["metrics"]["validation"]) == {"MASE", "RMSE", "MAE", "MAPE"}


def test_tsfcast_does_not_mutate_input(ts_df):
    before = ts_df.copy(deep=True)
    dx.tsfcast(ts_df, value="sales", method="drift", **KW)
    pd.testing.assert_frame_equal(ts_df, before)
    assert "dextra_audit" not in ts_df.attrs


def test_tsfcast_artifact_mode_reproduces(ts_df):
    f1, rep = dx.tsfcast(ts_df, value="sales", method="drift", horizon=5,
                         return_params=True, **KW)
    f2 = dx.tsfcast(ts_df, params=rep, **KW)
    pd.testing.assert_frame_equal(f1, f2)


def test_tsfcast_compare_leaderboard(ts_df):
    board = dx.tsfcast(ts_df, value="sales", method="compare", **KW)
    assert list(board.columns) == ["MASE", "RMSE", "MAE", "MAPE"]
    assert "snaive" in board.index
    assert board["MASE"].is_monotonic_increasing      # sorted best-first


def test_tsfcast_compare_rejects_return_params(ts_df):
    with pytest.raises(ValueError):
        dx.tsfcast(ts_df, value="sales", method="compare",
                   return_params=True, **KW)


def test_tsfcast_return_fig_two_panels(ts_df):
    fc, rep, fig = dx.tsfcast(ts_df, value="sales", method="drift",
                              return_params=True, return_fig=True,
                              show=False, plot=True)
    assert fig is not None
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_tsfcast_compare_fig_two_panels(ts_df):
    board, fig = dx.tsfcast(ts_df, value="sales", method="compare",
                            return_fig=True, show=False, plot=True)
    assert len(fig.axes) == 2
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_tsfcast_bad_method(ts_df):
    with pytest.raises(ValueError):
        dx.tsfcast(ts_df, value="sales", method="bad", **KW)


def test_tsfcast_bad_horizon(ts_df):
    with pytest.raises(ValueError):
        dx.tsfcast(ts_df, value="sales", horizon=0, **KW)


def test_tsfcast_bad_valid(ts_df):
    with pytest.raises(ValueError):
        dx.tsfcast(ts_df, value="sales", valid=0, **KW)


def test_tsfcast_not_enough_history():
    df = pd.DataFrame({"x": np.arange(3.0)})
    with pytest.raises(ValueError):
        dx.tsfcast(df, value="x", valid=2, **KW)


def test_tsfcast_requires_value(ts_df):
    with pytest.raises(ValueError):
        dx.tsfcast(ts_df, **KW)


def test_tsfcast_value_not_found(ts_df):
    with pytest.raises(KeyError):
        dx.tsfcast(ts_df, value="missing", **KW)


def test_tsfcast_snaive_without_period_errors():
    df = pd.DataFrame({"x": np.cumsum(np.random.default_rng(2).normal(0, 1, 40))})
    with pytest.raises(ValueError):
        dx.tsfcast(df, value="x", method="snaive", **KW)


def test_tsfcast_missing_values_rejected(ts_df):
    df = ts_df.copy()
    arr = df["sales"].to_numpy().copy()
    arr[4] = np.nan
    df["sales"] = arr
    with pytest.raises(ValueError):
        dx.tsfcast(df, value="sales", **KW)
