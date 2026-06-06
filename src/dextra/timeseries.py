"""Time-series helpers for dextra - Phase 8 of the Roadmap.

Implements the time-series framework documented in TIMESERIES_PHILOSOPHY.md at
the project root. Phase 8 treats rows as ORDERED observations of one series
(order carries information; the future must never inform the past) and ships
three baseline diagnostics:

* :func:`tsdecomp`     - split a series into trend / seasonal / residual.
* ``tsstat``      - ADF / KPSS tests + suggested differencing (8.2).
* ``tsfcast``    - validated baseline forecast (8.3).

Every function in this module:

* Accepts a pandas DataFrame and NEVER mutates it. The result is returned as a
  fresh frame; the audit trail is written to a copy's attrs.
* Supports two input modes:
    - SERIES mode   : ``df`` + ``value`` (+ optional ``time``). A self-contained
      analysis of one observed series.
    - ARTIFACT mode : ``df`` + ``params`` (a descriptor from a previous Phase-8
      call). The resolved ``value`` / ``time`` / ``period`` / ``model`` /
      ``method`` are read back so the analysis is reproduced without re-deciding.
* Prints a compact summary table and a one-line ``Decision:`` sentence naming
  the headline result and the window / mode it was measured on.
* Renders a multi-panel diagnostic figure.
* Appends an entry to a copy's ``df.attrs['dextra_audit']``.
* Exposes a JSON-safe descriptor via ``return_params=True`` (no estimator).

``statsmodels`` is imported LAZILY, only when STL decomposition or the ADF /
KPSS tests need it (install the ``ts`` extra). Classical decomposition and every
forecast baseline are dependency-free (numpy / pandas / matplotlib only).

Stage 8.1 - tsdecomp ; 8.2 - tsstat ; 8.3 - tsfcast.
"""

from __future__ import annotations

import re
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._utils import _ensure_pandas, get_variable_name
from ._version import __version__
from .modeling import (
    _append_audit,
    _display,
    _finalize_figure,
    _fmt_metric,
    _fmt_table,
    _json_safe_num,
    _now_iso,
    _print_header,
    _ret_pack,
)

# Seasonal period for the common pandas frequency families. Keys are matched as
# a prefix of the (multiplier-stripped, upper-cased) frequency string so that
# 'MS' / 'ME' / 'M' all map to 12, etc.
_FREQ_PERIOD = {
    "A": 1, "Y": 1,        # yearly
    "Q": 4,                # quarterly
    "M": 12,               # monthly
    "W": 52,               # weekly
    "B": 5,                # business-daily
    "D": 7,                # daily
    "H": 24,               # hourly
}


# ---------------------------------------------------------------------------
# Optional-dependency gate (mirrors modeling._require_sklearn)
# ---------------------------------------------------------------------------

def _require_statsmodels(func_name: str):
    """Import statsmodels lazily, with a clear, actionable error if absent."""
    try:
        import statsmodels  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without sm
        raise ImportError(
            f"{func_name}: this feature needs statsmodels. Install the extra "
            f"with `pip install \"dextra[ts]\"` (or `pip install statsmodels`)."
        ) from exc
    import statsmodels
    return statsmodels


# ---------------------------------------------------------------------------
# Shared helpers (input resolution for the two modes)
# ---------------------------------------------------------------------------

def _resolve_series(df, value, func_name):
    """Resolve a column name or array-like to (name, numeric Series)."""
    if value is None:
        raise ValueError(
            f"{func_name}: value= is required in series mode "
            f"(or pass params= for artifact mode).")
    if isinstance(value, str):
        if value not in df.columns:
            raise KeyError(f"{func_name}: value={value!r} not found in df.")
        s = df[value]
        name = value
    else:
        s = value if isinstance(value, pd.Series) else pd.Series(list(value))
        if len(s) != len(df):
            raise ValueError(
                f"{func_name}: value has length {len(s)}, df has {len(df)} rows.")
        s = s.copy()
        s.index = df.index
        name = str(s.name) if s.name is not None else "value"
    s = pd.to_numeric(s, errors="coerce")
    return name, s


def _resolve_time(df, time, func_name):
    """Resolve the optional time axis to (name, DatetimeIndex-or-None)."""
    if time is None:
        if isinstance(df.index, pd.DatetimeIndex):
            return None, df.index
        return None, None
    if isinstance(time, str):
        if time not in df.columns:
            raise KeyError(f"{func_name}: time={time!r} not found in df.")
        idx = pd.DatetimeIndex(pd.to_datetime(df[time], errors="coerce"))
        return time, idx
    idx = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(time)), errors="coerce"))
    if len(idx) != len(df):
        raise ValueError(
            f"{func_name}: time has length {len(idx)}, df has {len(df)} rows.")
    return None, idx


def _freq_string(tidx):
    """Best-effort pandas frequency string for a DatetimeIndex (or None)."""
    if tidx is None or len(tidx) < 3 or tidx.isna().any():
        return None
    freq = tidx.freqstr
    if freq is None:
        try:
            freq = pd.infer_freq(tidx)
        except (ValueError, TypeError):
            freq = None
    return freq


def _infer_period(tidx):
    """Infer the seasonal period from a DatetimeIndex's frequency, or None."""
    freq = _freq_string(tidx)
    if not freq:
        return None
    base = re.sub(r"^[0-9]+", "", freq.upper()).lstrip("-")
    for key, period in _FREQ_PERIOD.items():
        if base.startswith(key):
            return period
    return None


# ---------------------------------------------------------------------------
# Classical decomposition (dependency-free)
# ---------------------------------------------------------------------------

def _centered_ma(y, m):
    """Centred moving-average trend of order m (2xm for even m). NaN at edges."""
    n = len(y)
    trend = np.full(n, np.nan)
    if m < 2 or n < m:
        return trend
    if m % 2 == 0:
        w = np.concatenate(([0.5], np.ones(m - 1), [0.5])) / m  # length m+1
    else:
        w = np.ones(m) / m
    half = len(w) // 2
    valid = np.convolve(y, w, mode="valid")  # symmetric weights -> no flip needed
    trend[half:half + len(valid)] = valid
    return trend


def _classical_decompose(y, m, model):
    """Return (trend, seasonal, resid) via classical decomposition."""
    n = len(y)
    trend = _centered_ma(y, m)
    if model.startswith("mult"):
        detr = y / trend
    else:
        detr = y - trend
    phase = np.arange(n) % m
    season_avg = np.array(
        [np.nanmean(detr[phase == k]) for k in range(m)], dtype=float)
    if model.startswith("mult"):
        season_avg = season_avg / np.nanmean(season_avg)
    else:
        season_avg = season_avg - np.nanmean(season_avg)
    seasonal = season_avg[phase]
    if model.startswith("mult"):
        resid = y / (trend * seasonal)
    else:
        resid = y - trend - seasonal
    return trend, seasonal, resid


def _strengths(trend, seasonal, resid, model):
    """Hyndman trend / seasonal strengths in [0, 1] (additive, or on logs)."""
    if model.startswith("mult"):
        t = np.log(trend)
        s = np.log(seasonal)
        r = np.log(resid)
    else:
        t, s, r = trend, seasonal, resid
    mask = np.isfinite(t) & np.isfinite(s) & np.isfinite(r)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    rv = np.var(r[mask])
    tv = np.var((t + r)[mask])
    sv = np.var((s + r)[mask])
    f_t = max(0.0, 1.0 - rv / tv) if tv > 0 else 0.0
    f_s = max(0.0, 1.0 - rv / sv) if sv > 0 else 0.0
    return float(f_t), float(f_s)


def _stl_decompose(y, m, result_index, func_name):
    """STL decomposition via a lazy statsmodels import."""
    _require_statsmodels(func_name)
    from statsmodels.tsa.seasonal import STL
    series = pd.Series(np.asarray(y, dtype=float), index=result_index)
    res = STL(series, period=m).fit()
    return (res.trend.to_numpy(), res.seasonal.to_numpy(),
            res.resid.to_numpy())


# ===========================================================================
# 8.1  tsdecomp  --  trend / seasonal / residual decomposition
# ===========================================================================

def tsdecomp(
    df: pd.DataFrame,
    value=None,
    *,
    time=None,
    period: Optional[int] = None,
    model: str = "additive",
    method: str = "classical",
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 13.0,
    fig_height: float = 9.0,
    dpi: int = 110,
):
    """Decompose a time series into trend / seasonal / residual in one line.

    Two input modes. In SERIES mode pass ``value`` (a column name or array-like)
    and, ideally, ``time`` (a datetime column used for the x-axis and to infer
    the seasonal ``period``). In ARTIFACT mode pass ``params`` (a descriptor from
    a previous ``tsdecomp`` call) and the data; the resolved value / time /
    period / model / method are read back from it. Returns a dense components
    frame (``observed`` / ``trend`` / ``seasonal`` / ``resid``), a four-panel
    figure, and a one-line decision naming the trend and seasonal strengths.

    Parameters
    ----------
    df : pandas.DataFrame
        The data holding the series. Never mutated.
    value : str or array-like, optional
        The observed series (column name or values). Required in series mode.
    time : str or array-like, optional
        A datetime column / values for the x-axis and period inference.
    period : int, optional
        The seasonal period (e.g. 12 for monthly). Inferred from ``time`` when
        omitted; an error is raised if it cannot be inferred.
    model : {"additive", "multiplicative"}
        The decomposition model. ``multiplicative`` requires positive values.
    method : {"classical", "stl"}
        ``classical`` is dependency-free; ``stl`` uses a lazy statsmodels import.
    params : dict, optional
        A Phase-8 descriptor for artifact mode.
    return_params : bool
        Also return the JSON-safe descriptor (no estimator).
    show, plot, return_df, return_fig, decimals, df_name : see the dextra
        standard flags.
    fig_width, fig_height, dpi : figure geometry.

    Returns
    -------
    pandas.DataFrame
        The components frame, and -- when requested -- the descriptor and / or
        the matplotlib figure.

    Examples
    --------
    >>> dx.tsdecomp(df, value='sales', time='month', period=12)
    >>> res, p = dx.tsdecomp(df, value='sales', time='month',
    ...                          return_params=True)
    >>> dx.tsdecomp(df_new, params=p)              # artifact mode
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    mode = "artifact" if params is not None else "series"
    if mode == "artifact":
        if not isinstance(params, dict):
            raise ValueError(
                f"tsdecomp: params must be a dextra descriptor dict, "
                f"got {type(params).__name__}.")
        value = params.get("value", value)
        time = params.get("time", time)
        period = params.get("period", period)
        model = params.get("model", model)
        method = params.get("method", method)

    model = str(model).lower()
    if not (model.startswith("add") or model.startswith("mult")):
        raise ValueError(
            f"tsdecomp: model must be 'additive' or 'multiplicative', "
            f"got {model!r}.")
    method = str(method).lower()
    if method not in ("classical", "stl"):
        raise ValueError(
            f"tsdecomp: method must be 'classical' or 'stl', got {method!r}.")

    vname, s = _resolve_series(df, value, "tsdecomp")
    tname, tidx = _resolve_time(df, time, "tsdecomp")

    y = s.to_numpy(dtype=float)
    n = len(y)
    # Order by time when a clean, non-monotonic datetime axis is supplied.
    if tidx is not None and not tidx.isna().any():
        if not tidx.is_monotonic_increasing:
            order = np.argsort(tidx.values, kind="stable")
            y = y[order]
            tidx = tidx[order]
    result_index = tidx if tidx is not None else pd.RangeIndex(n)

    if np.isnan(y).any():
        raise ValueError(
            "tsdecomp: the series has missing values; clean it first "
            "(e.g. dx.handle_missing) before decomposing.")

    if period is None:
        period = _infer_period(tidx)
    if period is None:
        raise ValueError(
            "tsdecomp: could not infer a seasonal period from the time "
            "axis; pass period= explicitly (e.g. period=12 for monthly data).")
    period = int(period)
    if period < 2:
        raise ValueError(f"tsdecomp: period must be >= 2, got {period}.")
    if n < 2 * period:
        raise ValueError(
            f"tsdecomp: need at least 2 full periods ({2 * period} rows), "
            f"got {n}.")

    if model.startswith("mult") and np.any(y <= 0):
        raise ValueError(
            "tsdecomp: multiplicative model requires strictly positive "
            "values; use model='additive' or remove non-positive points.")

    if method == "stl":
        trend, seasonal, resid = _stl_decompose(y, period, result_index,
                                                 "tsdecomp")
    else:
        trend, seasonal, resid = _classical_decompose(y, period, model)

    f_t, f_s = _strengths(trend, seasonal, resid, model)
    rstd = float(np.nanstd(resid))
    freq = _freq_string(tidx)

    components = pd.DataFrame(
        {"observed": y, "trend": trend, "seasonal": seasonal, "resid": resid},
        index=result_index,
    )

    report = {
        "function": "tsdecomp",
        "task": "timeseries",
        "value": vname,
        "time": tname,
        "period": period,
        "model": "multiplicative" if model.startswith("mult") else "additive",
        "method": method,
        "metrics": {
            "strength_trend": _json_safe_num(f_t),
            "strength_seasonal": _json_safe_num(f_s),
            "resid_std": _json_safe_num(rstd),
        },
        "metadata": {"n": int(n), "input_mode": mode, "freq": freq},
        "version": __version__,
        "analyzed_at": _now_iso(),
    }

    decision = (
        f"{n} obs, period={period}, {report['model']} ({method}): "
        f"trend strength={_fmt_metric(f_t, decimals)}, seasonal "
        f"strength={_fmt_metric(f_s, decimals)}, residual "
        f"std={_fmt_metric(rstd, decimals)} (mode={mode}).")

    out = df.copy()
    out.attrs = dict(df.attrs)
    _append_audit(out, {
        "stage": "timeseries",
        "function": "tsdecomp",
        "timestamp": report["analyzed_at"],
        "mode": mode,
        "params": {"value": vname, "time": tname, "period": period,
                   "model": report["model"], "method": method},
        "decision": decision,
    })

    if show:
        summary = pd.DataFrame(
            {"value": [period, report["model"], method, f_t, f_s, rstd, n]},
            index=["period", "model", "method", "strength_trend",
                   "strength_seasonal", "resid_std", "n_obs"],
        )
        _print_header(f"Decomposition for: {df_name}  "
                      f"(value={vname}, mode={mode})")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_decompose(components, vname, report["model"], period,
                              fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(components, report, fig, return_df, return_params,
                     return_fig)


def _plot_decompose(components, vname, model, period, fig_width, fig_height,
                    dpi):
    """Four stacked panels: observed, trend, seasonal, residual."""
    x = components.index
    fig, axes = plt.subplots(4, 1, figsize=(fig_width, fig_height), dpi=dpi,
                             sharex=True)
    fig.suptitle(f"Time-series decomposition -- {vname} "
                 f"({model}, period={period})", fontsize=13, fontweight="bold")
    panels = [
        ("observed", "#1f4e79", "Observed"),
        ("trend", "#c0504d", "Trend"),
        ("seasonal", "#4c8c2b", "Seasonal"),
        ("resid", "#7f7f7f", "Residual"),
    ]
    for ax, (col, color, label) in zip(axes, panels):
        if col == "resid":
            base = 1.0 if model.startswith("mult") else 0.0
            ax.axhline(base, color="black", linewidth=0.8, alpha=0.5)
            ax.scatter(x, components[col], s=10, color=color)
        else:
            ax.plot(x, components[col], color=color, linewidth=1.4)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time")
    return fig


# ===========================================================================
# 8.2  tsstat  --  stationarity tests (ADF / KPSS) + suggested differencing
# ===========================================================================

def _acf(y, nlags):
    """Sample autocorrelation for lags 0..nlags (dependency-free)."""
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    n = len(y)
    denom = float(np.dot(y, y))
    if denom == 0.0:
        out = np.zeros(nlags + 1)
        out[0] = 1.0
        return out
    acf = [1.0]
    for k in range(1, nlags + 1):
        acf.append(float(np.dot(y[:n - k], y[k:]) / denom))
    return np.array(acf)


def _adf_kpss(y, regression):
    """Run ADF and KPSS via a lazy statsmodels import.

    Returns ``(adf_stat, adf_p, kpss_stat, kpss_p, usedlag, nobs)``. ADF's null
    is a unit root (non-stationary); KPSS's null is stationarity -- the two are
    complementary, which is exactly why dextra reports both.
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    arr = np.asarray(y, dtype=float)
    adf = adfuller(arr, regression=regression, autolag="AIC")
    adf_stat, adf_p, usedlag, nobs = adf[0], adf[1], adf[2], adf[3]
    with warnings.catch_warnings():
        # KPSS warns when the statistic falls outside its p-value lookup table;
        # the clamped p-value (0.01 / 0.10) is still the correct decision.
        warnings.simplefilter("ignore")
        kp = kpss(arr, regression=regression, nlags="auto")
    kpss_stat, kpss_p = kp[0], kp[1]
    return (float(adf_stat), float(adf_p), float(kpss_stat), float(kpss_p),
            int(usedlag), int(nobs))


def _verdict(adf_stationary, kpss_stationary):
    """The classic four-case ADF x KPSS interpretation."""
    if adf_stationary and kpss_stationary:
        return "stationary"
    if (not adf_stationary) and (not kpss_stationary):
        return "non-stationary (unit root) -> difference"
    if kpss_stationary and not adf_stationary:
        return "trend-stationary -> detrend"
    return "difference-stationary -> difference"


def _suggest_d(y, alpha, regression, max_diff):
    """Difference until ADF rejects a unit root AND KPSS fails to reject
    stationarity (capped at ``max_diff``). Returns ``(d, path)``."""
    series = np.asarray(y, dtype=float)
    path = []
    last_d = 0
    for d in range(max_diff + 1):
        if len(series) < 8:        # too few points to test reliably
            break
        _, adf_p, _, kpss_p, _, _ = _adf_kpss(series, regression)
        ok = (adf_p < alpha) and (kpss_p >= alpha)
        path.append({"d": d, "adf_p": _json_safe_num(adf_p),
                     "kpss_p": _json_safe_num(kpss_p), "stationary": bool(ok)})
        last_d = d
        if ok:
            return d, path
        series = np.diff(series)
    return last_d, path


def tsstat(
    df: pd.DataFrame,
    value=None,
    *,
    time=None,
    max_diff: int = 2,
    alpha: float = 0.05,
    regression: str = "c",
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.8,
    dpi: int = 110,
):
    """Test a series for stationarity (ADF + KPSS) and suggest differencing.

    Two input modes. In SERIES mode pass ``value`` (a column name or array-like)
    and, optionally, ``time``. In ARTIFACT mode pass ``params`` (a descriptor
    from a previous ``tsstat`` call) and the data; ``value`` / ``time`` /
    ``alpha`` / ``regression`` / ``max_diff`` are read back from it. Runs the
    Augmented Dickey-Fuller test (null: unit root) and the KPSS test (null:
    stationarity) -- complementary nulls -- reports both, gives the classic
    four-case verdict, and suggests a differencing order ``d`` by differencing
    until ADF rejects a unit root AND KPSS fails to reject stationarity (capped
    at ``max_diff``). Returns a two-row test table, a three-panel figure
    (series + rolling mean, rolling std, ACF) and a one-line decision.

    Requires statsmodels (the optional ``ts`` extra), imported lazily.

    Parameters
    ----------
    df : pandas.DataFrame
        The data holding the series. Never mutated.
    value : str or array-like, optional
        The observed series. Required in series mode.
    time : str or array-like, optional
        A datetime column / values for ordering and the x-axis.
    max_diff : int
        The cap on the suggested differencing order.
    alpha : float
        Significance level for both tests and the verdict.
    regression : {"c", "ct"}
        Deterministic term: ``c`` (constant) or ``ct`` (constant + trend).
    params : dict, optional
        A Phase-8 descriptor for artifact mode.
    return_params, show, plot, return_df, return_fig, decimals, df_name : see the
        dextra standard flags.
    fig_width, fig_height, dpi : figure geometry.

    Returns
    -------
    pandas.DataFrame
        The ADF / KPSS test table, and -- when requested -- the descriptor and /
        or the matplotlib figure.

    Examples
    --------
    >>> dx.tsstat(df, value='sales', time='month')
    >>> tbl, p = dx.tsstat(df, value='sales', return_params=True)
    >>> dx.tsstat(df_new, params=p)                    # artifact mode
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    mode = "artifact" if params is not None else "series"
    if mode == "artifact":
        if not isinstance(params, dict):
            raise ValueError(
                f"tsstat: params must be a dextra descriptor dict, "
                f"got {type(params).__name__}.")
        value = params.get("value", value)
        time = params.get("time", time)
        meta = params.get("metadata", {})
        alpha = meta.get("alpha", alpha)
        regression = meta.get("regression", regression)
        max_diff = meta.get("max_diff", max_diff)

    regression = str(regression).lower()
    if regression not in ("c", "ct"):
        raise ValueError(
            f"tsstat: regression must be 'c' or 'ct', got {regression!r}.")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"tsstat: alpha must be in (0, 1), got {alpha}.")
    max_diff = int(max_diff)
    if max_diff < 0:
        raise ValueError(f"tsstat: max_diff must be >= 0, got {max_diff}.")

    _require_statsmodels("tsstat")

    vname, s = _resolve_series(df, value, "tsstat")
    tname, tidx = _resolve_time(df, time, "tsstat")

    y = s.to_numpy(dtype=float)
    n = len(y)
    if tidx is not None and not tidx.isna().any():
        if not tidx.is_monotonic_increasing:
            order = np.argsort(tidx.values, kind="stable")
            y = y[order]
            tidx = tidx[order]
    result_index = tidx if tidx is not None else pd.RangeIndex(n)

    if np.isnan(y).any():
        raise ValueError(
            "tsstat: the series has missing values; clean it first "
            "(e.g. dx.handle_missing) before testing stationarity.")
    if n < 12:
        raise ValueError(
            f"tsstat: need at least 12 observations to test, got {n}.")
    if float(np.nanstd(y)) == 0.0:
        raise ValueError("tsstat: the series is constant; stationarity is undefined.")

    adf_stat, adf_p, kpss_stat, kpss_p, usedlag, nobs = _adf_kpss(y, regression)
    adf_stationary = adf_p < alpha
    kpss_stationary = kpss_p >= alpha
    verdict = _verdict(adf_stationary, kpss_stationary)
    suggested_d, path = _suggest_d(y, alpha, regression, max_diff)

    table = pd.DataFrame(
        {
            "statistic": [adf_stat, kpss_stat],
            "p_value": [adf_p, kpss_p],
            "null_hypothesis": ["unit root (non-stationary)", "stationary"],
            "reject_null": [bool(adf_p < alpha), bool(kpss_p < alpha)],
            "implies_stationary": [bool(adf_stationary), bool(kpss_stationary)],
        },
        index=["ADF", "KPSS"],
    )

    report = {
        "function": "tsstat",
        "task": "timeseries",
        "value": vname,
        "time": tname,
        "metrics": {
            "adf": {"stat": _json_safe_num(adf_stat),
                    "pvalue": _json_safe_num(adf_p),
                    "usedlag": usedlag, "nobs": nobs},
            "kpss": {"stat": _json_safe_num(kpss_stat),
                     "pvalue": _json_safe_num(kpss_p)},
            "verdict": verdict,
            "suggested_d": int(suggested_d),
            "differencing_path": path,
        },
        "metadata": {"n": int(n), "input_mode": mode, "alpha": float(alpha),
                     "regression": regression, "max_diff": int(max_diff),
                     "freq": _freq_string(tidx)},
        "version": __version__,
        "analyzed_at": _now_iso(),
    }

    decision = (
        f"{n} obs: ADF p={_fmt_metric(adf_p, decimals)} "
        f"({'stationary' if adf_stationary else 'unit root'}), KPSS "
        f"p={_fmt_metric(kpss_p, decimals)} "
        f"({'stationary' if kpss_stationary else 'non-stationary'}) -> "
        f"{verdict}; suggested d={suggested_d} (mode={mode}).")

    out = df.copy()
    out.attrs = dict(df.attrs)
    _append_audit(out, {
        "stage": "timeseries",
        "function": "tsstat",
        "timestamp": report["analyzed_at"],
        "mode": mode,
        "params": {"value": vname, "time": tname, "alpha": float(alpha),
                   "regression": regression, "max_diff": int(max_diff)},
        "decision": decision,
    })

    if show:
        _print_header(f"Stationarity for: {df_name}  "
                      f"(value={vname}, mode={mode})")
        _display(_fmt_table(table, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_stationarity(result_index, y, vname, n,
                                 _infer_period(tidx), fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(table, report, fig, return_df, return_params, return_fig)


def _plot_stationarity(index, y, vname, n, period, fig_width, fig_height, dpi):
    """Three panels: series + rolling mean, rolling std, and the ACF."""
    w = int(period) if period and period >= 2 else max(2, min(12, n // 4))
    w = max(2, min(w, n))
    s = pd.Series(y, index=index)
    roll_mean = s.rolling(window=w, center=True).mean()
    roll_std = s.rolling(window=w, center=True).std()
    nlags = int(min(40, max(10, n // 2)))
    nlags = min(nlags, n - 1)
    acf = _acf(y, nlags)
    conf = 1.96 / np.sqrt(n)

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Stationarity diagnostics -- {vname}", fontsize=13,
                 fontweight="bold")
    axes[0].plot(index, y, color="#1f4e79", linewidth=1.2, label="observed")
    axes[0].plot(index, roll_mean, color="#c0504d", linewidth=1.6,
                 label=f"rolling mean (w={w})")
    axes[0].set_title("Series & rolling mean")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(index, roll_std, color="#4c8c2b", linewidth=1.6)
    axes[1].set_title(f"Rolling std (w={w})")
    axes[1].grid(True, alpha=0.3)
    lags = np.arange(len(acf))
    axes[2].vlines(lags, 0.0, acf, color="#1f4e79")
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].axhline(conf, color="red", linestyle="--", linewidth=0.8)
    axes[2].axhline(-conf, color="red", linestyle="--", linewidth=0.8)
    axes[2].set_title("Autocorrelation (ACF)")
    axes[2].set_xlabel("lag")
    axes[2].grid(True, alpha=0.3)
    return fig
