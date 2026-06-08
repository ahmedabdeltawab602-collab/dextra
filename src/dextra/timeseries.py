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

from ._utils import _ensure_pandas, append_audit, get_variable_name, now_iso
from ._version import __version__
from .modeling import (
    _display,
    _finalize_figure,
    _fmt_metric,
    _fmt_table,
    _json_safe_num,
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
        "analyzed_at": now_iso(),
    }

    decision = (
        f"{n} obs, period={period}, {report['model']} ({method}): "
        f"trend strength={_fmt_metric(f_t, decimals)}, seasonal "
        f"strength={_fmt_metric(f_s, decimals)}, residual "
        f"std={_fmt_metric(rstd, decimals)} (mode={mode}).")

    out = df.copy()
    out.attrs = dict(df.attrs)
    append_audit(out, {
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
        "analyzed_at": now_iso(),
    }

    decision = (
        f"{n} obs: ADF p={_fmt_metric(adf_p, decimals)} "
        f"({'stationary' if adf_stationary else 'unit root'}), KPSS "
        f"p={_fmt_metric(kpss_p, decimals)} "
        f"({'stationary' if kpss_stationary else 'non-stationary'}) -> "
        f"{verdict}; suggested d={suggested_d} (mode={mode}).")

    out = df.copy()
    out.attrs = dict(df.attrs)
    append_audit(out, {
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


# ===========================================================================
# 8.3  tsfcast  --  validated baseline forecast (naive/snaive/drift/mean)
# ===========================================================================

_FC_METHODS = ("naive", "snaive", "drift", "mean")


def _baseline_forecast(train, steps, method, m):
    """Forecast ``steps`` ahead from ``train`` with one baseline. No look-ahead."""
    train = np.asarray(train, dtype=float)
    last = float(train[-1])
    if method == "naive":
        return np.full(steps, last)
    if method == "mean":
        return np.full(steps, float(train.mean()))
    if method == "drift":
        if len(train) < 2:
            return np.full(steps, last)
        slope = (train[-1] - train[0]) / (len(train) - 1)
        return last + slope * np.arange(1, steps + 1)
    if method == "snaive":
        if not m or m < 2 or len(train) < m:
            raise ValueError(
                "tsfcast: snaive needs a seasonal period (>=2) and at least "
                "one full season of history.")
        season = train[-m:]
        return season[np.arange(steps) % m]
    raise ValueError(f"tsfcast: unknown method {method!r}.")


def _mase_scale(train, m):
    """In-sample seasonal-naive mean absolute error (the MASE denominator)."""
    train = np.asarray(train, dtype=float)
    mm = m if (m and m >= 2 and len(train) > m) else 1
    diffs = np.abs(train[mm:] - train[:-mm])
    scale = float(diffs.mean()) if len(diffs) else 0.0
    if scale > 0:
        return scale
    d1 = np.abs(np.diff(train))
    alt = float(d1.mean()) if len(d1) else 0.0
    return alt if alt > 0 else 1.0


def _fc_metrics(actual, pred, scale):
    """MASE / RMSE / MAE / MAPE for a validation window."""
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = actual != 0
    mape = (float(np.mean(np.abs(err[nz] / actual[nz])) * 100.0)
            if nz.any() else float("nan"))
    mase = float(mae / scale) if scale > 0 else float("nan")
    return {"MASE": mase, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def _future_index(tidx, n, horizon):
    """Continue a datetime index by its frequency, else an integer range."""
    if tidx is not None and len(tidx) >= 3 and not tidx.isna().any():
        freq = tidx.freqstr or pd.infer_freq(tidx)
        if freq:
            return pd.date_range(start=tidx[-1], periods=horizon + 1,
                                 freq=freq)[1:]
    return pd.RangeIndex(n, n + horizon)


def tsfcast(
    df: pd.DataFrame,
    value=None,
    *,
    time=None,
    horizon: int = 12,
    valid: Optional[int] = None,
    period: Optional[int] = None,
    method: str = "auto",
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 13.0,
    fig_height: float = 5.6,
    dpi: int = 110,
):
    """Baseline forecast for a series, validated on a held-out tail, in one line.

    Trains a simple baseline on every observation *before* the last ``valid``
    points, scores it on that untouched tail (no look-ahead), then re-fits on the
    full series to project ``horizon`` steps ahead. Baselines: ``naive`` (last
    value), ``snaive`` (last season), ``drift`` (last value + average slope),
    ``mean``. ``method="auto"`` picks ``snaive`` when a seasonal period is
    available else ``naive``. ``method="compare"`` ranks every baseline on the
    validation window (by MASE) and writes no artifact.

    Two input modes: SERIES (``value`` (+ ``time``)) and ARTIFACT (``params``
    from a prior call -- settings are replayed on the new data). Dependency-free.

    Parameters
    ----------
    df : pandas.DataFrame
        The data holding the series. Never mutated.
    value : str or array-like, optional
        The observed series. Required in series mode.
    time : str or array-like, optional
        A datetime column / values for ordering, the x-axis and the future index.
    horizon : int
        Number of steps to forecast ahead.
    valid : int, optional
        Length of the held-out validation tail (defaults to ``horizon``).
    period : int, optional
        Seasonal period for ``snaive`` / MASE; inferred from ``time`` when omitted.
    method : {"auto", "naive", "snaive", "drift", "mean", "compare"}
        The baseline (or ``compare`` to rank them all).
    params, return_params, show, plot, return_df, return_fig, decimals, df_name :
        see the dextra standard flags.
    fig_width, fig_height, dpi : figure geometry.

    Returns
    -------
    pandas.DataFrame
        For a named method: a forward-forecast frame (``forecast`` / ``lower`` /
        ``upper``) indexed by the future periods. For ``compare``: the validation
        leaderboard (one row per baseline). Plus the descriptor / figure when
        requested.

    Examples
    --------
    >>> dx.tsfcast(df, value='sales', time='month', horizon=12)
    >>> dx.tsfcast(df, value='sales', method='compare')        # leaderboard
    >>> fc, p = dx.tsfcast(df, value='sales', method='drift', return_params=True)
    >>> dx.tsfcast(df_new, params=p)                           # artifact mode
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    mode = "artifact" if params is not None else "series"
    if mode == "artifact":
        if not isinstance(params, dict):
            raise ValueError(
                f"tsfcast: params must be a dextra descriptor dict, "
                f"got {type(params).__name__}.")
        value = params.get("value", value)
        time = params.get("time", time)
        method = params.get("method", method)
        meta = params.get("metadata", {})
        horizon = meta.get("horizon", horizon)
        valid = meta.get("valid", valid)
        period = meta.get("period", period)

    method = str(method).lower()
    allowed = ("auto", "compare") + _FC_METHODS
    if method not in allowed:
        raise ValueError(
            f"tsfcast: method must be one of {allowed}, got {method!r}.")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(f"tsfcast: horizon must be >= 1, got {horizon}.")
    if method == "compare" and return_params:
        raise ValueError(
            "tsfcast: method='compare' ranks baselines and writes no artifact; "
            "drop return_params or pick a single method.")

    vname, s = _resolve_series(df, value, "tsfcast")
    tname, tidx = _resolve_time(df, time, "tsfcast")

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
            "tsfcast: the series has missing values; clean it first "
            "(e.g. dx.handle_missing) before forecasting.")
    if period is None:
        period = _infer_period(tidx)
    period = int(period) if period else None

    valid = int(valid) if valid is not None else int(horizon)
    if valid < 1:
        raise ValueError(f"tsfcast: valid must be >= 1, got {valid}.")
    if n - valid < 2:
        raise ValueError(
            f"tsfcast: not enough history -- n={n}, valid={valid} leaves "
            f"{n - valid} training points (need >= 2). Lower valid.")

    resolved = method
    if method == "auto":
        resolved = "snaive" if (period and period >= 2 and
                                n - valid >= period) else "naive"

    train = y[:n - valid]
    actual = y[n - valid:]
    scale = _mase_scale(train, period)

    # ---- compare: rank every baseline on the validation tail, write nothing --
    if method == "compare":
        cands = ["naive", "drift", "mean"]
        if period and period >= 2 and len(train) >= period:
            cands.insert(1, "snaive")
        rows = {}
        val_preds = {}
        for cand in cands:
            vp = _baseline_forecast(train, valid, cand, period)
            val_preds[cand] = vp
            rows[cand] = _fc_metrics(actual, vp, scale)
        board = pd.DataFrame(rows).T[["MASE", "RMSE", "MAE", "MAPE"]]
        board = board.sort_values(
            by=["MASE", "RMSE"], ascending=True, na_position="last")
        best = str(board.index[0])
        decision = (
            f"{n} obs: compared {len(cands)} baselines on the last {valid} pts; "
            f"best by MASE = '{best}' (MASE="
            f"{_fmt_metric(board.loc[best, 'MASE'], decimals)}). Re-run "
            f"tsfcast(method='{best}') to forecast (mode={mode}).")
        out = df.copy()
        out.attrs = dict(df.attrs)
        append_audit(out, {
            "stage": "timeseries", "function": "tsfcast",
            "timestamp": now_iso(), "mode": mode,
            "params": {"value": vname, "time": tname, "compare": cands,
                       "valid": valid, "period": period},
            "decision": decision})
        if show:
            _print_header(f"Forecast comparison for: {df_name}  "
                          f"(value={vname}, mode={mode})")
            _display(_fmt_table(board, decimals))
            print(f"\nDecision: {decision}\n")
        fig = None
        if plot:
            fig = _plot_compare(result_index, y, actual, val_preds, board,
                                vname, valid, fig_width, fig_height, dpi)
        _finalize_figure(fig, return_fig)
        return _ret_pack(board, None, fig, return_df, False, return_fig)

    # ---- named method: validate on the tail, refit on full, forecast ahead ---
    val_pred = _baseline_forecast(train, valid, resolved, period)
    vmetrics = _fc_metrics(actual, val_pred, scale)
    fc = _baseline_forecast(y, horizon, resolved, period)
    band = 1.96 * vmetrics["RMSE"]
    fut_index = _future_index(tidx, n, horizon)
    forecast = pd.DataFrame(
        {"forecast": fc, "lower": fc - band, "upper": fc + band},
        index=fut_index)

    report = {
        "function": "tsfcast",
        "task": "timeseries",
        "value": vname,
        "time": tname,
        "method": resolved,
        "metrics": {"validation": {k: _json_safe_num(v)
                                   for k, v in vmetrics.items()}},
        "metadata": {"n": int(n), "input_mode": mode, "method_requested": method,
                     "horizon": int(horizon), "valid": int(valid),
                     "period": period, "freq": _freq_string(tidx)},
        "version": __version__,
        "analyzed_at": now_iso(),
    }
    decision = (
        f"{n} obs, method={resolved}"
        f"{' (auto)' if method == 'auto' else ''}: validation "
        f"MASE={_fmt_metric(vmetrics['MASE'], decimals)}, "
        f"RMSE={_fmt_metric(vmetrics['RMSE'], decimals)} on the last {valid} "
        f"pts; forecasting {horizon} steps ahead (mode={mode}).")

    out = df.copy()
    out.attrs = dict(df.attrs)
    append_audit(out, {
        "stage": "timeseries", "function": "tsfcast",
        "timestamp": report["analyzed_at"], "mode": mode,
        "params": {"value": vname, "time": tname, "method": resolved,
                   "horizon": int(horizon), "valid": int(valid),
                   "period": period},
        "decision": decision})

    if show:
        _print_header(f"Forecast for: {df_name}  "
                      f"(value={vname}, method={resolved}, mode={mode})")
        _display(_fmt_table(pd.DataFrame({"validation": vmetrics}), decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_forecast(result_index, y, actual, val_pred, forecast,
                             vname, resolved, valid, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(forecast, report, fig, return_df, return_params, return_fig)


def _plot_forecast(index, y, actual, val_pred, forecast, vname, method, valid,
                   fig_width, fig_height, dpi):
    """Two panels: full history + forward forecast, and the validation window."""
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Baseline forecast -- {vname} ({method})", fontsize=13,
                 fontweight="bold")
    axes[0].plot(index, y, color="#1f4e79", linewidth=1.2, label="observed")
    axes[0].plot(forecast.index, forecast["forecast"], color="#c0504d",
                 linewidth=1.8, linestyle="--", label="forecast")
    axes[0].fill_between(forecast.index, forecast["lower"], forecast["upper"],
                         color="#c0504d", alpha=0.18, label="~95% band")
    axes[0].set_title("History & forward forecast")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    vx = index[len(index) - valid:]
    axes[1].plot(vx, actual, color="#1f4e79", marker="o", markersize=3,
                 linewidth=1.2, label="actual")
    axes[1].plot(vx, val_pred, color="#c0504d", marker="x", markersize=4,
                 linewidth=1.4, linestyle="--", label="predicted")
    axes[1].set_title(f"Validation window (last {valid})")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    return fig


def _plot_compare(index, y, actual, val_preds, board, vname, valid, fig_width,
                  fig_height, dpi):
    """Two panels: validation actual vs each baseline, and a MASE bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Baseline comparison -- {vname}", fontsize=13,
                 fontweight="bold")
    vx = index[len(index) - valid:]
    axes[0].plot(vx, actual, color="black", marker="o", markersize=3,
                 linewidth=1.6, label="actual")
    for cand, vp in val_preds.items():
        axes[0].plot(vx, vp, linewidth=1.2, linestyle="--", label=cand)
    axes[0].set_title(f"Validation window (last {valid})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    names = list(board.index)
    mase = board["MASE"].to_numpy(dtype=float)
    axes[1].barh(names, mase, color="#4c72b0")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("MASE (lower is better)")
    axes[1].set_title("Leaderboard")
    axes[1].grid(True, axis="x", alpha=0.3)
    return fig
