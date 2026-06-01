"""Feature-engineering helpers for dextra - Phase 4 of the Roadmap.

Implements the fit / transform framework documented in FEATURES_PHILOSOPHY.md
at the project root. Every function in this module:

* Accepts a pandas DataFrame and returns a NEW DataFrame (immutable; the
  original is never mutated).
* Supports two modes:
    - FIT mode  : learns parameters from the data (the training set).
    - APPLY mode: re-uses a saved ``params`` dict verbatim (test / production)
      WITHOUT re-fitting -- the technical safeguard against data leakage.
* Exposes a JSON-serialisable ``params`` dict via ``return_params=True``.
* Prints a before/after summary table.
* Renders a multi-panel visual showing the change.
* Prints a one-line ``Decision:`` sentence.
* Appends an entry to ``df.attrs['dextra_audit']``.
* Is idempotent under apply mode when ``inplace=False`` (the default): the
  source column is preserved, so re-applying reproduces the same new column.

Stage 4.1 - Numerical transforms + Scaling:
  - transform(df, method=...)  log / log1p / sqrt / boxcox / yeojohnson / compare
  - scale(df, method=...)      standard / minmax / robust / maxabs / compare

Stage 4.2 - Binning + Categorical encoding:
  - bin(df, method=...)        equal_width / quantile / kmeans / compare
  - encode(df, method=...)     onehot / ordinal / target / frequency / compare

Stage 4.3 - Temporal + Interactions + Aggregations:
  - dtfeats(df, method=...)    calendar / cyclical / both / compare
  - cross(df, method=...)      ratio / product / diff / polynomial / compare
  - aggfeat(df, agg=...)       groupby static / as_of expanding window

Stage 4.4 - Pipeline wrapper:
  - featpipe(df, steps=...)    chain the seven functions; fit -> combined
                               params -> apply / save / load (feature store)
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import _ensure_pandas, get_variable_name
from ._version import __version__

try:
    from IPython.display import display as _ipy_display
except ImportError:  # pragma: no cover
    _ipy_display = None


sns.set_style("whitegrid")

AUDIT_KEY = "dextra_audit"

_VALID_TRANSFORM_METHODS = ("log", "log1p", "sqrt", "boxcox", "yeojohnson",
                            "compare")
_TRANSFORM_CANDIDATES = ("log", "log1p", "sqrt", "boxcox", "yeojohnson")
_VALID_SCALE_METHODS = ("standard", "minmax", "robust", "maxabs", "compare")
_SCALE_CANDIDATES = ("standard", "minmax", "robust", "maxabs")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_scipy_stats(method: str):
    """Lazily import scipy.stats; only boxcox / yeojohnson need it.

    Keeping the import lazy means log / log1p / sqrt and the whole scale()
    family work even when SciPy is not installed.
    """
    try:
        from scipy import stats as _sst
        return _sst
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"transform method '{method}' requires SciPy, which is not "
            f"installed. Install it with `pip install scipy`, or use "
            f"method='log' / 'log1p' / 'sqrt' which need only NumPy."
        ) from exc


def _display(frame: pd.DataFrame) -> None:
    if _ipy_display is not None:
        _ipy_display(frame)
    else:
        print(frame.to_string())


def _print_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finalize_figure(fig, return_fig: bool) -> None:
    """Display the figure when one was created.

    ``show`` and ``plot`` are INDEPENDENT in dextra. A figure is created only
    when ``plot=True``; this helper displays it unless the caller asked for
    the figure object back via ``return_fig``.
    """
    if fig is None:
        return
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if not return_fig:
        plt.show()


def _ret_pack(out, params, fig, return_df, return_params, return_fig):
    """Pack outputs in the fixed order: dataframe, params, figure.

    Only the requested pieces are returned. A single requested piece is
    returned bare; multiple pieces are returned as a tuple.
    """
    results = []
    if return_df:
        results.append(out)
    if return_params:
        results.append(params)
    if return_fig:
        results.append(fig)
    if not results:
        return out
    if len(results) == 1:
        return results[0]
    return tuple(results)


def _append_audit(out: pd.DataFrame, entry: dict) -> None:
    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
    out.attrs[AUDIT_KEY].append(entry)


def _fmt_table(frame: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Format every cell of a summary table for readable display."""
    def _fmt(v):
        if pd.isna(v):
            return "-"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        try:
            return f"{float(v):,.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)
    return frame.map(_fmt)


def _auto_numeric_cols(df: pd.DataFrame) -> list:
    """Return numeric, non-boolean column names."""
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _resolve_cols(df: pd.DataFrame, cols, func_name: str) -> list:
    """Validate an explicit cols selector or auto-pick numeric columns."""
    if cols is None:
        chosen = _auto_numeric_cols(df)
        if not chosen:
            raise ValueError(
                f"{func_name}: no numeric columns found. Pass cols= explicitly.")
        return chosen
    chosen = list(cols)
    bad = [c for c in chosen if c not in df.columns]
    if bad:
        raise KeyError(f"{func_name}: cols references columns not in df: {bad}")
    non_num = [c for c in chosen if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        raise TypeError(
            f"{func_name} requires numeric columns; non-numeric passed: {non_num}")
    return chosen


# ===========================================================================
# 1. transform  --  numerical transforms (reshape distributions)
# ===========================================================================

def _check_transform_domain(df: pd.DataFrame, cols: Sequence[str],
                             method: str) -> None:
    """Raise a clear error when a method's mathematical domain is violated.

    This refuses the 'log(x) on zeros / negatives -> silent NaN' anti-pattern
    listed in FEATURES_PHILOSOPHY.md section 5.
    """
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        if v.empty:
            raise ValueError(
                f"transform: column '{c}' has no numeric values to transform.")
        if method in ("log", "boxcox") and (v <= 0).any():
            n_bad = int((v <= 0).sum())
            raise ValueError(
                f"transform: column '{c}' has {n_bad} value(s) <= 0; "
                f"'{method}' requires strictly positive input. Use "
                f"method='log1p' or method='yeojohnson' (handles any value).")
        if method == "sqrt" and (v < 0).any():
            n_bad = int((v < 0).sum())
            raise ValueError(
                f"transform: column '{c}' has {n_bad} negative value(s); "
                f"'sqrt' requires non-negative input. Use method='yeojohnson'.")
        if method == "log1p" and (v <= -1).any():
            n_bad = int((v <= -1).sum())
            raise ValueError(
                f"transform: column '{c}' has {n_bad} value(s) <= -1; "
                f"'log1p' requires input > -1. Use method='yeojohnson'.")


def _transform_one(s: pd.Series, method: str, lmbda=None):
    """Apply one transform to a Series, preserving index and NaN positions.

    Returns
    -------
    (transformed_series, fitted_lambda_or_None)
    """
    num = pd.to_numeric(s, errors="coerce")
    mask = num.notna()
    vals = num[mask].astype(float).to_numpy()
    fitted = None
    if method == "log":
        tvals = np.log(vals)
    elif method == "log1p":
        tvals = np.log1p(vals)
    elif method == "sqrt":
        tvals = np.sqrt(vals)
    elif method == "boxcox":
        _sst = _get_scipy_stats("boxcox")
        if lmbda is None:
            tvals, fitted = _sst.boxcox(vals)
            fitted = float(fitted)
        else:
            tvals = _sst.boxcox(vals, lmbda=lmbda)
            fitted = float(lmbda)
    elif method == "yeojohnson":
        _sst = _get_scipy_stats("yeojohnson")
        if lmbda is None:
            tvals, fitted = _sst.yeojohnson(vals)
            fitted = float(fitted)
        else:
            tvals = _sst.yeojohnson(vals, lmbda=lmbda)
            fitted = float(lmbda)
    else:
        raise ValueError(f"Unknown transform method {method!r}")
    result = pd.Series(np.nan, index=s.index, dtype="float64")
    result.loc[mask] = np.asarray(tvals, dtype="float64")
    return result, fitted


def transform(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "log",
    *,
    inplace: bool = False,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.2,
    dpi: int = 110,
):
    """Reshape numeric distributions toward symmetry / normality.

    Two modes. In FIT mode (``params=None``) the transform is learned from
    ``df`` and a reproducible ``params`` dict is produced. In APPLY mode
    (``params`` supplied) the saved transform is applied verbatim with no
    re-fitting -- the safeguard against statistic leakage.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    cols : sequence of str, optional
        Numeric columns to transform. If ``None`` all numeric (non-boolean)
        columns are used. Ignored in apply mode (columns come from params).
    method : {'log', 'log1p', 'sqrt', 'boxcox', 'yeojohnson', 'compare'}
        ``'compare'`` writes nothing -- it reports the resulting skewness of
        every candidate method so you can choose.
    inplace : bool, default False
        If False a new column ``<col>_<method>`` is added and the source kept.
        If True the source column is overwritten.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the learned params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        And, when requested, the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.transform(df_train, cols=['price'], method='boxcox',
    ...                         return_params=True)
    >>> df_te = dx.transform(df_test, params=p)        # apply, no re-fit
    >>> dx.transform(df, cols=['price'], method='compare')   # explore options
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _transform_apply(df, params, inplace, show, plot, return_df,
                                return_params, return_fig, decimals, df_name,
                                fig_width, fig_height, dpi)

    # ---- FIT MODE -------------------------------------------------------
    if method not in _VALID_TRANSFORM_METHODS:
        raise ValueError(
            f"'method' must be one of {_VALID_TRANSFORM_METHODS}, got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "method='compare' does not fit parameters; call transform with a "
            "concrete method (log/sqrt/boxcox/...) to obtain a params dict.")

    cols = _resolve_cols(df, cols, "transform")

    if method == "compare":
        return _transform_compare(df, cols, show, plot, return_df,
                                  return_params, return_fig, decimals,
                                  df_name, fig_width, fig_height, dpi)

    _check_transform_domain(df, cols, method)

    out = df.copy()
    out.attrs = dict(df.attrs)
    col_params, rows, col_map = {}, [], {}
    for c in cols:
        before = pd.to_numeric(df[c], errors="coerce")
        try:
            transformed, fitted = _transform_one(df[c], method)
        except ImportError:
            raise
        except Exception as exc:  # pragma: no cover - informative re-raise
            raise ValueError(
                f"transform: '{method}' failed on column '{c}': {exc}") from exc
        new_col = c if inplace else f"{c}_{method}"
        out[new_col] = transformed
        col_map[c] = new_col
        n_inf = int(np.isinf(transformed).sum())
        entry = {"source": c, "new_col": new_col}
        if fitted is not None:
            entry["lambda"] = fitted
        col_params[c] = entry
        rows.append({
            "new_col": new_col,
            "skew_before": float(before.skew()),
            "skew_after": float(transformed.skew()),
            "mean_after": float(transformed.mean()),
            "n_inf_new": n_inf,
        })
        if n_inf:
            warnings.warn(f"transform: column '{new_col}' has {n_inf} infinite "
                          f"value(s) after '{method}'.")

    params_out = {
        "function": "transform",
        "method": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "columns": col_params,
        "metadata": {"inplace": bool(inplace), "n_cols": len(cols)},
    }

    improved = sum(1 for r in rows
                   if abs(r["skew_after"]) < abs(r["skew_before"]))
    placement = ("overwrote source column(s)" if inplace
                 else f"added new *_{method} column(s)")
    decision = (f"Fitted '{method}' transform on {len(cols)} column(s); "
                f"|skew| reduced on {improved}/{len(cols)}; {placement}. "
                f"Created distribution feature(s).")

    _append_audit(out, {
        "stage": "feature_transform",
        "function": "transform",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "cols": list(cols),
                   "inplace": bool(inplace)},
        "decision": decision,
    })

    summary = pd.DataFrame(rows, index=list(cols))
    summary.index.name = "source"
    if show:
        _print_header(f"Numeric transform for: {df_name}  "
                      f"(method={method}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_transform(df, out, col_map, method,
                              fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _transform_apply(df, params, inplace, show, plot, return_df,
                     return_params, return_fig, decimals, df_name,
                     fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "transform":
        got = params.get("function") if isinstance(params, dict) else params
        raise ValueError(f"params dict is not for 'transform' (function={got!r}).")
    method = params["method"]
    col_params = params["columns"]
    # Decision 3(a): explicit rejection on column mismatch.
    missing = [src for src in col_params if src not in df.columns]
    if missing:
        raise KeyError(
            f"transform apply failed: params expects column(s) {missing} which "
            f"are not present in this DataFrame. The data does not match the "
            f"fitted transformer.")
    _check_transform_domain(df, list(col_params), method)

    out = df.copy()
    out.attrs = dict(df.attrs)
    rows, col_map = [], {}
    for src, cp in col_params.items():
        lmbda = cp.get("lambda")
        transformed, _ = _transform_one(df[src], method, lmbda=lmbda)
        new_col = src if inplace else cp.get("new_col", f"{src}_{method}")
        out[new_col] = transformed
        col_map[src] = new_col
        before = pd.to_numeric(df[src], errors="coerce")
        n_inf = int(np.isinf(transformed).sum())
        rows.append({
            "new_col": new_col,
            "skew_before": float(before.skew()),
            "skew_after": float(transformed.skew()),
            "mean_after": float(transformed.mean()),
            "n_inf_new": n_inf,
        })
        if n_inf:
            warnings.warn(f"transform: column '{new_col}' has {n_inf} infinite "
                          f"value(s) after '{method}'.")

    decision = (f"Applied saved '{method}' transform (fitted "
                f"{params.get('fit_at', '?')}) to {len(col_params)} "
                f"column(s); no re-fit -- leakage-safe.")
    _append_audit(out, {
        "stage": "feature_transform",
        "function": "transform",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"method": method, "cols": list(col_params),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    summary = pd.DataFrame(rows, index=list(col_params))
    summary.index.name = "source"
    if show:
        _print_header(f"Numeric transform for: {df_name}  "
                      f"(method={method}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_transform(df, out, col_map, method,
                              fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _transform_compare(df, cols, show, plot, return_df, return_params,
                       return_fig, decimals, df_name,
                       fig_width, fig_height, dpi):
    rows = []
    for c in cols:
        num = pd.to_numeric(df[c], errors="coerce")
        v = num.dropna()
        rec = {"skew_raw": float(num.skew())}
        for m in _TRANSFORM_CANDIDATES:
            ok = True
            if m in ("log", "boxcox") and (v <= 0).any():
                ok = False
            elif m == "sqrt" and (v < 0).any():
                ok = False
            elif m == "log1p" and (v <= -1).any():
                ok = False
            if not ok:
                rec[m] = np.nan
                continue
            try:
                t, _ = _transform_one(df[c], m)
                rec[m] = float(t.skew())
            except Exception:
                rec[m] = np.nan
        rows.append(rec)
    summary = pd.DataFrame(rows, index=list(cols))
    summary.index.name = "column"

    decision = (f"Compared {len(_TRANSFORM_CANDIDATES)} transform(s) on "
                f"{len(cols)} column(s). Table shows resulting skewness "
                f"(closer to 0 = more symmetric; '-' = method invalid for "
                f"that column's domain). No columns written -- pick a method "
                f"then call transform(method=...).")
    if show:
        _print_header(f"Transform COMPARE for: {df_name}  (nothing written)")
        _display(_fmt_table(summary, decimals))
        for c in summary.index:
            r = summary.loc[c, list(_TRANSFORM_CANDIDATES)].astype(float)
            if r.notna().any():
                best = r.abs().idxmin()
                print(f"  '{c}': smallest |skew| -> '{best}' "
                      f"({r[best]:.{decimals}f})")
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_transform_compare(df, cols, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# 2. scale  --  place numeric columns on a common scale
# ===========================================================================

def _scale_fit_one(vals: np.ndarray, method: str) -> dict:
    """Learn scaler parameters from a 1-D array of non-NaN floats."""
    if method == "standard":
        return {"mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=0))}
    if method == "minmax":
        return {"min": float(np.min(vals)), "max": float(np.max(vals))}
    if method == "robust":
        q1, q3 = (float(x) for x in np.percentile(vals, [25, 75]))
        return {"median": float(np.median(vals)), "iqr": float(q3 - q1)}
    if method == "maxabs":
        return {"max_abs": float(np.max(np.abs(vals)))}
    raise ValueError(f"Unknown scale method {method!r}")


def _scale_apply_one(num: pd.Series, method: str, cp: dict) -> pd.Series:
    """Apply scaler parameters to a numeric Series (NaN preserved)."""
    if method == "standard":
        denom = cp["std"] if cp["std"] != 0 else 1.0
        return (num - cp["mean"]) / denom
    if method == "minmax":
        rng = cp["max"] - cp["min"]
        denom = rng if rng != 0 else 1.0
        return (num - cp["min"]) / denom
    if method == "robust":
        denom = cp["iqr"] if cp["iqr"] != 0 else 1.0
        return (num - cp["median"]) / denom
    if method == "maxabs":
        denom = cp["max_abs"] if cp["max_abs"] != 0 else 1.0
        return num / denom
    raise ValueError(f"Unknown scale method {method!r}")


def _scale_warn_degenerate(c: str, method: str, cp: dict) -> None:
    if method == "standard" and cp["std"] == 0:
        warnings.warn(f"scale: column '{c}' has zero variance; "
                      f"standard output is all zeros.")
    elif method == "minmax" and cp["max"] == cp["min"]:
        warnings.warn(f"scale: column '{c}' is constant; "
                      f"minmax output is all zeros.")
    elif method == "robust" and cp["iqr"] == 0:
        warnings.warn(f"scale: column '{c}' has zero IQR; "
                      f"robust output is all zeros.")
    elif method == "maxabs" and cp["max_abs"] == 0:
        warnings.warn(f"scale: column '{c}' is all zeros; "
                      f"maxabs output is unchanged.")


def _scale_metrics(num: pd.Series, scaled: pd.Series) -> dict:
    return {
        "mean_before": float(num.mean()), "std_before": float(num.std(ddof=0)),
        "min_before": float(num.min()), "max_before": float(num.max()),
        "mean_after": float(scaled.mean()), "std_after": float(scaled.std(ddof=0)),
        "min_after": float(scaled.min()), "max_after": float(scaled.max()),
    }


def scale(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "standard",
    *,
    inplace: bool = False,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.2,
    dpi: int = 110,
):
    """Place numeric columns on a common scale.

    Two modes. In FIT mode (``params=None``) the scaler statistics are learned
    from ``df``. In APPLY mode (``params`` supplied) the saved statistics are
    applied verbatim with no re-fitting -- the safeguard against statistic
    leakage (computing the mean on train+test together).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    cols : sequence of str, optional
        Numeric columns to scale. If ``None`` all numeric (non-boolean)
        columns are used. Ignored in apply mode (columns come from params).
    method : {'standard', 'minmax', 'robust', 'maxabs', 'compare'}
        'standard' -> (x - mean) / std            (z-score)
        'minmax'   -> (x - min) / (max - min)     (0..1)
        'robust'   -> (x - median) / IQR          (outlier-resistant)
        'maxabs'   -> x / max(|x|)                (-1..1, sign preserved)
        'compare'  -> writes nothing; reports the resulting range/spread of
                      every candidate scaler so you can choose.
    inplace : bool, default False
        If False a new column ``<col>_<method>`` is added and the source kept.
        If True the source column is overwritten.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the learned params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        And, when requested, the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.scale(df_train, cols=['price', 'age'], method='robust',
    ...                     return_params=True)
    >>> df_te = dx.scale(df_test, params=p)            # apply, no re-fit
    >>> dx.scale(df, cols=['price'], method='compare')  # explore options
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _scale_apply(df, params, inplace, show, plot, return_df,
                            return_params, return_fig, decimals, df_name,
                            fig_width, fig_height, dpi)

    # ---- FIT MODE -------------------------------------------------------
    if method not in _VALID_SCALE_METHODS:
        raise ValueError(
            f"'method' must be one of {_VALID_SCALE_METHODS}, got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "method='compare' does not fit parameters; call scale with a "
            "concrete method (standard/minmax/robust/maxabs) to get params.")

    cols = _resolve_cols(df, cols, "scale")

    if method == "compare":
        return _scale_compare(df, cols, show, plot, return_df, return_params,
                              return_fig, decimals, df_name,
                              fig_width, fig_height, dpi)

    out = df.copy()
    out.attrs = dict(df.attrs)
    col_params, rows, col_map = {}, [], {}
    for c in cols:
        num = pd.to_numeric(df[c], errors="coerce")
        v = num.dropna().astype(float).to_numpy()
        if v.size == 0:
            raise ValueError(f"scale: column '{c}' has no numeric values.")
        cp = _scale_fit_one(v, method)
        _scale_warn_degenerate(c, method, cp)
        scaled = _scale_apply_one(num, method, cp)
        new_col = c if inplace else f"{c}_{method}"
        out[new_col] = scaled
        col_map[c] = new_col
        cp_entry = dict(cp)
        cp_entry["source"] = c
        cp_entry["new_col"] = new_col
        col_params[c] = cp_entry
        rows.append(_scale_metrics(num, scaled))

    params_out = {
        "function": "scale",
        "method": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "columns": col_params,
        "metadata": {"inplace": bool(inplace), "n_cols": len(cols)},
    }

    placement = ("overwrote source column(s)" if inplace
                 else f"added new *_{method} column(s)")
    decision = (f"Fitted '{method}' scaler on {len(cols)} column(s); "
                f"{placement}. Apply to held-out data with "
                f"scale(df_test, params=...).")

    _append_audit(out, {
        "stage": "feature_scaling",
        "function": "scale",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "cols": list(cols),
                   "inplace": bool(inplace)},
        "decision": decision,
    })

    summary = pd.DataFrame(rows, index=list(cols))
    summary.index.name = "source"
    if show:
        _print_header(f"Numeric scaling for: {df_name}  "
                      f"(method={method}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_scale(df, out, col_map, method, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _scale_apply(df, params, inplace, show, plot, return_df,
                 return_params, return_fig, decimals, df_name,
                 fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "scale":
        got = params.get("function") if isinstance(params, dict) else params
        raise ValueError(f"params dict is not for 'scale' (function={got!r}).")
    method = params["method"]
    col_params = params["columns"]
    # Decision 3(a): explicit rejection on column mismatch.
    missing = [src for src in col_params if src not in df.columns]
    if missing:
        raise KeyError(
            f"scale apply failed: params expects column(s) {missing} which are "
            f"not present in this DataFrame. The data does not match the "
            f"fitted scaler.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    rows, col_map = [], {}
    for src, cp in col_params.items():
        num = pd.to_numeric(df[src], errors="coerce")
        scaled = _scale_apply_one(num, method, cp)
        new_col = src if inplace else cp.get("new_col", f"{src}_{method}")
        out[new_col] = scaled
        col_map[src] = new_col
        rows.append(_scale_metrics(num, scaled))

    decision = (f"Applied saved '{method}' scaler (fitted "
                f"{params.get('fit_at', '?')}) to {len(col_params)} "
                f"column(s); no re-fit -- leakage-safe.")
    _append_audit(out, {
        "stage": "feature_scaling",
        "function": "scale",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"method": method, "cols": list(col_params),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    summary = pd.DataFrame(rows, index=list(col_params))
    summary.index.name = "source"
    if show:
        _print_header(f"Numeric scaling for: {df_name}  "
                      f"(method={method}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_scale(df, out, col_map, method, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _scale_compare(df, cols, show, plot, return_df, return_params,
                   return_fig, decimals, df_name,
                   fig_width, fig_height, dpi):
    rows, index = [], []
    for c in cols:
        num = pd.to_numeric(df[c], errors="coerce")
        v = num.dropna().astype(float).to_numpy()
        if v.size == 0:
            raise ValueError(f"scale: column '{c}' has no numeric values.")
        for m in _SCALE_CANDIDATES:
            cp = _scale_fit_one(v, m)
            scaled = _scale_apply_one(num, m, cp)
            index.append((c, m))
            rows.append({
                "min": float(scaled.min()), "max": float(scaled.max()),
                "mean": float(scaled.mean()),
                "std": float(scaled.std(ddof=0)),
            })
    summary = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        index, names=["column", "method"]))

    decision = (f"Compared {len(_SCALE_CANDIDATES)} scaler(s) on {len(cols)} "
                f"column(s). Table shows the resulting range and spread of "
                f"each. Scaling preserves distribution shape; it only changes "
                f"location/spread. No columns written -- pick a method then "
                f"call scale(method=...).")
    if show:
        _print_header(f"Scale COMPARE for: {df_name}  (nothing written)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_scale_compare(df, cols, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# Plotting helpers
# ===========================================================================

def _hist_bins(n: int) -> int:
    if n <= 1:
        return 10
    return max(10, min(60, int(np.sqrt(n))))


def _plot_transform(df_before, out, col_map, method,
                    fig_width, fig_height, dpi):
    items = list(col_map.items())[:4]
    n = len(items)
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 2, figsize=(fig_width, fig_height * n), dpi=dpi)
    axes = np.atleast_2d(axes)
    for i, (src, new_col) in enumerate(items):
        before = pd.to_numeric(df_before[src], errors="coerce").dropna()
        after = pd.to_numeric(out[new_col], errors="coerce")
        after = after[np.isfinite(after)]
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.hist(before, bins=_hist_bins(len(before)),
                 color="#ec7853", edgecolor="black", alpha=0.85)
        ax0.set_title(f"'{src}' before  (skew={before.skew():.3f})",
                      fontweight="bold")
        ax0.set_xlabel(src)
        ax0.set_ylabel("count")
        ax1.hist(after, bins=_hist_bins(len(after)),
                 color="#2ca02c", edgecolor="black", alpha=0.85)
        ax1.set_title(f"'{new_col}' after {method}  (skew={after.skew():.3f})",
                      fontweight="bold")
        ax1.set_xlabel(new_col)
        ax1.set_ylabel("count")
    fig.suptitle(f"Numeric transform -- {method}  (Stage 4.1)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_transform_compare(df, cols, fig_width, fig_height, dpi):
    c = cols[0]
    num = pd.to_numeric(df[c], errors="coerce").dropna()
    panels = [("raw", num)]
    for m in _TRANSFORM_CANDIDATES:
        ok = True
        if m in ("log", "boxcox") and (num <= 0).any():
            ok = False
        elif m == "sqrt" and (num < 0).any():
            ok = False
        elif m == "log1p" and (num <= -1).any():
            ok = False
        if not ok:
            panels.append((m, None))
            continue
        try:
            t, _ = _transform_one(df[c], m)
            t = t[np.isfinite(t)]
            panels.append((m, t))
        except Exception:
            panels.append((m, None))
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width, fig_height * nrows), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, series) in zip(axes, panels):
        if series is None or len(series) == 0:
            ax.text(0.5, 0.5, f"{label}\n(not valid for\nthis column)",
                    ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            continue
        ax.hist(series, bins=_hist_bins(len(series)),
                color="#4c72b0", edgecolor="black", alpha=0.85)
        ax.set_title(f"{label}  (skew={series.skew():.3f})", fontweight="bold")
        ax.set_ylabel("count")
    for ax in axes[len(panels):]:
        ax.set_axis_off()
    fig.suptitle(f"Transform COMPARE -- column '{c}'  (Stage 4.1)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_scale(df_before, out, col_map, method,
                fig_width, fig_height, dpi):
    items = list(col_map.items())[:4]
    n = len(items)
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 2, figsize=(fig_width, fig_height * n), dpi=dpi)
    axes = np.atleast_2d(axes)
    for i, (src, new_col) in enumerate(items):
        before = pd.to_numeric(df_before[src], errors="coerce").dropna()
        after = pd.to_numeric(out[new_col], errors="coerce").dropna()
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.hist(before, bins=_hist_bins(len(before)),
                 color="#ec7853", edgecolor="black", alpha=0.85)
        ax0.axvline(before.mean(), color="black", linestyle="--", linewidth=1)
        ax0.set_title(f"'{src}' before  "
                      f"(min={before.min():.2f}, max={before.max():.2f})",
                      fontweight="bold")
        ax0.set_xlabel(src)
        ax0.set_ylabel("count")
        ax1.hist(after, bins=_hist_bins(len(after)),
                 color="#2ca02c", edgecolor="black", alpha=0.85)
        ax1.axvline(after.mean(), color="black", linestyle="--", linewidth=1)
        ax1.set_title(f"'{new_col}' after {method}  "
                      f"(min={after.min():.2f}, max={after.max():.2f})",
                      fontweight="bold")
        ax1.set_xlabel(new_col)
        ax1.set_ylabel("count")
    fig.suptitle(f"Numeric scaling -- {method}  (shape preserved, axis rescaled)"
                 f"  (Stage 4.1)", fontsize=14, fontweight="bold")
    return fig


def _plot_scale_compare(df, cols, fig_width, fig_height, dpi):
    c = cols[0]
    num = pd.to_numeric(df[c], errors="coerce").dropna()
    v = num.astype(float).to_numpy()
    panels = [("raw", num)]
    for m in _SCALE_CANDIDATES:
        cp = _scale_fit_one(v, m)
        panels.append((m, _scale_apply_one(num, m, cp).dropna()))
    ncols = 3
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width, fig_height * nrows), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, series) in zip(axes, panels):
        ax.hist(series, bins=_hist_bins(len(series)),
                color="#4c72b0", edgecolor="black", alpha=0.85)
        ax.set_title(f"{label}  "
                     f"(range {series.min():.2f}..{series.max():.2f})",
                     fontweight="bold")
        ax.set_ylabel("count")
    for ax in axes[len(panels):]:
        ax.set_axis_off()
    fig.suptitle(f"Scale COMPARE -- column '{c}'  "
                 f"(same shape, different axis)  (Stage 4.1)",
                 fontsize=14, fontweight="bold")
    return fig


# ===========================================================================
# 3. bin  --  discretise continuous values into ordered bins
# ===========================================================================

_VALID_BIN_METHODS = ("equal_width", "quantile", "kmeans", "compare")
_BIN_CANDIDATES = ("equal_width", "quantile", "kmeans")


def _kmeans_1d(vals: np.ndarray, k: int, n_iter: int = 100,
               tol: float = 1e-9) -> np.ndarray:
    """Deterministic 1-D k-means; return the sorted cluster centres.

    Centres are initialised at evenly-spaced quantiles -- there is no random
    seed -- so the same column always yields the same bin edges. This keeps
    binning reproducible between a fit and any later apply.

    Parameters
    ----------
    vals : numpy.ndarray
        1-D array of finite floats (NaN already removed).
    k : int
        Desired number of clusters.

    Returns
    -------
    numpy.ndarray
        Sorted cluster centres. May contain fewer than ``k`` entries when the
        column has fewer than ``k`` distinct values.
    """
    v = np.sort(np.asarray(vals, dtype=float))
    uniq = np.unique(v)
    if uniq.size <= k:
        return uniq.astype(float)
    qs = np.linspace(0.0, 1.0, k + 2)[1:-1]
    centres = np.unique(np.quantile(v, qs))
    for _ in range(n_iter):
        dist = np.abs(v[:, None] - centres[None, :])
        labels = np.argmin(dist, axis=1)
        new = np.array([v[labels == j].mean() if np.any(labels == j)
                        else centres[j] for j in range(centres.size)])
        new = np.unique(np.sort(new))
        if new.size == centres.size and np.max(np.abs(new - centres)) < tol:
            centres = new
            break
        centres = new
    return np.sort(centres)


def _bin_fit_one(vals: np.ndarray, method: str, n_bins: int):
    """Learn bin edges from a 1-D array of non-NaN floats.

    Returns
    -------
    (edges, note)
        ``edges`` is a strictly-increasing list of floats; ``note`` is an
        optional human-readable string set when the requested bin count could
        not be honoured exactly (e.g. tied quantiles collapsed some edges).
    """
    v = np.asarray(vals, dtype=float)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    note = None
    if vmin == vmax:
        raise ValueError(
            f"bin: column is constant (every value == {vmin}); there is "
            f"nothing to discretise.")
    if method == "equal_width":
        edges = np.linspace(vmin, vmax, n_bins + 1)
    elif method == "quantile":
        edges = np.quantile(v, np.linspace(0.0, 1.0, n_bins + 1))
    elif method == "kmeans":
        centres = _kmeans_1d(v, n_bins)
        if centres.size < 2:
            raise ValueError(
                "bin: kmeans could not form 2 or more clusters for this "
                "column; try method='equal_width'.")
        inner = (centres[1:] + centres[:-1]) / 2.0
        edges = np.concatenate([[vmin], inner, [vmax]])
    else:
        raise ValueError(f"Unknown bin method {method!r}")
    edges = np.unique(np.asarray(edges, dtype=float))
    if edges.size < 2:
        raise ValueError(
            f"bin: '{method}' produced fewer than 2 distinct edges; the "
            f"column is too concentrated to split into {n_bins} bins.")
    actual = edges.size - 1
    if actual < n_bins:
        note = (f"requested {n_bins} bins but method '{method}' supports only "
                f"{actual} distinct bin(s) for this column")
    return [float(e) for e in edges], note


def _bin_labels(n_bins_actual: int, labels):
    """Resolve bin labels: default ``B1..Bk`` or validate a caller list."""
    if labels is None:
        return [f"B{i + 1}" for i in range(n_bins_actual)]
    labels = list(labels)
    if len(labels) != n_bins_actual:
        raise ValueError(
            f"bin: 'labels' has {len(labels)} entry/entries but "
            f"{n_bins_actual} bin(s) were formed; pass exactly "
            f"{n_bins_actual} label(s) or labels=None.")
    if len(set(map(str, labels))) != len(labels):
        raise ValueError("bin: 'labels' must be unique (bins are ordered).")
    return labels


def _bin_apply_one(num: pd.Series, edges, labels) -> pd.Series:
    """Assign each value to an ordered bin.

    Apply-mode decision: values below the lowest fitted edge or above the
    highest are CLIPPED into the outer bins rather than turned into NaN, so
    held-out rows never silently vanish.
    """
    edges = [float(e) for e in edges]
    clipped = num.clip(lower=edges[0], upper=edges[-1])
    return pd.cut(clipped, bins=edges, labels=labels,
                  include_lowest=True, ordered=True)


def bin(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "equal_width",
    n_bins: int = 5,
    *,
    labels: Optional[Sequence] = None,
    inplace: bool = False,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.2,
    dpi: int = 110,
):
    """Discretise continuous numeric columns into ordered bins.

    Two modes. In FIT mode (``params=None``) the bin edges are learned from
    ``df``. In APPLY mode (``params`` supplied) the saved edges are applied
    verbatim with no re-fitting -- the safeguard against the 'bin edges chosen
    on the full dataset' anti-pattern in FEATURES_PHILOSOPHY.md section 5.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    cols : sequence of str, optional
        Numeric columns to bin. If ``None`` all numeric (non-boolean) columns
        are used. Ignored in apply mode (columns come from params).
    method : {'equal_width', 'quantile', 'kmeans', 'compare'}
        'equal_width' -> edges equally spaced between min and max.
        'quantile'    -> edges at quantiles (each bin holds ~equal counts).
        'kmeans'      -> edges from 1-D k-means cluster centres.
        'compare'     -> writes nothing; reports how balanced each method's
                         bins would be so you can choose.
    n_bins : int, default 5
        Desired number of bins (>= 2). Ties may yield fewer; a warning is
        emitted when that happens.
    labels : sequence, optional
        Custom ordered bin labels. Default labels are ``B1..Bk``.
    inplace : bool, default False
        If False a new column ``<col>_bin`` is added and the source kept.
        If True the source column is overwritten with the binned column.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the learned params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        And, when requested, the params dict and/or the matplotlib figure.
        The binned column has an ordered ``Categorical`` dtype.

    Examples
    --------
    >>> df_tr, p = dx.bin(df_train, cols=['price'], method='quantile',
    ...                   n_bins=4, return_params=True)
    >>> df_te = dx.bin(df_test, params=p)              # apply, no re-fit
    >>> dx.bin(df, cols=['price'], method='compare')   # explore options
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _bin_apply(df, params, inplace, show, plot, return_df,
                          return_params, return_fig, decimals, df_name,
                          fig_width, fig_height, dpi)

    # ---- FIT MODE -------------------------------------------------------
    if method not in _VALID_BIN_METHODS:
        raise ValueError(
            f"'method' must be one of {_VALID_BIN_METHODS}, got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "method='compare' does not fit parameters; call bin with a "
            "concrete method (equal_width/quantile/kmeans) to obtain params.")
    if not isinstance(n_bins, (int, np.integer)) or bool(n_bins < 2):
        raise ValueError(f"'n_bins' must be an integer >= 2, got {n_bins!r}")
    n_bins = int(n_bins)

    cols = _resolve_cols(df, cols, "bin")

    if method == "compare":
        return _bin_compare(df, cols, n_bins, show, plot, return_df,
                            return_params, return_fig, decimals, df_name,
                            fig_width, fig_height, dpi)

    out = df.copy()
    out.attrs = dict(df.attrs)
    col_params, rows, plot_items = {}, [], []
    for c in cols:
        num = pd.to_numeric(df[c], errors="coerce")
        v = num.dropna().astype(float).to_numpy()
        if v.size == 0:
            raise ValueError(f"bin: column '{c}' has no numeric values.")
        edges, note = _bin_fit_one(v, method, n_bins)
        actual = len(edges) - 1
        lab = _bin_labels(actual, labels)
        if note:
            warnings.warn(f"bin: column '{c}' -- {note}.")
        binned = _bin_apply_one(num, edges, lab)
        new_col = c if inplace else f"{c}_bin"
        out[new_col] = binned
        col_params[c] = {
            "edges": edges, "labels": list(lab), "n_bins": actual,
            "source": c, "new_col": new_col,
        }
        plot_items.append((c, new_col, edges, list(lab)))
        counts = binned.value_counts().reindex(lab, fill_value=0)
        rows.append({
            "new_col": new_col,
            "n_bins": actual,
            "min_count": int(counts.min()),
            "max_count": int(counts.max()),
            "n_empty_bins": int((counts == 0).sum()),
            "n_na": int(binned.isna().sum()),
        })

    params_out = {
        "function": "bin",
        "method": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "columns": col_params,
        "metadata": {"inplace": bool(inplace), "n_cols": len(cols),
                     "n_bins": int(n_bins)},
    }

    placement = ("overwrote source column(s)" if inplace
                 else "added new *_bin column(s)")
    decision = (f"Fitted '{method}' binning on {len(cols)} column(s) into up "
                f"to {n_bins} ordered bin(s); {placement}. Created ordinal "
                f"discretised feature(s).")

    _append_audit(out, {
        "stage": "feature_binning",
        "function": "bin",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "cols": list(cols),
                   "n_bins": int(n_bins), "inplace": bool(inplace)},
        "decision": decision,
    })

    summary = pd.DataFrame(rows, index=list(cols))
    summary.index.name = "source"
    if show:
        _print_header(f"Numeric binning for: {df_name}  "
                      f"(method={method}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_bin(df, out, plot_items, method,
                        fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _bin_apply(df, params, inplace, show, plot, return_df,
               return_params, return_fig, decimals, df_name,
               fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "bin":
        got = params.get("function") if isinstance(params, dict) else params
        raise ValueError(f"params dict is not for 'bin' (function={got!r}).")
    method = params["method"]
    col_params = params["columns"]
    # Decision 3(a): explicit rejection on column mismatch.
    missing = [src for src in col_params if src not in df.columns]
    if missing:
        raise KeyError(
            f"bin apply failed: params expects column(s) {missing} which are "
            f"not present in this DataFrame. The data does not match the "
            f"fitted binner.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    rows, plot_items = [], []
    for src, cp in col_params.items():
        num = pd.to_numeric(df[src], errors="coerce")
        edges = [float(e) for e in cp["edges"]]
        lab = list(cp["labels"])
        present = num.dropna()
        n_below = int((present < edges[0]).sum())
        n_above = int((present > edges[-1]).sum())
        binned = _bin_apply_one(num, edges, lab)
        new_col = src if inplace else cp.get("new_col", f"{src}_bin")
        out[new_col] = binned
        plot_items.append((src, new_col, edges, lab))
        if n_below or n_above:
            warnings.warn(
                f"bin: column '{src}' -- {n_below + n_above} held-out value(s) "
                f"fell outside the fitted edge range and were clipped into the "
                f"outer bin(s).")
        counts = binned.value_counts().reindex(lab, fill_value=0)
        rows.append({
            "new_col": new_col,
            "n_bins": len(lab),
            "n_clipped": n_below + n_above,
            "min_count": int(counts.min()),
            "max_count": int(counts.max()),
            "n_na": int(binned.isna().sum()),
        })

    decision = (f"Applied saved '{method}' binning (fitted "
                f"{params.get('fit_at', '?')}) to {len(col_params)} "
                f"column(s); no re-fit -- leakage-safe.")
    _append_audit(out, {
        "stage": "feature_binning",
        "function": "bin",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"method": method, "cols": list(col_params),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    summary = pd.DataFrame(rows, index=list(col_params))
    summary.index.name = "source"
    if show:
        _print_header(f"Numeric binning for: {df_name}  "
                      f"(method={method}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_bin(df, out, plot_items, method,
                        fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _bin_compare(df, cols, n_bins, show, plot, return_df, return_params,
                 return_fig, decimals, df_name, fig_width, fig_height, dpi):
    rows, index = [], []
    for c in cols:
        num = pd.to_numeric(df[c], errors="coerce")
        v = num.dropna().astype(float).to_numpy()
        if v.size == 0:
            raise ValueError(f"bin: column '{c}' has no numeric values.")
        for m in _BIN_CANDIDATES:
            try:
                edges, _ = _bin_fit_one(v, m, n_bins)
                lab = [f"B{i + 1}" for i in range(len(edges) - 1)]
                binned = _bin_apply_one(num, edges, lab)
                counts = binned.value_counts().reindex(lab, fill_value=0)
                mx = int(counts.max())
                index.append((c, m))
                rows.append({
                    "n_bins": len(lab),
                    "min_count": int(counts.min()),
                    "max_count": mx,
                    "balance": float(counts.min() / mx) if mx else np.nan,
                    "n_empty": int((counts == 0).sum()),
                })
            except Exception:
                index.append((c, m))
                rows.append({"n_bins": np.nan, "min_count": np.nan,
                             "max_count": np.nan, "balance": np.nan,
                             "n_empty": np.nan})
    summary = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        index, names=["column", "method"]))

    decision = (f"Compared {len(_BIN_CANDIDATES)} binning method(s) on "
                f"{len(cols)} column(s) at n_bins={n_bins}. 'balance' is "
                f"min_count/max_count (1.0 = perfectly even bins; higher is "
                f"more even). No columns written -- pick a method then call "
                f"bin(method=...).")
    if show:
        _print_header(f"Bin COMPARE for: {df_name}  (nothing written)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_bin_compare(df, cols, n_bins, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# 4. encode  --  turn categories into numbers without losing information
# ===========================================================================

_VALID_ENCODE_METHODS = ("onehot", "ordinal", "target", "frequency", "compare")
_ENCODE_CANDIDATES = ("onehot", "ordinal", "target", "frequency")
_CARDINALITY_WARN = 50


def _auto_categorical_cols(df: pd.DataFrame) -> list:
    """Return object / string / categorical column names."""
    out = []
    for c in df.columns:
        dt = df[c].dtype
        if isinstance(dt, pd.CategoricalDtype):
            out.append(c)
        elif (pd.api.types.is_object_dtype(dt)
              or pd.api.types.is_string_dtype(dt)):
            out.append(c)
    return out


def _resolve_cat_cols(df: pd.DataFrame, cols, func_name: str) -> list:
    """Validate an explicit cols selector or auto-pick categorical columns."""
    if cols is None:
        chosen = _auto_categorical_cols(df)
        if not chosen:
            raise ValueError(
                f"{func_name}: no categorical (object/string/category) "
                f"columns found. Pass cols= explicitly.")
        return chosen
    chosen = list(cols)
    bad = [c for c in chosen if c not in df.columns]
    if bad:
        raise KeyError(
            f"{func_name}: cols references columns not in df: {bad}")
    return chosen


def _encode_categories(col: pd.Series) -> list:
    """Sorted list of string category keys present (non-null) in a column."""
    keys = col[col.notna()].astype(str)
    uniq = keys.unique().tolist()
    try:
        return sorted(uniq)
    except TypeError:  # pragma: no cover - defensive
        return sorted(uniq, key=str)


def _resolve_y(df: pd.DataFrame, y) -> pd.Series:
    """Resolve the target ``y`` to a numeric Series aligned with ``df``."""
    if y is None:
        raise ValueError(
            "encode: method='target' requires y (the target). Pass y= as a "
            "Series, an array, or the name of a column in df.")
    if isinstance(y, str):
        if y not in df.columns:
            raise KeyError(f"encode: y='{y}' is not a column in df.")
        raw = df[y]
    elif isinstance(y, pd.Series):
        raw = y
    else:
        raw = pd.Series(np.asarray(y))
    if len(raw) != len(df):
        raise ValueError(
            f"encode: y has length {len(raw)} but df has {len(df)} rows; "
            f"they must align.")
    yv = pd.to_numeric(pd.Series(np.asarray(raw), index=df.index),
                       errors="coerce")
    if int(yv.notna().sum()) == 0:
        raise ValueError(
            "encode: target y has no numeric values; target encoding needs a "
            "numeric (regression or 0/1) target.")
    return yv


def _resolve_order(order, cols, c):
    """Pick the explicit category order for column ``c`` (or None)."""
    if order is None:
        return None
    if isinstance(order, dict):
        return order.get(c)
    if len(cols) == 1:
        return list(order)
    raise ValueError(
        "encode: 'order' as a flat list is only allowed for a single column; "
        "pass a dict {column: [ordered categories]} for multiple columns.")


def _target_oof(key_arr: np.ndarray, y_arr: np.ndarray, valid_pos: np.ndarray,
                n_folds: int, global_mean: float) -> np.ndarray:
    """Out-of-fold category means -- the leakage-safe TRAINING encoding.

    Each row is encoded with the category mean computed on the OTHER folds, so
    a row never sees its own target. This refuses the 'target encoding without
    held-out folds' anti-pattern in FEATURES_PHILOSOPHY.md section 5.
    """
    n = len(key_arr)
    result = np.full(n, np.nan, dtype="float64")
    k = min(int(n_folds), len(valid_pos))
    if k < 2:
        # Too few labelled rows to fold; fall back to the global mean.
        result[valid_pos] = global_mean
        return result
    rng = np.random.default_rng(0)
    perm = rng.permutation(valid_pos)
    for hold in np.array_split(perm, k):
        if len(hold) == 0:
            continue
        rest = np.setdiff1d(perm, hold, assume_unique=True)
        means = (pd.Series(y_arr[rest], index=key_arr[rest])
                 .groupby(level=0).mean().to_dict())
        for j in hold:
            result[j] = means.get(key_arr[j], global_mean)
    return result


def _encode_map_apply_series(col: pd.Series, mapping: dict, default):
    """Map a column through a category->number dict (apply-style).

    Returns
    -------
    (series, n_unknown)
        ``series`` is float64 with NaN kept where the source was NaN; unseen
        categories become ``default`` (or NaN when ``default`` is None).
    """
    keys = col.astype(str)
    notna = col.notna()
    mapped = keys.map(mapping)
    unknown_mask = notna & mapped.isna()
    n_unknown = int(unknown_mask.sum())
    if default is not None:
        mapped = mapped.mask(unknown_mask, default)
    mapped = mapped.where(notna, other=np.nan)
    return mapped.astype("float64"), n_unknown


def _encode_fit_column(df, c, method, drop_first, order_for_col,
                       y_resolved, n_folds, inplace):
    """Fit one column; return (params_entry, {new_col_name: Series})."""
    col = df[c]
    cats = _encode_categories(col)
    if not cats:
        raise ValueError(
            f"encode: column '{c}' has no non-null categories to encode.")
    new_cols: dict = {}

    if method == "onehot":
        if len(cats) > _CARDINALITY_WARN:
            warnings.warn(
                f"encode: column '{c}' has {len(cats)} categories "
                f"(> {_CARDINALITY_WARN}); one-hot will create "
                f"{len(cats) - (1 if drop_first else 0)} columns. Consider "
                f"method='frequency' or 'target' for high-cardinality "
                f"features.")
        used = cats[1:] if drop_first else list(cats)
        keys = col.astype(str)
        notna = col.notna()
        names = [f"{c}_{cat}" for cat in used]
        for cat, name in zip(used, names):
            new_cols[name] = ((keys == cat) & notna).astype("int64")
        entry = {"method": "onehot", "categories": list(cats),
                 "used_categories": used, "drop_first": bool(drop_first),
                 "source": c, "new_cols": names}

    elif method == "ordinal":
        if order_for_col is not None:
            order_list = [str(x) for x in order_for_col]
            if len(set(order_list)) != len(order_list):
                raise ValueError(
                    f"encode: ordinal 'order' for column '{c}' has duplicates.")
            uncovered = [x for x in cats if x not in order_list]
            if uncovered:
                raise ValueError(
                    f"encode: ordinal 'order' for column '{c}' does not cover "
                    f"categories {uncovered}.")
            mapping = {cat: i for i, cat in enumerate(order_list)}
        else:
            mapping = {cat: i for i, cat in enumerate(cats)}
            warnings.warn(
                f"encode: ordinal encoding of '{c}' imposes a rank order but "
                f"no order= was given; used alphabetical order {list(cats)}. "
                f"Pass order= to set a meaningful ranking.")
        name = c if inplace else f"{c}_ord"
        vals, _ = _encode_map_apply_series(col, mapping, None)
        new_cols[name] = vals
        entry = {"method": "ordinal", "mapping": mapping, "default": None,
                 "source": c, "new_cols": [name]}

    elif method == "frequency":
        keys = col[col.notna()].astype(str)
        freq = keys.value_counts(normalize=True)
        mapping = {k: float(v) for k, v in freq.items()}
        name = c if inplace else f"{c}_freq"
        vals, _ = _encode_map_apply_series(col, mapping, 0.0)
        new_cols[name] = vals
        entry = {"method": "frequency", "mapping": mapping, "default": 0.0,
                 "source": c, "new_cols": [name]}

    elif method == "target":
        col_notna = col.notna()
        both = col_notna & y_resolved.notna()
        if int(both.sum()) == 0:
            raise ValueError(
                f"encode: column '{c}' has no rows with both a category and a "
                f"non-null target; cannot target-encode.")
        keys = col.astype(str)
        global_mean = float(y_resolved[both].mean())
        full_means = y_resolved[both].groupby(keys[both]).mean()
        mapping = {k: float(v) for k, v in full_means.items()}
        key_arr = keys.to_numpy()
        y_arr = y_resolved.to_numpy(dtype="float64")
        valid_pos = np.where(both.to_numpy())[0]
        oof = _target_oof(key_arr, y_arr, valid_pos, n_folds, global_mean)
        encoded = pd.Series(oof, index=df.index, dtype="float64")
        # Category known but target missing -> fill via the full-data mapping.
        cat_only = col_notna & ~both
        if bool(cat_only.any()):
            filled = keys[cat_only].map(mapping).fillna(global_mean)
            encoded.loc[cat_only] = filled.to_numpy(dtype="float64")
        name = c if inplace else f"{c}_target"
        new_cols[name] = encoded
        entry = {"method": "target", "mapping": mapping,
                 "default": global_mean, "source": c, "new_cols": [name]}

    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unknown encode method {method!r}")

    return entry, new_cols


def encode(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "onehot",
    *,
    y=None,
    n_folds: int = 5,
    order=None,
    drop_first: bool = False,
    handle_unknown: str = "ignore",
    inplace: bool = False,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.2,
    dpi: int = 110,
):
    """Encode categorical columns into model-ready numeric features.

    Two modes. In FIT mode (``params=None``) the encoding maps are learned
    from ``df``. In APPLY mode (``params`` supplied) the saved maps are applied
    verbatim with no re-fitting -- the safeguard against leakage.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    cols : sequence of str, optional
        Categorical columns to encode. If ``None`` all object/string/category
        columns are used. Ignored in apply mode (columns come from params).
    method : {'onehot', 'ordinal', 'target', 'frequency', 'compare'}
        'onehot'    -> one 0/1 column per category (``<col>_<category>``).
        'ordinal'   -> one integer-rank column (``<col>_ord``); needs order=.
        'target'    -> mean of ``y`` per category (``<col>_target``); the
                       training output is computed out-of-fold (K-fold) to
                       prevent target leakage.
        'frequency' -> category proportion in the data (``<col>_freq``).
        'compare'   -> writes nothing; reports how many columns each method
                       would add so you can choose.
    y : Series, array, or str, optional
        The target. Required for method='target' (a column name is accepted).
    n_folds : int, default 5
        Number of out-of-fold folds for target encoding.
    order : list or dict, optional
        Explicit category order for ordinal encoding. A flat list for a single
        column, or ``{column: [ordered categories]}`` for several.
    drop_first : bool, default False
        For one-hot, drop the first category to avoid the dummy-variable trap.
    handle_unknown : {'ignore', 'error'}, default 'ignore'
        Apply-mode behaviour for categories unseen during fit. 'ignore' maps
        them to the default (one-hot: all-zero row); 'error' raises.
    inplace : bool, default False
        If False new column(s) are added and the source kept. If True the
        source column is replaced (one-hot drops it and adds the dummies).
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the learned params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        And, when requested, the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.encode(df_train, cols=['city'], method='target',
    ...                      y=df_train['churn'], return_params=True)
    >>> df_te = dx.encode(df_test, params=p)              # apply, no re-fit
    >>> dx.encode(df, cols=['city'], method='compare')    # explore options
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _encode_apply(df, params, show, plot, return_df,
                             return_params, return_fig, decimals, df_name,
                             fig_width, fig_height, dpi)

    # ---- FIT MODE -------------------------------------------------------
    if method not in _VALID_ENCODE_METHODS:
        raise ValueError(
            f"'method' must be one of {_VALID_ENCODE_METHODS}, got {method!r}")
    if handle_unknown not in ("ignore", "error"):
        raise ValueError(
            f"'handle_unknown' must be 'ignore' or 'error', got "
            f"{handle_unknown!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "method='compare' does not fit parameters; call encode with a "
            "concrete method (onehot/ordinal/target/frequency) to get params.")
    if method == "target" and (not isinstance(n_folds, (int, np.integer))
                               or bool(n_folds < 2)):
        raise ValueError(f"'n_folds' must be an integer >= 2, got {n_folds!r}")

    cols = _resolve_cat_cols(df, cols, "encode")

    if method == "compare":
        return _encode_compare(df, cols, y, show, plot, return_df,
                               return_params, return_fig, decimals, df_name,
                               fig_width, fig_height, dpi)

    y_resolved = _resolve_y(df, y) if method == "target" else None

    out = df.copy()
    out.attrs = dict(df.attrs)
    col_params, rows, plot_items = {}, [], []
    for c in cols:
        order_for_col = _resolve_order(order, cols, c)
        entry, new_cols = _encode_fit_column(
            df, c, method, drop_first, order_for_col, y_resolved,
            int(n_folds), inplace)
        if inplace and method == "onehot":
            out = out.drop(columns=[c])
        for name, series in new_cols.items():
            out[name] = series
        col_params[c] = entry
        plot_items.append((c, entry))
        n_cats = (len(entry["categories"]) if method == "onehot"
                  else len(entry["mapping"]))
        names = list(new_cols.keys())
        rows.append({
            "source": c,
            "method": method,
            "n_categories": n_cats,
            "n_new_cols": len(names),
            "new_cols": ", ".join(names[:4]) + ("  ..." if len(names) > 4
                                                else ""),
        })

    target_name = None
    if method == "target":
        target_name = y if isinstance(y, str) else getattr(y, "name", None)
    params_out = {
        "function": "encode",
        "method": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "columns": col_params,
        "metadata": {
            "inplace": bool(inplace),
            "n_cols": len(cols),
            "drop_first": bool(drop_first),
            "handle_unknown": handle_unknown,
            "n_folds": int(n_folds) if method == "target" else None,
            "target": target_name,
        },
    }

    total_new = sum(r["n_new_cols"] for r in rows)
    decision = (f"Fitted '{method}' encoding on {len(cols)} categorical "
                f"column(s); produced {total_new} numeric feature column(s). "
                f"Apply to held-out data with encode(df_test, params=...).")

    _append_audit(out, {
        "stage": "feature_encoding",
        "function": "encode",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "cols": list(cols),
                   "inplace": bool(inplace)},
        "decision": decision,
    })

    summary = pd.DataFrame(rows).set_index("source")
    if show:
        _print_header(f"Categorical encoding for: {df_name}  "
                      f"(method={method}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_encode(df, plot_items, method, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _encode_apply(df, params, show, plot, return_df, return_params,
                  return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "encode":
        got = params.get("function") if isinstance(params, dict) else params
        raise ValueError(f"params dict is not for 'encode' (function={got!r}).")
    method = params["method"]
    col_params = params["columns"]
    meta = params.get("metadata", {})
    handle_unknown = meta.get("handle_unknown", "ignore")
    inplace = bool(meta.get("inplace", False))
    # Decision 3(a): explicit rejection on column mismatch.
    missing = [src for src in col_params if src not in df.columns]
    if missing:
        raise KeyError(
            f"encode apply failed: params expects column(s) {missing} which "
            f"are not present in this DataFrame. The data does not match the "
            f"fitted encoder.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    rows, plot_items = [], []
    for src, entry in col_params.items():
        m = entry.get("method", method)
        col = df[src]
        if m == "onehot":
            used = entry["used_categories"]
            names = entry["new_cols"]
            known = set(entry["categories"])
            keys = col.astype(str)
            notna = col.notna()
            unknown_mask = notna & ~keys.isin(known)
            n_unknown = int(unknown_mask.sum())
            if n_unknown and handle_unknown == "error":
                bad = sorted(keys[unknown_mask].unique())[:10]
                raise ValueError(
                    f"encode apply: column '{src}' contains categories unseen "
                    f"during fit: {bad}. Pass handle_unknown='ignore' to map "
                    f"them to an all-zero row.")
            if inplace:
                out = out.drop(columns=[src])
            for cat, name in zip(used, names):
                out[name] = ((keys == cat) & notna).astype("int64")
            n_new = len(names)
            n_cats = len(entry["categories"])
        else:
            mapping = entry["mapping"]
            default = entry["default"]
            keys = col.astype(str)
            notna = col.notna()
            mapped = keys.map(mapping)
            unknown_mask = notna & mapped.isna()
            n_unknown = int(unknown_mask.sum())
            if n_unknown and handle_unknown == "error":
                bad = sorted(keys[unknown_mask].unique())[:10]
                raise ValueError(
                    f"encode apply: column '{src}' contains categories unseen "
                    f"during fit: {bad}. Pass handle_unknown='ignore' to map "
                    f"them to the default value.")
            if default is not None:
                mapped = mapped.mask(unknown_mask, default)
            mapped = mapped.where(notna, other=np.nan).astype("float64")
            name = entry["new_cols"][0]
            out[name] = mapped
            n_new = 1
            n_cats = len(mapping)
        if n_unknown:
            warnings.warn(
                f"encode: column '{src}' had {n_unknown} row(s) with "
                f"categories unseen during fit; handled via "
                f"handle_unknown='{handle_unknown}'.")
        plot_items.append((src, entry))
        rows.append({
            "source": src, "method": m, "n_categories": n_cats,
            "n_new_cols": n_new, "n_unknown": n_unknown,
        })

    decision = (f"Applied saved '{method}' encoding (fitted "
                f"{params.get('fit_at', '?')}) to {len(col_params)} "
                f"column(s); no re-fit -- leakage-safe.")
    _append_audit(out, {
        "stage": "feature_encoding",
        "function": "encode",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"method": method, "cols": list(col_params),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    summary = pd.DataFrame(rows).set_index("source")
    if show:
        _print_header(f"Categorical encoding for: {df_name}  "
                      f"(method={method}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_encode(df, plot_items, method, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _encode_compare(df, cols, y, show, plot, return_df, return_params,
                    return_fig, decimals, df_name, fig_width, fig_height, dpi):
    yv = None
    if y is not None:
        try:
            yv = _resolve_y(df, y)
        except Exception:
            yv = None
    rows, index = [], []
    for c in cols:
        n = len(_encode_categories(df[c]))
        for m in _ENCODE_CANDIDATES:
            if m == "onehot":
                rec = {"n_categories": n, "n_new_cols": n,
                       "note": ("HIGH cardinality" if n > _CARDINALITY_WARN
                                else "ok")}
            elif m == "ordinal":
                rec = {"n_categories": n, "n_new_cols": 1,
                       "note": "imposes a rank order"}
            elif m == "frequency":
                rec = {"n_categories": n, "n_new_cols": 1, "note": "ok"}
            else:  # target
                rec = {"n_categories": n, "n_new_cols": 1,
                       "note": ("ok (y supplied)" if yv is not None
                                else "needs y=")}
            index.append((c, m))
            rows.append(rec)
    summary = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        index, names=["column", "method"]))

    decision = (f"Compared {len(_ENCODE_CANDIDATES)} encoding method(s) on "
                f"{len(cols)} column(s). 'n_new_cols' is how many columns each "
                f"method would add. No columns written -- pick a method then "
                f"call encode(method=...).")
    if show:
        _print_header(f"Encode COMPARE for: {df_name}  (nothing written)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_encode_compare(df, cols, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# Plotting helpers -- Stage 4.2 (bin + encode)
# ===========================================================================

def _plot_bin(df_before, out, plot_items, method,
              fig_width, fig_height, dpi):
    """Per column: continuous histogram with edge lines, then bin counts."""
    items = plot_items[:4]
    n = len(items)
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 2, figsize=(fig_width, fig_height * n), dpi=dpi)
    axes = np.atleast_2d(axes)
    for i, (src, new_col, edges, labels) in enumerate(items):
        before = pd.to_numeric(df_before[src], errors="coerce").dropna()
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.hist(before, bins=_hist_bins(len(before)),
                 color="#ec7853", edgecolor="black", alpha=0.85)
        for e in edges:
            ax0.axvline(e, color="black", linestyle="--", linewidth=1)
        ax0.set_title(f"'{src}' continuous  ({len(edges) - 1} bin edges)",
                      fontweight="bold")
        ax0.set_xlabel(src)
        ax0.set_ylabel("count")
        binned = out[new_col]
        counts = binned.value_counts().reindex(labels, fill_value=0)
        ax1.bar([str(x) for x in labels], counts.to_numpy(),
                color="#2ca02c", edgecolor="black", alpha=0.85)
        ax1.set_title(f"'{new_col}' bin counts", fontweight="bold")
        ax1.set_xlabel("bin")
        ax1.set_ylabel("count")
        ax1.tick_params(axis="x", rotation=45)
    fig.suptitle(f"Binning -- {method}  (Stage 4.2)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_bin_compare(df, cols, n_bins, fig_width, fig_height, dpi):
    c = cols[0]
    num = pd.to_numeric(df[c], errors="coerce")
    v = num.dropna().astype(float).to_numpy()
    panels = [("raw", "raw")]
    for m in _BIN_CANDIDATES:
        try:
            edges, _ = _bin_fit_one(v, m, n_bins)
            lab = [f"B{i + 1}" for i in range(len(edges) - 1)]
            binned = _bin_apply_one(num, edges, lab)
            panels.append((m, (binned, lab)))
        except Exception:
            panels.append((m, None))
    ncols = 2
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width, fig_height * nrows), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax, (label, data) in zip(axes, panels):
        if label == "raw":
            ax.hist(v, bins=_hist_bins(len(v)),
                    color="#4c72b0", edgecolor="black", alpha=0.85)
            ax.set_title(f"raw '{c}'  (continuous)", fontweight="bold")
            ax.set_ylabel("count")
            continue
        if data is None:
            ax.text(0.5, 0.5, f"{label}\n(not valid for\nthis column)",
                    ha="center", va="center", fontsize=11)
            ax.set_axis_off()
            continue
        binned, lab = data
        counts = binned.value_counts().reindex(lab, fill_value=0)
        ax.bar([str(x) for x in lab], counts.to_numpy(),
               color="#2ca02c", edgecolor="black", alpha=0.85)
        ax.set_title(f"{label}  (n_bins={len(lab)})", fontweight="bold")
        ax.set_ylabel("count")
        ax.tick_params(axis="x", rotation=45)
    for ax in axes[len(panels):]:
        ax.set_axis_off()
    fig.suptitle(f"Bin COMPARE -- column '{c}'  (Stage 4.2)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_encode(df_before, plot_items, method, fig_width, fig_height, dpi):
    """Per column: category counts, then the encoded value per category."""
    items = plot_items[:4]
    n = len(items)
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 2, figsize=(fig_width, fig_height * n), dpi=dpi)
    axes = np.atleast_2d(axes)
    for i, (src, entry) in enumerate(items):
        col = df_before[src]
        keys = col[col.notna()].astype(str)
        vc = keys.value_counts()
        top = vc.head(20)
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.bar([str(x) for x in top.index], top.to_numpy(),
                color="#ec7853", edgecolor="black", alpha=0.85)
        ax0.set_title(f"'{src}' category counts  "
                      f"(top {len(top)} of {len(vc)})", fontweight="bold")
        ax0.set_ylabel("count")
        ax0.tick_params(axis="x", rotation=60)
        m = entry.get("method", method)
        if m == "onehot":
            share = top / top.sum()
            ax1.bar([str(x) for x in share.index], share.to_numpy(),
                    color="#4c72b0", edgecolor="black", alpha=0.85)
            ax1.set_title(f"one-hot -> {len(entry['new_cols'])} binary "
                          f"column(s)", fontweight="bold")
            ax1.set_ylabel("share")
        else:
            mapping = entry.get("mapping", {})
            mp = pd.Series(mapping).reindex([str(x) for x in top.index])
            mp = mp.dropna()
            label = {"ordinal": "rank", "frequency": "proportion",
                     "target": "mean target"}.get(m, "encoded value")
            ax1.bar([str(x) for x in mp.index], mp.to_numpy(),
                    color="#2ca02c", edgecolor="black", alpha=0.85)
            ax1.set_title(f"{m} encoding  ({label} per category)",
                          fontweight="bold")
            ax1.set_ylabel(label)
        ax1.tick_params(axis="x", rotation=60)
    fig.suptitle(f"Categorical encoding -- {method}  (Stage 4.2)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_encode_compare(df, cols, fig_width, fig_height, dpi):
    c = cols[0]
    col = df[c]
    keys = col[col.notna()].astype(str)
    vc = keys.value_counts()
    top = vc.head(25)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height * 1.5), dpi=dpi)
    ax.bar([str(x) for x in top.index], top.to_numpy(),
           color="#4c72b0", edgecolor="black", alpha=0.85)
    ax.set_title(f"'{c}' -- {len(vc)} categories  (one-hot would add "
                 f"{len(vc)} columns; ordinal / target / frequency add 1 "
                 f"each)", fontweight="bold")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=60)
    fig.suptitle(f"Encode COMPARE -- column '{c}'  (Stage 4.2)",
                 fontsize=14, fontweight="bold")
    return fig


# ===========================================================================
# 5. dtfeats  --  temporal features (calendar + cyclical sin/cos)
# ===========================================================================

_VALID_DTFEATS_METHODS = ("calendar", "cyclical", "both", "compare")
_DT_CALENDAR_ALL = ("year", "month", "day", "dayofweek", "dayofyear",
                    "quarter", "weekofyear", "hour", "minute", "second",
                    "is_weekend", "is_month_start", "is_month_end",
                    "is_quarter_start", "is_quarter_end",
                    "is_year_start", "is_year_end")
_DT_CALENDAR_DEFAULT = ("year", "month", "day", "dayofweek", "dayofyear",
                        "quarter", "hour", "is_weekend",
                        "is_month_start", "is_month_end")
_DT_CYCLICAL_PERIODS = {"month": 12.0, "dayofweek": 7.0, "hour": 24.0,
                        "dayofyear": 365.0, "quarter": 4.0, "minute": 60.0,
                        "day": 31.0, "second": 60.0}
_DT_CYCLICAL_DEFAULT = ("month", "dayofweek", "hour")


def _resolve_dt_cols(df, cols, func_name):
    """Validate explicit cols or auto-pick datetime columns."""
    if cols is None:
        chosen = [c for c in df.columns
                  if pd.api.types.is_datetime64_any_dtype(df[c])]
        if not chosen:
            raise ValueError(
                f"{func_name}: no datetime columns found. Pass cols= "
                f"explicitly (they will be parsed with pd.to_datetime).")
        return chosen
    chosen = list(cols)
    bad = [c for c in chosen if c not in df.columns]
    if bad:
        raise KeyError(f"{func_name}: cols references columns not in df: {bad}")
    return chosen


def _as_datetime(s, c):
    """Coerce a Series to datetime64 or raise a clear error."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    try:
        return pd.to_datetime(s, errors="raise")
    except Exception as exc:
        raise TypeError(
            f"dtfeats: column '{c}' is not datetime and could not be parsed "
            f"as one ({exc}). Convert it with pd.to_datetime first.") from exc


def _dt_component(s, name):
    """Raw calendar component Series for a datetime Series."""
    dt = s.dt
    table = {
        "year": lambda: dt.year, "month": lambda: dt.month,
        "day": lambda: dt.day, "dayofweek": lambda: dt.dayofweek,
        "dayofyear": lambda: dt.dayofyear, "quarter": lambda: dt.quarter,
        "weekofyear": lambda: dt.isocalendar().week,
        "hour": lambda: dt.hour, "minute": lambda: dt.minute,
        "second": lambda: dt.second,
        "is_weekend": lambda: (dt.dayofweek >= 5),
        "is_month_start": lambda: dt.is_month_start,
        "is_month_end": lambda: dt.is_month_end,
        "is_quarter_start": lambda: dt.is_quarter_start,
        "is_quarter_end": lambda: dt.is_quarter_end,
        "is_year_start": lambda: dt.is_year_start,
        "is_year_end": lambda: dt.is_year_end,
    }
    if name not in table:
        raise ValueError(f"dtfeats: unknown calendar feature {name!r}")
    return table[name]()


def _dt_numeric(s, name):
    """Calendar component as float64 with NaN where the source is NaT."""
    comp = pd.Series(_dt_component(s, name), index=s.index)
    out = pd.to_numeric(comp, errors="coerce").astype("float64")
    return out.mask(s.isna())


def _dtfeats_extract(s, calendar_feats, cyclical_feats, col):
    """Return an ordered {new_col_name: Series} dict for one datetime column."""
    new = {}
    for f in calendar_feats:
        new[f"{col}_{f}"] = _dt_numeric(s, f)
    for comp in cyclical_feats:
        period = _DT_CYCLICAL_PERIODS[comp]
        ang = 2.0 * np.pi * _dt_numeric(s, comp) / period
        new[f"{col}_{comp}_sin"] = np.sin(ang)
        new[f"{col}_{comp}_cos"] = np.cos(ang)
    return new


def dtfeats(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "both",
    *,
    features: Optional[Sequence[str]] = None,
    drop_original: bool = False,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.2,
    dpi: int = 110,
):
    """Extract calendar and cyclical features from datetime columns.

    Two modes. In FIT mode (``params=None``) the recipe (which features to
    extract) is recorded. In APPLY mode (``params`` supplied) the exact same
    feature set is re-created on new data -- guaranteeing identical columns
    between train and test.

    This function learns no statistics; extraction is deterministic. The
    ``params`` dict is a reproducible recipe, not a fitted transformer.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    cols : sequence of str, optional
        Datetime columns. If ``None`` all datetime columns are used; explicit
        columns are parsed with ``pd.to_datetime``.
    method : {'calendar', 'cyclical', 'both', 'compare'}
        'calendar' -> integer parts (year, month, dayofweek, is_weekend ...).
        'cyclical' -> sin/cos pairs so the model sees December next to January.
        'both'     -> calendar + cyclical.
        'compare'  -> writes nothing; reports how many features each produces.
    features : sequence of str, optional
        Subset of calendar features to extract. Default is a 10-feature set.
    drop_original : bool, default False
        If True the source datetime column is dropped after extraction.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the recipe dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        And, when requested, the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.dtfeats(df_train, cols=['signup'], method='both',
    ...                       return_params=True)
    >>> df_te = dx.dtfeats(df_test, params=p)            # apply, same columns
    >>> dx.dtfeats(df, cols=['signup'], method='compare')
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    if params is not None:
        return _dtfeats_apply(df, params, show, plot, return_df,
                              return_params, return_fig, decimals, df_name,
                              fig_width, fig_height, dpi)

    if method not in _VALID_DTFEATS_METHODS:
        raise ValueError(
            f"'method' must be one of {_VALID_DTFEATS_METHODS}, got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "method='compare' does not produce a recipe; call dtfeats with a "
            "concrete method (calendar/cyclical/both) to obtain params.")

    cols = _resolve_dt_cols(df, cols, "dtfeats")

    if features is not None:
        cal_all = list(features)
        bad = [f for f in cal_all if f not in _DT_CALENDAR_ALL]
        if bad:
            raise ValueError(
                f"dtfeats: unknown calendar feature(s) {bad}; valid options "
                f"are {_DT_CALENDAR_ALL}.")
    else:
        cal_all = list(_DT_CALENDAR_DEFAULT)
    cyc_all = list(_DT_CYCLICAL_DEFAULT)

    if method == "compare":
        return _dtfeats_compare(df, cols, cal_all, cyc_all, show, plot,
                                return_df, return_params, return_fig,
                                decimals, df_name, fig_width, fig_height, dpi)

    use_cal = cal_all if method in ("calendar", "both") else []
    use_cyc = cyc_all if method in ("cyclical", "both") else []

    out = df.copy()
    out.attrs = dict(df.attrs)
    col_params, rows, plot_items = {}, [], []
    for c in cols:
        s = _as_datetime(df[c], c)
        new = _dtfeats_extract(s, use_cal, use_cyc, c)
        for name, series in new.items():
            out[name] = series
        if drop_original and c in out.columns:
            out = out.drop(columns=[c])
        col_params[c] = {"calendar": list(use_cal), "cyclical": list(use_cyc),
                         "source": c, "new_cols": list(new.keys())}
        plot_items.append((c, s, list(new.keys())))
        rows.append({
            "source": c,
            "n_calendar": len(use_cal),
            "n_cyclical_pairs": len(use_cyc),
            "n_new_cols": len(new),
            "n_nat": int(s.isna().sum()),
        })

    params_out = {
        "function": "dtfeats",
        "method": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "columns": col_params,
        "metadata": {"drop_original": bool(drop_original), "n_cols": len(cols)},
    }

    total_new = sum(r["n_new_cols"] for r in rows)
    decision = (f"Extracted {total_new} temporal feature(s) from {len(cols)} "
                f"datetime column(s) using method '{method}'. Cyclical "
                f"sin/cos pairs keep periodic distance meaningful.")

    _append_audit(out, {
        "stage": "feature_datetime",
        "function": "dtfeats",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "cols": list(cols),
                   "drop_original": bool(drop_original)},
        "decision": decision,
    })

    summary = pd.DataFrame(rows).set_index("source")
    if show:
        _print_header(f"Datetime features for: {df_name}  "
                      f"(method={method}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_dtfeats(out, plot_items, method,
                            fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _dtfeats_apply(df, params, show, plot, return_df, return_params,
                   return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "dtfeats":
        got = params.get("function") if isinstance(params, dict) else params
        raise ValueError(f"params dict is not for 'dtfeats' (function={got!r}).")
    method = params["method"]
    col_params = params["columns"]
    drop_original = bool(params.get("metadata", {}).get("drop_original", False))
    missing = [src for src in col_params if src not in df.columns]
    if missing:
        raise KeyError(
            f"dtfeats apply failed: params expects column(s) {missing} which "
            f"are not present in this DataFrame.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    rows, plot_items = [], []
    for src, cp in col_params.items():
        s = _as_datetime(df[src], src)
        new = _dtfeats_extract(s, cp.get("calendar", []),
                               cp.get("cyclical", []), src)
        for name, series in new.items():
            out[name] = series
        if drop_original and src in out.columns:
            out = out.drop(columns=[src])
        plot_items.append((src, s, list(new.keys())))
        rows.append({
            "source": src,
            "n_calendar": len(cp.get("calendar", [])),
            "n_cyclical_pairs": len(cp.get("cyclical", [])),
            "n_new_cols": len(new),
            "n_nat": int(s.isna().sum()),
        })

    decision = (f"Applied saved 'dtfeats' recipe (fitted "
                f"{params.get('fit_at', '?')}) to {len(col_params)} "
                f"column(s); identical feature set to the fit.")
    _append_audit(out, {
        "stage": "feature_datetime",
        "function": "dtfeats",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"method": method, "cols": list(col_params),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    summary = pd.DataFrame(rows).set_index("source")
    if show:
        _print_header(f"Datetime features for: {df_name}  "
                      f"(method={method}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_dtfeats(out, plot_items, method,
                            fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _dtfeats_compare(df, cols, cal_all, cyc_all, show, plot, return_df,
                     return_params, return_fig, decimals, df_name,
                     fig_width, fig_height, dpi):
    rows = []
    for c in cols:
        rows.append({
            "source": c,
            "calendar_cols": len(cal_all),
            "cyclical_cols": 2 * len(cyc_all),
            "both_cols": len(cal_all) + 2 * len(cyc_all),
        })
    summary = pd.DataFrame(rows).set_index("source")
    decision = (f"Compared dtfeats methods on {len(cols)} datetime column(s). "
                f"'calendar' adds integer parts; 'cyclical' adds sin/cos "
                f"pairs; 'both' adds all. No columns written -- pick a method "
                f"then call dtfeats(method=...).")
    if show:
        _print_header(f"dtfeats COMPARE for: {df_name}  (nothing written)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_dtfeats_compare(df, cols, cal_all, cyc_all,
                                    fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# 6. cross  --  interaction features (ratio / product / diff / polynomial)
# ===========================================================================

_VALID_CROSS_METHODS = ("ratio", "product", "diff", "polynomial", "compare")
_CROSS_PAIR_METHODS = ("ratio", "product", "diff")
_CROSS_OP_NAMING = {"ratio": "div", "product": "x", "diff": "minus"}


def _resolve_cross_pairs(df, pairs, cols, method, func_name):
    """Resolve the (a, b) pairs or the single-column list for an interaction.

    Returns
    -------
    (pair_list, poly_cols)
        For pair methods ``pair_list`` is a list of (a, b) tuples and
        ``poly_cols`` is None. For 'polynomial' it is the reverse.
    """
    from itertools import combinations
    if method == "polynomial":
        chosen = _resolve_cols(df, cols, func_name)
        if len(chosen) < 1:
            raise ValueError(f"{func_name}: polynomial needs >= 1 column.")
        return None, chosen
    if pairs is not None:
        pair_list = [tuple(p) for p in pairs]
        for p in pair_list:
            if len(p) != 2:
                raise ValueError(
                    f"{func_name}: each pair must have exactly 2 columns, "
                    f"got {p!r}.")
            for x in p:
                if x not in df.columns:
                    raise KeyError(
                        f"{func_name}: pair references a missing column "
                        f"{x!r}.")
                if not pd.api.types.is_numeric_dtype(df[x]):
                    raise TypeError(
                        f"{func_name}: column {x!r} is not numeric.")
        return pair_list, None
    chosen = _resolve_cols(df, cols, func_name)
    if len(chosen) < 2:
        raise ValueError(
            f"{func_name}: need >= 2 numeric columns (or pass pairs=) to "
            f"form interactions.")
    return list(combinations(chosen, 2)), None


def _cross_compute(df, op, inputs, power=None):
    """Compute one interaction Series from a recipe."""
    if op == "ratio":
        a, b = inputs
        num = pd.to_numeric(df[a], errors="coerce")
        den = pd.to_numeric(df[b], errors="coerce").replace(0, np.nan)
        return num / den
    if op == "product":
        a, b = inputs
        return (pd.to_numeric(df[a], errors="coerce")
                * pd.to_numeric(df[b], errors="coerce"))
    if op == "diff":
        a, b = inputs
        return (pd.to_numeric(df[a], errors="coerce")
                - pd.to_numeric(df[b], errors="coerce"))
    if op == "power":
        (a,) = inputs
        return pd.to_numeric(df[a], errors="coerce") ** power
    raise ValueError(f"cross: unknown op {op!r}")


def _cross_recipe(method, pair_list, poly_cols, degree):
    """Build the ordered {new_col: recipe} dict for a cross method."""
    from itertools import combinations
    recipe = {}
    if method in _CROSS_PAIR_METHODS:
        tag = _CROSS_OP_NAMING[method]
        for a, b in pair_list:
            recipe[f"{a}_{tag}_{b}"] = {"op": method, "inputs": [a, b],
                                        "power": None}
    else:  # polynomial
        for c in poly_cols:
            for k in range(2, degree + 1):
                recipe[f"{c}_pow{k}"] = {"op": "power", "inputs": [c],
                                         "power": k}
        for a, b in combinations(poly_cols, 2):
            recipe[f"{a}_x_{b}"] = {"op": "product", "inputs": [a, b],
                                    "power": None}
    return recipe


def cross(
    df: pd.DataFrame,
    pairs: Optional[Sequence] = None,
    method: str = "product",
    *,
    cols: Optional[Sequence[str]] = None,
    degree: int = 2,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.2,
    dpi: int = 110,
):
    """Build interaction features by crossing numeric columns.

    Two modes. In FIT mode (``params=None``) the recipe (which columns to
    cross and how) is recorded. In APPLY mode (``params`` supplied) the exact
    same interactions are re-created on new data.

    This function learns no statistics; interactions are deterministic
    arithmetic. The ``params`` dict is a reproducible recipe.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    pairs : sequence of (str, str), optional
        Explicit column pairs to cross. If ``None`` all pairwise combinations
        of ``cols`` are used (not for 'polynomial').
    method : {'ratio', 'product', 'diff', 'polynomial', 'compare'}
        'ratio'      -> a / b  (division by zero becomes NaN, never Inf).
        'product'    -> a * b.
        'diff'       -> a - b.
        'polynomial' -> powers (x**2 .. x**degree) plus pairwise products.
        'compare'    -> writes nothing; reports the spread of each candidate.
    cols : sequence of str, optional
        Numeric columns used when ``pairs`` is None / for 'polynomial'.
    degree : int, default 2
        Highest power for 'polynomial'.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the recipe dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        And, when requested, the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.cross(df_train, pairs=[('price', 'area')],
    ...                     method='ratio', return_params=True)
    >>> df_te = dx.cross(df_test, params=p)              # apply, same recipe
    >>> dx.cross(df, cols=['price', 'area'], method='compare')
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    if params is not None:
        return _cross_apply(df, params, show, plot, return_df,
                            return_params, return_fig, decimals, df_name,
                            fig_width, fig_height, dpi)

    if method not in _VALID_CROSS_METHODS:
        raise ValueError(
            f"'method' must be one of {_VALID_CROSS_METHODS}, got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "method='compare' does not produce a recipe; call cross with a "
            "concrete method (ratio/product/diff/polynomial) to get params.")
    if method == "polynomial" and (not isinstance(degree, (int, np.integer))
                                   or bool(degree < 2)):
        raise ValueError(f"'degree' must be an integer >= 2, got {degree!r}")

    if method == "compare":
        return _cross_compare(df, pairs, cols, show, plot, return_df,
                              return_params, return_fig, decimals, df_name,
                              fig_width, fig_height, dpi)

    pair_list, poly_cols = _resolve_cross_pairs(df, pairs, cols, method,
                                                "cross")
    recipe = _cross_recipe(method, pair_list, poly_cols, int(degree))
    if not recipe:
        raise ValueError("cross: no interaction terms were produced.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    rows = []
    for new_col, rc in recipe.items():
        series = _cross_compute(df, rc["op"], rc["inputs"], rc["power"])
        out[new_col] = series
        n_inf = int(np.isinf(series).sum())
        rows.append({
            "new_col": new_col,
            "inputs": " , ".join(rc["inputs"]),
            "std": float(series.std(ddof=0)),
            "n_nan": int(series.isna().sum()),
            "n_inf": n_inf,
        })
        if n_inf:
            warnings.warn(f"cross: column '{new_col}' has {n_inf} infinite "
                          f"value(s).")

    params_out = {
        "function": "cross",
        "method": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "columns": recipe,
        "metadata": {"degree": int(degree) if method == "polynomial" else None,
                     "n_terms": len(recipe)},
    }

    decision = (f"Created {len(recipe)} interaction feature(s) with method "
                f"'{method}'. Interactions expose relationships a linear "
                f"model cannot see from the raw columns alone.")

    _append_audit(out, {
        "stage": "feature_interaction",
        "function": "cross",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "n_terms": len(recipe)},
        "decision": decision,
    })

    summary = pd.DataFrame(rows).set_index("new_col")
    if show:
        _print_header(f"Interaction features for: {df_name}  "
                      f"(method={method}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_cross(out, list(recipe.keys()), method,
                          fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _cross_apply(df, params, show, plot, return_df, return_params,
                 return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "cross":
        got = params.get("function") if isinstance(params, dict) else params
        raise ValueError(f"params dict is not for 'cross' (function={got!r}).")
    method = params["method"]
    recipe = params["columns"]
    needed = sorted({x for rc in recipe.values() for x in rc["inputs"]})
    missing = [x for x in needed if x not in df.columns]
    if missing:
        raise KeyError(
            f"cross apply failed: the recipe needs input column(s) {missing} "
            f"which are not present in this DataFrame.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    rows = []
    for new_col, rc in recipe.items():
        series = _cross_compute(df, rc["op"], rc["inputs"], rc.get("power"))
        out[new_col] = series
        n_inf = int(np.isinf(series).sum())
        rows.append({
            "new_col": new_col,
            "inputs": " , ".join(rc["inputs"]),
            "std": float(series.std(ddof=0)),
            "n_nan": int(series.isna().sum()),
            "n_inf": n_inf,
        })
        if n_inf:
            warnings.warn(f"cross: column '{new_col}' has {n_inf} infinite "
                          f"value(s).")

    decision = (f"Applied saved 'cross' recipe (fitted "
                f"{params.get('fit_at', '?')}) -- recreated {len(recipe)} "
                f"interaction feature(s); identical to the fit.")
    _append_audit(out, {
        "stage": "feature_interaction",
        "function": "cross",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"method": method, "n_terms": len(recipe),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    summary = pd.DataFrame(rows).set_index("new_col")
    if show:
        _print_header(f"Interaction features for: {df_name}  "
                      f"(method={method}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_cross(out, list(recipe.keys()), method,
                          fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _cross_compare(df, pairs, cols, show, plot, return_df, return_params,
                   return_fig, decimals, df_name, fig_width, fig_height, dpi):
    pair_list, _ = _resolve_cross_pairs(df, pairs, cols, "ratio", "cross")
    rows, index = [], []
    for a, b in pair_list:
        for m in _CROSS_PAIR_METHODS:
            series = _cross_compute(df, m, [a, b])
            index.append((f"{a} , {b}", m))
            rows.append({
                "std": float(series.std(ddof=0)),
                "n_nan": int(series.isna().sum()),
                "n_inf": int(np.isinf(series).sum()),
            })
    summary = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        index, names=["pair", "method"]))
    decision = (f"Compared {len(_CROSS_PAIR_METHODS)} interaction method(s) "
                f"on {len(pair_list)} pair(s). 'std' shows how much each "
                f"interaction varies (a flat column carries no signal). No "
                f"columns written -- pick a method then call cross(method=...).")
    if show:
        _print_header(f"cross COMPARE for: {df_name}  (nothing written)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_cross_compare(df, pair_list, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# 7. aggfeat  --  aggregation features (groupby, with an as_of leakage guard)
# ===========================================================================

_VALID_AGG_METHODS = ("mean", "median", "sum", "std", "min", "max",
                      "count", "nunique", "compare")
_AGG_CANDIDATES = ("mean", "median", "sum", "std", "min", "max",
                   "count", "nunique")
_AGG_NUMERIC = ("mean", "median", "sum", "std", "min", "max")


def _as_list(x):
    """Normalise a str / sequence / None argument to a list (or None)."""
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def _group_key_series(df, group):
    """A single string key per row built from one or more group columns."""
    parts = [df[g].astype(str) for g in group]
    key = parts[0].copy()
    for p in parts[1:]:
        key = key.str.cat(p, sep=" | ")
    return key


def _agg_compute(values, name):
    """Aggregate an array-like to one scalar (NaN-safe)."""
    s = values if isinstance(values, pd.Series) else pd.Series(list(values))
    if name == "count":
        return float(s.notna().sum())
    if name == "nunique":
        return float(s.dropna().nunique())
    num = pd.to_numeric(s, errors="coerce").dropna()
    if num.empty:
        return float("nan")
    funcs = {"mean": num.mean, "median": num.median, "sum": num.sum,
             "std": lambda: num.std(ddof=0), "min": num.min, "max": num.max}
    if name not in funcs:
        raise ValueError(f"aggfeat: unknown agg {name!r}")
    return float(funcs[name]())


def _iso_or_num(x, is_dt):
    """JSON-safe representation of an as_of value."""
    if is_dt:
        return pd.Timestamp(x).isoformat()
    return float(x)


def _json_val(v):
    """JSON-safe representation of a value-column entry."""
    if pd.isna(v):
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    return str(v)


def _aggfeat_validate(df, group, value, as_of, agg):
    """Shared argument checks for aggfeat fit."""
    if not group:
        raise ValueError("aggfeat: 'group' is required (the column(s) to "
                          "group by).")
    if not value:
        raise ValueError("aggfeat: 'value' is required (the column(s) to "
                          "aggregate).")
    cols_needed = list(group) + list(value) + ([as_of] if as_of else [])
    bad = [c for c in cols_needed if c not in df.columns]
    if bad:
        raise KeyError(f"aggfeat: column(s) not in df: {bad}")
    if agg in _AGG_NUMERIC:
        non_num = [v for v in value
                   if not pd.api.types.is_numeric_dtype(df[v])]
        if non_num:
            raise TypeError(
                f"aggfeat: agg='{agg}' needs numeric value column(s); "
                f"non-numeric passed: {non_num}.")


def aggfeat(
    df: pd.DataFrame,
    group=None,
    value=None,
    agg: str = "mean",
    *,
    as_of: Optional[str] = None,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.2,
    dpi: int = 110,
):
    """Build group-aggregation features, with a temporal-leakage guard.

    Two modes. In FIT mode (``params=None``) the per-group statistics are
    learned from ``df``. In APPLY mode (``params`` supplied) they are applied
    verbatim with no re-fitting.

    Two aggregation modes:

    * STATIC (``as_of=None``) -- one statistic per group computed over all
      rows, joined back to every row. Leakage-safe across a train/test split
      because the statistics are learned on the training data only.
    * AS-OF (``as_of`` given) -- an expanding window: each row sees only rows
      strictly earlier in the ``as_of`` column. This refuses the 'lag feature
      that peeks at the future' anti-pattern in FEATURES_PHILOSOPHY.md.

    When ``as_of`` is omitted but the data contains datetime columns, a
    warning is emitted (the rows may be time-ordered).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    group : str or sequence of str
        Column(s) to group by.
    value : str or sequence of str
        Numeric column(s) to aggregate (any dtype for count / nunique).
    agg : {'mean','median','sum','std','min','max','count','nunique','compare'}
        The aggregation. 'compare' writes nothing and reports every option.
    as_of : str, optional
        A datetime or numeric column. When given, switches to the expanding
        (past-only) aggregation that prevents temporal leakage.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the learned params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        And, when requested, the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.aggfeat(df_train, group='city', value='price',
    ...                       agg='mean', return_params=True)
    >>> df_te = dx.aggfeat(df_test, params=p)            # apply, no re-fit
    >>> dx.aggfeat(df, group='city', value='price', agg='compare')
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    if params is not None:
        return _aggfeat_apply(df, params, show, plot, return_df,
                              return_params, return_fig, decimals, df_name,
                              fig_width, fig_height, dpi)

    if agg not in _VALID_AGG_METHODS:
        raise ValueError(
            f"'agg' must be one of {_VALID_AGG_METHODS}, got {agg!r}")
    if agg == "compare" and return_params:
        raise ValueError(
            "agg='compare' does not fit parameters; call aggfeat with a "
            "concrete agg (mean/median/sum/...) to obtain a params dict.")

    group = _as_list(group)
    value = _as_list(value)
    _aggfeat_validate(df, group, value, as_of, agg)

    if agg == "compare":
        return _aggfeat_compare(df, group, value, show, plot, return_df,
                                return_params, return_fig, decimals, df_name,
                                fig_width, fig_height, dpi)

    gk = _group_key_series(df, group)
    out = df.copy()
    out.attrs = dict(df.attrs)
    col_params, rows, plot_items = {}, [], []
    gtag = "_".join(group)

    if as_of is None:
        dt_cols = [c for c in df.columns
                   if pd.api.types.is_datetime64_any_dtype(df[c])]
        if dt_cols:
            warnings.warn(
                f"aggfeat: df has datetime column(s) {dt_cols} but as_of= was "
                f"not given. If the rows are time-ordered, pass as_of= for an "
                f"expanding (past-only) aggregation to prevent temporal "
                f"leakage. Proceeding with a static aggregation over all rows.")
        for v in value:
            stat = df[v].groupby(gk).apply(lambda s: _agg_compute(s, agg))
            mapping = {str(k): float(x) for k, x in stat.items()
                       if pd.notna(x)}
            default = _agg_compute(df[v], agg)
            new_col = f"{v}_{agg}_by_{gtag}"
            feature = gk.map(mapping).astype("float64")
            out[new_col] = feature
            col_params[new_col] = {
                "group": list(group), "value": v, "agg": agg,
                "mode": "static", "mapping": mapping,
                "default": float(default) if pd.notna(default) else None,
                "source": v, "new_col": new_col}
            plot_items.append((new_col, gk, feature))
            rows.append({
                "new_col": new_col, "mode": "static",
                "n_groups": len(mapping),
                "global_stat": float(default),
                "n_nan": int(feature.isna().sum())})
        agg_mode = "static"
    else:
        t_raw = df[as_of]
        is_dt = pd.api.types.is_datetime64_any_dtype(t_raw)
        if not is_dt and not pd.api.types.is_numeric_dtype(t_raw):
            try:
                t_raw = pd.to_datetime(t_raw, errors="raise")
                is_dt = True
            except Exception as exc:
                raise TypeError(
                    f"aggfeat: as_of column '{as_of}' must be datetime or "
                    f"numeric ({exc}).") from exc
        for v in value:
            feature = pd.Series(np.nan, index=df.index, dtype="float64")
            history = {}
            for key, idx in gk.groupby(gk).groups.items():
                sub_t = t_raw.loc[idx]
                sub_v = df[v].loc[idx]
                order = sub_t.sort_values(kind="mergesort").index
                st = sub_t.loc[order].to_numpy()
                sv = sub_v.loc[order].to_numpy()
                for pos in range(len(order)):
                    cutoff = int(np.searchsorted(st, st[pos], side="left"))
                    if cutoff > 0:
                        feature.loc[order[pos]] = _agg_compute(
                            sv[:cutoff], agg)
                history[str(key)] = [[_iso_or_num(st[j], is_dt),
                                      _json_val(sv[j])]
                                     for j in range(len(st))]
            default = _agg_compute(df[v], agg)
            new_col = f"{v}_{agg}_by_{gtag}"
            out[new_col] = feature
            col_params[new_col] = {
                "group": list(group), "value": v, "agg": agg,
                "mode": "as_of", "as_of": as_of,
                "as_of_kind": "datetime" if is_dt else "numeric",
                "history": history,
                "default": float(default) if pd.notna(default) else None,
                "source": v, "new_col": new_col}
            plot_items.append((new_col, gk, feature))
            rows.append({
                "new_col": new_col, "mode": "as_of",
                "n_groups": len(history),
                "global_stat": float(default),
                "n_nan": int(feature.isna().sum())})
        agg_mode = "as_of"

    params_out = {
        "function": "aggfeat",
        "method": agg,
        "version": __version__,
        "fit_at": _now_iso(),
        "columns": col_params,
        "metadata": {"as_of": as_of, "agg_mode": agg_mode,
                     "n_cols": len(value)},
    }

    guard = ("expanding (past-only) aggregation -- temporal-leak-safe"
             if as_of is not None
             else "static aggregation -- leakage-safe across a train/test "
                  "split")
    decision = (f"Built {len(value)} aggregation feature(s): {agg} of "
                f"{value} grouped by {group}. Used {guard}.")

    _append_audit(out, {
        "stage": "feature_aggregation",
        "function": "aggfeat",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"agg": agg, "group": list(group), "value": list(value),
                   "as_of": as_of},
        "decision": decision,
    })

    summary = pd.DataFrame(rows).set_index("new_col")
    if show:
        _print_header(f"Aggregation features for: {df_name}  "
                      f"(agg={agg}, mode=fit/{agg_mode})")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_aggfeat(df, plot_items, agg, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _aggfeat_apply(df, params, show, plot, return_df, return_params,
                   return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "aggfeat":
        got = params.get("function") if isinstance(params, dict) else params
        raise ValueError(f"params dict is not for 'aggfeat' (function={got!r}).")
    agg = params["method"]
    col_params = params["columns"]
    # validate every required source column is present
    need = set()
    for cp in col_params.values():
        need.update(cp["group"])
        if cp["mode"] == "as_of":
            need.add(cp["as_of"])
    missing = sorted(c for c in need if c not in df.columns)
    if missing:
        raise KeyError(
            f"aggfeat apply failed: params expects column(s) {missing} which "
            f"are not present in this DataFrame.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    rows, plot_items = [], []
    for new_col, cp in col_params.items():
        gk = _group_key_series(df, cp["group"])
        default = cp.get("default")
        if cp["mode"] == "static":
            mapping = cp["mapping"]
            feature = gk.map(mapping)
            if default is not None:
                feature = feature.where(gk.isin(mapping), other=default)
            feature = feature.astype("float64")
            n_unknown = int((~gk.isin(mapping)).sum())
        else:  # as_of
            is_dt = cp.get("as_of_kind") == "datetime"
            history = cp["history"]
            t_raw = df[cp["as_of"]]
            if is_dt:
                t_vals = pd.to_datetime(t_raw, errors="coerce")
            else:
                t_vals = pd.to_numeric(t_raw, errors="coerce")
            # pre-parse history per group (pure-Python compare, leak-safe)
            parsed = {}
            for k, recs in history.items():
                if is_dt:
                    ths = [pd.Timestamp(r[0]) for r in recs]
                else:
                    ths = [float(r[0]) for r in recs]
                parsed[k] = (ths, [r[1] for r in recs])
            feature = pd.Series(np.nan, index=df.index, dtype="float64")
            n_unknown = 0
            gk_arr = gk.to_numpy()
            for pos in range(len(df)):
                key = gk_arr[pos]
                if key not in parsed:
                    n_unknown += 1
                    if default is not None:
                        feature.iloc[pos] = default
                    continue
                ths, tvs = parsed[key]
                row_t = t_vals.iloc[pos]
                if pd.isna(row_t):
                    continue
                past = [tvs[j] for j in range(len(ths)) if ths[j] < row_t]
                if past:
                    feature.iloc[pos] = _agg_compute(past, agg)
                elif default is not None:
                    feature.iloc[pos] = default
        out[new_col] = feature
        plot_items.append((new_col, gk, feature))
        rows.append({
            "new_col": new_col, "mode": cp["mode"],
            "n_groups": len(cp.get("mapping", cp.get("history", {}))),
            "n_unknown_groups": n_unknown,
            "n_nan": int(feature.isna().sum())})

    decision = (f"Applied saved 'aggfeat' params (fitted "
                f"{params.get('fit_at', '?')}) -- recreated {len(col_params)} "
                f"aggregation feature(s); no re-fit -- leakage-safe.")
    _append_audit(out, {
        "stage": "feature_aggregation",
        "function": "aggfeat",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"agg": agg, "cols": list(col_params),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    summary = pd.DataFrame(rows).set_index("new_col")
    if show:
        _print_header(f"Aggregation features for: {df_name}  "
                      f"(agg={agg}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_aggfeat(df, plot_items, agg, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _aggfeat_compare(df, group, value, show, plot, return_df, return_params,
                     return_fig, decimals, df_name, fig_width, fig_height, dpi):
    gk = _group_key_series(df, group)
    rows, index = [], []
    for v in value:
        v_numeric = pd.api.types.is_numeric_dtype(df[v])
        for a in _AGG_CANDIDATES:
            if a in _AGG_NUMERIC and not v_numeric:
                index.append((v, a))
                rows.append({"global_stat": np.nan, "n_groups": np.nan,
                             "group_spread": np.nan})
                continue
            stat = df[v].groupby(gk).apply(lambda s, a=a: _agg_compute(s, a))
            index.append((v, a))
            rows.append({
                "global_stat": _agg_compute(df[v], a),
                "n_groups": int(stat.notna().sum()),
                "group_spread": float(stat.std(ddof=0)),
            })
    summary = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(
        index, names=["value", "agg"]))
    decision = (f"Compared {len(_AGG_CANDIDATES)} aggregation(s) on "
                f"{len(value)} value column(s) grouped by {group}. "
                f"'group_spread' is the std of the per-group statistic -- a "
                f"larger spread means a more discriminative feature. No "
                f"columns written -- pick an agg then call aggfeat(agg=...).")
    if show:
        _print_header(f"aggfeat COMPARE for: {df_name}  (nothing written)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_aggfeat_compare(df, group, value[0], gk,
                                    fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# Plotting helpers -- Stage 4.3 (dtfeats + cross + aggfeat)
# ===========================================================================

def _plot_dtfeats(out, plot_items, method, fig_width, fig_height, dpi):
    items = plot_items[:3]
    n = len(items)
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 2, figsize=(fig_width, fig_height * n), dpi=dpi)
    axes = np.atleast_2d(axes)
    for i, (col, s, names) in enumerate(items):
        ax0, ax1 = axes[i, 0], axes[i, 1]
        cal_name = next((nm for nm in names
                         if not nm.endswith(("_sin", "_cos"))), None)
        if cal_name is not None and cal_name in out.columns:
            vals = pd.to_numeric(out[cal_name], errors="coerce").dropna()
            vc = vals.value_counts().sort_index()
            ax0.bar([str(x) for x in vc.index], vc.to_numpy(),
                    color="#ec7853", edgecolor="black", alpha=0.85)
            ax0.set_title(f"'{cal_name}' counts", fontweight="bold")
        else:
            yr = pd.to_numeric(s.dt.year, errors="coerce").dropna()
            ax0.hist(yr, color="#ec7853", edgecolor="black", alpha=0.85)
            ax0.set_title(f"'{col}' year", fontweight="bold")
        ax0.set_ylabel("count")
        ax0.tick_params(axis="x", rotation=45)
        sin_name = next((nm for nm in names if nm.endswith("_sin")), None)
        cos_name = next((nm for nm in names if nm.endswith("_cos")), None)
        if sin_name and cos_name and sin_name in out.columns:
            ax1.scatter(out[cos_name], out[sin_name], s=12, color="#4c72b0",
                        alpha=0.5, edgecolor="black", linewidth=0.3)
            ax1.set_title(f"cyclical: {sin_name[:-4]}  (cos vs sin)",
                          fontweight="bold")
            ax1.set_xlabel("cos")
            ax1.set_ylabel("sin")
            ax1.set_aspect("equal", "box")
        else:
            other = names[-1] if names else None
            if other and other in out.columns:
                ov = pd.to_numeric(out[other], errors="coerce").dropna()
                ax1.hist(ov, color="#2ca02c", edgecolor="black", alpha=0.85)
                ax1.set_title(f"'{other}'", fontweight="bold")
                ax1.set_ylabel("count")
            else:
                ax1.set_axis_off()
    fig.suptitle(f"Datetime features -- {method}  (Stage 4.3)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_dtfeats_compare(df, cols, cal_all, cyc_all,
                          fig_width, fig_height, dpi):
    methods = ["calendar", "cyclical", "both"]
    counts = [len(cal_all), 2 * len(cyc_all), len(cal_all) + 2 * len(cyc_all)]
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height * 1.4), dpi=dpi)
    ax.bar(methods, counts, color=["#ec7853", "#4c72b0", "#2ca02c"],
           edgecolor="black", alpha=0.85)
    for i, v in enumerate(counts):
        ax.text(i, v, str(v), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("new columns")
    ax.set_title(f"dtfeats: columns produced per method  "
                 f"({len(cols)} datetime column(s))", fontweight="bold")
    fig.suptitle("dtfeats COMPARE  (Stage 4.3)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_cross(out, new_cols, method, fig_width, fig_height, dpi):
    names = list(new_cols)[:6]
    n = len(names)
    if n == 0:
        return None
    ncols = 3 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_width, fig_height * nrows), dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax, nm in zip(axes, names):
        v = pd.to_numeric(out[nm], errors="coerce")
        v = v[np.isfinite(v)]
        if len(v):
            ax.hist(v, bins=_hist_bins(len(v)),
                    color="#4c72b0", edgecolor="black", alpha=0.85)
        ax.set_title(f"'{nm}'", fontweight="bold")
        ax.set_ylabel("count")
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle(f"Interaction features -- {method}  (Stage 4.3)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_cross_compare(df, pair_list, fig_width, fig_height, dpi):
    a, b = pair_list[0]
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height * 1.2),
                             dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax, m in zip(axes, _CROSS_PAIR_METHODS):
        v = _cross_compute(df, m, [a, b])
        v = v[np.isfinite(v)]
        if len(v):
            ax.hist(v, bins=_hist_bins(len(v)),
                    color="#4c72b0", edgecolor="black", alpha=0.85)
        ax.set_title(f"{m}:  {a} , {b}", fontweight="bold")
        ax.set_ylabel("count")
    fig.suptitle(f"cross COMPARE -- pair '{a}' , '{b}'  (Stage 4.3)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_aggfeat(df, plot_items, agg, fig_width, fig_height, dpi):
    items = plot_items[:3]
    n = len(items)
    if n == 0:
        return None
    fig, axes = plt.subplots(n, 2, figsize=(fig_width, fig_height * n), dpi=dpi)
    axes = np.atleast_2d(axes)
    for i, (new_col, gk, feature) in enumerate(items):
        ax0, ax1 = axes[i, 0], axes[i, 1]
        per_group = (pd.Series(np.asarray(feature), index=np.asarray(gk))
                     .groupby(level=0).first().dropna()
                     .sort_values(ascending=False).head(20))
        ax0.bar([str(x) for x in per_group.index], per_group.to_numpy(),
                color="#ec7853", edgecolor="black", alpha=0.85)
        ax0.set_title(f"'{new_col}' by group (top {len(per_group)})",
                      fontweight="bold")
        ax0.set_ylabel(agg)
        ax0.tick_params(axis="x", rotation=60)
        fv = pd.to_numeric(pd.Series(np.asarray(feature)),
                           errors="coerce").dropna()
        if len(fv):
            ax1.hist(fv, bins=_hist_bins(len(fv)),
                     color="#2ca02c", edgecolor="black", alpha=0.85)
        ax1.set_title(f"'{new_col}' across all rows", fontweight="bold")
        ax1.set_ylabel("count")
    fig.suptitle(f"Aggregation features -- {agg}  (Stage 4.3)",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_aggfeat_compare(df, group, value, gk, fig_width, fig_height, dpi):
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height * 1.4), dpi=dpi)
    use = "mean" if pd.api.types.is_numeric_dtype(df[value]) else "count"
    stat = df[value].groupby(gk).apply(lambda s: _agg_compute(s, use))
    stat = stat.dropna().sort_values(ascending=False).head(20)
    ax.bar([str(x) for x in stat.index], stat.to_numpy(),
           color="#4c72b0", edgecolor="black", alpha=0.85)
    ax.set_ylabel(f"{use}({value})")
    ax.set_title(f"'{value}' aggregated by {group}  "
                 f"(top {len(stat)} group(s))", fontweight="bold")
    ax.tick_params(axis="x", rotation=60)
    fig.suptitle("aggfeat COMPARE  (Stage 4.3)",
                 fontsize=14, fontweight="bold")
    return fig


# ===========================================================================
# 8. featpipe  --  chain the seven feature-engineering functions (Stage 4.4)
# ===========================================================================

# Maps the short fn name used inside a step dict to the public function.
_FEATPIPE_DISPATCH = {
    "transform": transform,
    "scale": scale,
    "bin": bin,
    "encode": encode,
    "dtfeats": dtfeats,
    "cross": cross,
    "aggfeat": aggfeat,
}

# dextra I/O / return-control flags that featpipe owns. If a user step carries
# one of these it would collide with featpipe's own keyword call, so they are
# stripped (with a warning) before the sub-function is invoked.
_FEATPIPE_CONTROL_FLAGS = (
    "return_params", "return_df", "return_fig", "show", "plot", "params",
    "df_name", "decimals", "fig_width", "fig_height", "dpi",
)

# Functions whose apply path honours an 'inplace' call argument. For these,
# featpipe re-supplies the fitted 'inplace' choice during apply so the column
# layout reproduces exactly. encode/dtfeats/cross/aggfeat derive placement
# from their own params and must not receive 'inplace'.
_FEATPIPE_INPLACE_FNS = ("transform", "scale", "bin")


def _featpipe_compare_key(fn_name: str) -> str:
    """Return the keyword that selects the method for a given function."""
    return "agg" if fn_name == "aggfeat" else "method"


def _featpipe_validate_steps(steps) -> list:
    """Validate a steps list; return a clean list of ``(fn_name, kwargs)``.

    Rejects malformed steps, unknown function names, and any step that asks
    for ``compare`` mode (compare is exploratory and single-function only;
    featpipe is the commit tool). Control flags owned by featpipe are stripped
    with a warning so they cannot collide with featpipe's own call.
    """
    if not isinstance(steps, (list, tuple)) or len(steps) == 0:
        raise ValueError(
            "featpipe fit mode requires 'steps' to be a non-empty list of "
            "dicts, e.g. [{'fn': 'scale', 'cols': ['price'], "
            "'method': 'robust'}].")
    clean = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(
                f"featpipe step {idx}: each step must be a dict, got "
                f"{type(step).__name__}.")
        if "fn" not in step:
            raise ValueError(
                f"featpipe step {idx}: missing required key 'fn' (one of "
                f"{tuple(_FEATPIPE_DISPATCH)}).")
        fn_name = step["fn"]
        if fn_name not in _FEATPIPE_DISPATCH:
            raise ValueError(
                f"featpipe step {idx}: unknown fn {fn_name!r}; valid "
                f"functions are {tuple(_FEATPIPE_DISPATCH)}.")
        kwargs = {k: v for k, v in step.items() if k != "fn"}
        # Stage 4.4 decision: compare mode is forbidden inside a pipeline.
        ckey = _featpipe_compare_key(fn_name)
        if str(kwargs.get(ckey, "")).lower() == "compare":
            raise ValueError(
                f"featpipe step {idx} (fn={fn_name}): {ckey}='compare' is not "
                f"allowed inside a pipeline. featpipe is a commit tool -- "
                f"explore options with {fn_name}({ckey}='compare') on its own "
                f"first, then chain the chosen {ckey} here.")
        collided = [k for k in kwargs if k in _FEATPIPE_CONTROL_FLAGS]
        if collided:
            warnings.warn(
                f"featpipe step {idx} (fn={fn_name}): control flag(s) "
                f"{collided} are managed by featpipe and were ignored.",
                stacklevel=3)
            kwargs = {k: v for k, v in kwargs.items()
                      if k not in _FEATPIPE_CONTROL_FLAGS}
        clean.append((fn_name, kwargs))
    return clean


def _plot_featpipe(summary_rows, input_cols, df_name, mode,
                   fig_width, fig_height, dpi):
    """Two-panel visual: columns added per step + DataFrame width growth."""
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    labels = [f"{i}:{r['fn']}" for i, r in enumerate(summary_rows)]
    added = [r["cols_added"] for r in summary_rows]
    after = [r["cols_after"] for r in summary_rows]

    ax0 = axes[0]
    ax0.bar(labels, added, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax0.set_ylabel("new columns")
    ax0.set_title("Columns added per step", fontweight="bold")
    ax0.tick_params(axis="x", rotation=45)
    for i, v in enumerate(added):
        ax0.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    ax1 = axes[1]
    xs = ["input"] + labels
    ys = [input_cols] + after
    pos = list(range(len(xs)))
    ax1.plot(pos, ys, marker="o", color="#dd8452", linewidth=2)
    ax1.fill_between(pos, ys, alpha=0.15, color="#dd8452")
    ax1.set_xticks(pos)
    ax1.set_xticklabels(xs, rotation=45, ha="right")
    ax1.set_ylabel("total columns")
    ax1.set_title("DataFrame width through the pipeline", fontweight="bold")
    for x, y in zip(pos, ys):
        ax1.text(x, y, str(y), ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"featpipe  (mode={mode})  --  {df_name}",
                 fontsize=14, fontweight="bold")
    return fig


def featpipe(
    df: pd.DataFrame,
    steps: Optional[Sequence[dict]] = None,
    params: Optional[dict] = None,
    *,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.6,
    dpi: int = 110,
):
    """Chain the seven dextra feature-engineering functions into one pipeline.

    featpipe is the Stage 4.4 convenience wrapper. It runs ``transform``,
    ``scale``, ``bin``, ``encode``, ``dtfeats``, ``cross`` and ``aggfeat`` in
    sequence, threading the transformed DataFrame from one step to the next,
    and collects every step's ``params`` dict into a single combined,
    versioned, JSON-serialisable artifact -- a lightweight feature store.

    Two modes mirror the per-function contract in FEATURES_PHILOSOPHY.md.
    In FIT mode (``steps`` supplied) each step is fitted on ``df`` and its
    params recorded. In APPLY mode (``params`` or ``load_path`` supplied) the
    saved per-step params are replayed verbatim, in order, with no re-fitting
    -- the safeguard against leakage across a train/test boundary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    steps : sequence of dict, optional
        Fit-mode recipe. Each dict has a ``'fn'`` key naming one of
        ``transform / scale / bin / encode / dtfeats / cross / aggfeat``; every
        other key is forwarded as a keyword argument to that function, e.g.
        ``{'fn': 'scale', 'cols': ['price'], 'method': 'robust'}``. A step may
        reference a column produced by an earlier step. ``method='compare'``
        (or ``agg='compare'`` for aggfeat) is rejected -- featpipe commits a
        chosen recipe; explore options with the single function first.
    params : dict, optional
        Apply-mode artifact: a combined dict returned by an earlier fit.
        Triggers apply mode; ``steps`` must not also be given.
    save_path : str, optional
        Fit mode only. After fitting, the combined params dict is written to
        this path as indented JSON.
    load_path : str, optional
        Apply-mode shortcut. The combined params dict is read from this JSON
        file, then applied. Mutually exclusive with ``params`` and ``steps``.
    return_params : bool, default False
        If True the combined params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The fully transformed DataFrame, and -- when requested -- the combined
        params dict and/or the matplotlib figure.

    Notes
    -----
    The combined params dict has the shape::

        {"function": "featpipe", "version": ..., "fit_at": ...,
         "steps": [<params of step 0>, <params of step 1>, ...],
         "metadata": {"n_steps": ..., "step_summary": [...],
                      "input_shape": [...], "output_shape": [...]}}

    Each element of ``steps`` is exactly the JSON-serialisable params dict the
    corresponding function already returns, so the whole artifact survives
    ``json.dump`` / ``json.load`` and reproduces the transform on another
    machine or day.

    Examples
    --------
    >>> recipe = [
    ...     {'fn': 'transform', 'cols': ['income'], 'method': 'log1p'},
    ...     {'fn': 'scale', 'cols': ['income_log1p', 'age'], 'method': 'robust'},
    ...     {'fn': 'encode', 'cols': ['city'], 'method': 'onehot'},
    ... ]
    >>> df_tr, p = dx.featpipe(df_train, steps=recipe, return_params=True,
    ...                        save_path='pipeline.json')
    >>> df_te = dx.featpipe(df_test, params=p)            # apply, no re-fit
    >>> df_te2 = dx.featpipe(df_test, load_path='pipeline.json')  # same result
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- resolve mode ---------------------------------------------------
    if load_path is not None:
        if params is not None or steps is not None:
            raise ValueError(
                "featpipe: 'load_path' is an apply-mode shortcut; do not also "
                "pass 'params' or 'steps'.")
        import json
        with open(load_path, "r", encoding="utf-8") as fh:
            params = json.load(fh)

    if params is not None:
        if steps is not None:
            raise ValueError(
                "featpipe: pass EITHER 'steps' (fit mode) OR 'params' "
                "(apply mode), not both.")
        if save_path is not None:
            raise ValueError(
                "featpipe: 'save_path' saves a freshly fitted pipeline and is "
                "valid in fit mode only (when 'steps' is given).")
        return _featpipe_apply(df, params, show, plot, return_df,
                               return_params, return_fig, decimals, df_name,
                               fig_width, fig_height, dpi)

    if steps is None:
        raise ValueError(
            "featpipe: provide 'steps' to fit a pipeline, or 'params' / "
            "'load_path' to apply a saved one.")

    return _featpipe_fit(df, steps, save_path, show, plot, return_df,
                         return_params, return_fig, decimals, df_name,
                         fig_width, fig_height, dpi)


def _featpipe_fit(df, steps, save_path, show, plot, return_df,
                  return_params, return_fig, decimals, df_name,
                  fig_width, fig_height, dpi):
    clean = _featpipe_validate_steps(steps)

    out = df.copy()
    out.attrs = dict(df.attrs)
    input_cols = out.shape[1]
    step_params_list, summary_rows, step_summary = [], [], []
    prev_cols = list(out.columns)

    for idx, (fn_name, kwargs) in enumerate(clean):
        fn = _FEATPIPE_DISPATCH[fn_name]
        before_n = out.shape[1]
        try:
            new_out, sp = fn(out, return_params=True, return_df=True,
                             show=False, plot=False, **kwargs)
        except Exception as exc:
            raise type(exc)(
                f"featpipe step {idx} (fn={fn_name}, fit): {exc}") from exc
        out = new_out
        after_n = out.shape[1]
        added = [c for c in out.columns if c not in prev_cols]
        prev_cols = list(out.columns)
        step_params_list.append(sp)
        summary_rows.append({
            "fn": fn_name, "method": str(sp.get("method", "-")),
            "cols_before": before_n, "cols_after": after_n,
            "cols_added": len(added)})
        step_summary.append({"step": idx, "fn": fn_name,
                             "method": sp.get("method")})

    combined = {
        "function": "featpipe",
        "version": __version__,
        "fit_at": _now_iso(),
        "steps": step_params_list,
        "metadata": {
            "n_steps": len(clean),
            "step_summary": step_summary,
            "input_shape": list(df.shape),
            "output_shape": list(out.shape),
        },
    }

    saved_note = ""
    if save_path is not None:
        import json
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2)
        saved_note = f" Saved to '{save_path}'."

    n_new = sum(r["cols_added"] for r in summary_rows)
    chain = " -> ".join(r["fn"] for r in summary_rows)
    decision = (f"Fitted a {len(clean)}-step featpipe pipeline ({chain}); "
                f"{n_new} new column(s) produced; combined params is a "
                f"versioned, JSON-serialisable artifact.{saved_note} Apply to "
                f"held-out data with featpipe(df_test, params=...).")

    _append_audit(out, {
        "stage": "feature_pipeline",
        "function": "featpipe",
        "timestamp": combined["fit_at"],
        "mode": "fit",
        "params": {"n_steps": len(clean), "chain": chain,
                   "save_path": save_path},
        "decision": decision,
    })

    summary = pd.DataFrame(summary_rows)
    summary.index.name = "step"
    if show:
        _print_header(f"Feature pipeline for: {df_name}  "
                      f"({len(clean)} step(s), mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_featpipe(summary_rows, input_cols, df_name, "fit",
                             fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, combined, fig, return_df, return_params, return_fig)


def _featpipe_apply(df, params, show, plot, return_df, return_params,
                    return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "featpipe":
        got = (params.get("function") if isinstance(params, dict)
               else type(params).__name__)
        raise ValueError(
            f"featpipe apply: params is not a featpipe pipeline "
            f"(function={got!r}).")
    step_params = params.get("steps")
    if not isinstance(step_params, list) or len(step_params) == 0:
        raise ValueError(
            "featpipe apply: params['steps'] must be a non-empty list of "
            "per-function params dicts.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    input_cols = out.shape[1]
    summary_rows = []
    prev_cols = list(out.columns)

    for idx, sp in enumerate(step_params):
        if not isinstance(sp, dict) or "function" not in sp:
            raise ValueError(
                f"featpipe apply: step {idx} is not a valid params dict.")
        fn_name = sp["function"]
        if fn_name not in _FEATPIPE_DISPATCH:
            raise ValueError(
                f"featpipe apply: step {idx} references unknown function "
                f"{fn_name!r}.")
        fn = _FEATPIPE_DISPATCH[fn_name]
        before_n = out.shape[1]
        call_kwargs = {"params": sp, "show": False, "plot": False,
                       "return_df": True, "return_params": False}
        if fn_name in _FEATPIPE_INPLACE_FNS:
            ip = sp.get("metadata", {}).get("inplace")
            if ip is not None:
                call_kwargs["inplace"] = bool(ip)
        try:
            out = fn(out, **call_kwargs)
        except Exception as exc:
            raise type(exc)(
                f"featpipe step {idx} (fn={fn_name}, apply): {exc}") from exc
        after_n = out.shape[1]
        added = [c for c in out.columns if c not in prev_cols]
        prev_cols = list(out.columns)
        summary_rows.append({
            "fn": fn_name, "method": str(sp.get("method", "-")),
            "cols_before": before_n, "cols_after": after_n,
            "cols_added": len(added)})

    n_new = sum(r["cols_added"] for r in summary_rows)
    fit_at = params.get("fit_at", "?")
    chain = " -> ".join(r["fn"] for r in summary_rows)
    decision = (f"Applied a saved {len(summary_rows)}-step featpipe pipeline "
                f"({chain}; fitted {fit_at}); {n_new} new column(s) produced; "
                f"no re-fit -- leakage-safe.")

    _append_audit(out, {
        "stage": "feature_pipeline",
        "function": "featpipe",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"n_steps": len(summary_rows), "chain": chain,
                   "fit_at": fit_at},
        "decision": decision,
    })

    summary = pd.DataFrame(summary_rows)
    summary.index.name = "step"
    if show:
        _print_header(f"Feature pipeline for: {df_name}  "
                      f"({len(summary_rows)} step(s), mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_featpipe(summary_rows, input_cols, df_name, "apply",
                             fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)
