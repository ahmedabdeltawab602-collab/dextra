"""dextra features - numeric transforms & scaling (transform, scale)."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._features_common import (
    _display,
    _finalize_figure,
    _fmt_table,
    _get_scipy_stats,
    _hist_bins,
    _print_header,
    _resolve_cols,
    _ret_pack,
)
from ._utils import _ensure_pandas, append_audit, get_variable_name, now_iso
from ._version import __version__

_VALID_TRANSFORM_METHODS = ("log", "log1p", "sqrt", "boxcox", "yeojohnson",
                            "compare")


_TRANSFORM_CANDIDATES = ("log", "log1p", "sqrt", "boxcox", "yeojohnson")


_VALID_SCALE_METHODS = ("standard", "minmax", "robust", "maxabs", "compare")


_SCALE_CANDIDATES = ("standard", "minmax", "robust", "maxabs")


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
        "fit_at": now_iso(),
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

    append_audit(out, {
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
    append_audit(out, {
        "stage": "feature_transform",
        "function": "transform",
        "timestamp": now_iso(),
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
        "fit_at": now_iso(),
        "columns": col_params,
        "metadata": {"inplace": bool(inplace), "n_cols": len(cols)},
    }

    placement = ("overwrote source column(s)" if inplace
                 else f"added new *_{method} column(s)")
    decision = (f"Fitted '{method}' scaler on {len(cols)} column(s); "
                f"{placement}. Apply to held-out data with "
                f"scale(df_test, params=...).")

    append_audit(out, {
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
    append_audit(out, {
        "stage": "feature_scaling",
        "function": "scale",
        "timestamp": now_iso(),
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
