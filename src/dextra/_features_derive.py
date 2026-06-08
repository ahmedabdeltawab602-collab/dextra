"""dextra features - datetime / interaction / aggregation features."""

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
    _hist_bins,
    _print_header,
    _resolve_cols,
    _ret_pack,
)
from ._utils import _ensure_pandas, append_audit, get_variable_name, now_iso
from ._version import __version__

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
        "fit_at": now_iso(),
        "columns": col_params,
        "metadata": {"drop_original": bool(drop_original), "n_cols": len(cols)},
    }

    total_new = sum(r["n_new_cols"] for r in rows)
    decision = (f"Extracted {total_new} temporal feature(s) from {len(cols)} "
                f"datetime column(s) using method '{method}'. Cyclical "
                f"sin/cos pairs keep periodic distance meaningful.")

    append_audit(out, {
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
    append_audit(out, {
        "stage": "feature_datetime",
        "function": "dtfeats",
        "timestamp": now_iso(),
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
        "fit_at": now_iso(),
        "columns": recipe,
        "metadata": {"degree": int(degree) if method == "polynomial" else None,
                     "n_terms": len(recipe)},
    }

    decision = (f"Created {len(recipe)} interaction feature(s) with method "
                f"'{method}'. Interactions expose relationships a linear "
                f"model cannot see from the raw columns alone.")

    append_audit(out, {
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
    append_audit(out, {
        "stage": "feature_interaction",
        "function": "cross",
        "timestamp": now_iso(),
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
        "fit_at": now_iso(),
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

    append_audit(out, {
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
    append_audit(out, {
        "stage": "feature_aggregation",
        "function": "aggfeat",
        "timestamp": now_iso(),
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
