"""dextra features - binning & categorical encoding (bin, encode)."""
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._features_common import (
    _display,
    _print_header,
    _now_iso,
    _finalize_figure,
    _ret_pack,
    _append_audit,
    _fmt_table,
    _resolve_cols,
    _hist_bins,
)
from ._utils import _ensure_pandas, get_variable_name
from ._version import __version__


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
