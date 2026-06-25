"""Descriptive statistics helpers.

This module exposes :func:`describe_numeric`, a drop-in replacement for
``pandas.DataFrame.describe`` that reports a richer set of metrics and
formats the output for quick visual scanning.

The legacy short alias ``numdesc`` is preserved for backward compatibility
with earlier versions of *dextra*.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

from ._utils import (
    append_audit,
    format_value,
    get_variable_name,
    json_safe,
    now_iso,
    resolve_columns,
    safe_divide,
    to_numeric_frame,
)

try:  # Pretty display inside notebooks; fall back to stdout elsewhere.
    from IPython.display import display as _ipy_display
except ImportError:  # pragma: no cover - exercised only outside IPython
    _ipy_display = None


# The canonical order of metrics produced by :func:`describe_numeric`.
_METRIC_ORDER: List[str] = [
    "mean", "std", "var", "cv_%",
    "min", "q1", "median", "diff_mean_median_%", "q3", "max",
    "IQR", "lower_bound", "upper_bound",
    "outliers_count", "outliers_%",
    "count", "missing", "unique",
    "skewness", "kurtosis", "modes",
]


def _display(frame: pd.DataFrame) -> None:
    """Render ``frame`` nicely in a notebook or fall back to ``print``."""
    if _ipy_display is not None:
        _ipy_display(frame)
    else:
        with pd.option_context("display.max_colwidth", 60):
            print(frame.to_string())


def _ret_pack(out, params, fig, return_df, return_params, return_fig):
    """Pack outputs in the fixed order: dataframe, params, figure.

    Only the requested pieces are returned; a single piece is returned bare,
    several as a tuple. Shared Phase-1 contract helper.
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


def describe_numeric(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    decimals: int = 2,
    df_name: Optional[str] = None,
    iqr_multiplier: float = 1.5,
    ddof: int = 1,
    metrics_as_rows: bool = True,
    show: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    return_params: bool = False,
    params: Optional[dict] = None,
    plot: bool = False,
    raw: bool = False,
) -> Optional[pd.DataFrame]:
    """Return a rich numeric summary of ``df``.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Non-numeric values in selected columns are coerced to NaN.
    cols : sequence of str, optional
        Columns to summarise. Defaults to every numeric column in ``df``.
    decimals : int, default ``2``
        Number of fractional digits used for formatting.
    df_name : str, optional
        Name shown in the summary header. If omitted, the caller-side
        variable name is inferred when possible.
    iqr_multiplier : float, default ``1.5``
        Multiplier on the IQR when deriving lower / upper bounds for
        outlier detection. Tukey's classical value is ``1.5``.
    metrics_as_rows : bool, default ``True``
        If ``True`` the output has metrics as rows and columns as columns
        (the dense reading layout). If ``False`` the output is transposed.
    show : bool, default ``True``
        If ``True`` the formatted table is rendered to the notebook / stdout.
    return_df : bool, default ``False``
        If ``True`` a DataFrame is returned.
    raw : bool, default ``False``
        When returning a DataFrame, controls whether numbers are kept as
        ``float64`` (``raw=True``) or pre-formatted strings (``raw=False``).
        ``raw=True`` is what you want if you plan to post-process or export
        the summary (e.g. to Excel).

    Returns
    -------
    pandas.DataFrame or None
        The summary frame, or ``None`` when ``return_df=False``.

    Raises
    ------
    TypeError
        If ``df`` is not a DataFrame.
    KeyError
        If any entry of ``cols`` is missing from ``df``.
    ValueError
        If the resolved column set contains no numeric data.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'a': np.random.randn(100), 'b': np.random.randn(100)})
    >>> summary = describe_numeric(df, return_df=True, raw=True, show=False)
    >>> summary.loc['mean']  # doctest: +SKIP
    """
    if params is not None:
        _cfg = params.get("params", params)
        cols = _cfg.get("cols", cols)
        decimals = _cfg.get("decimals", decimals)
        iqr_multiplier = _cfg.get("iqr_multiplier", iqr_multiplier)
        ddof = _cfg.get("ddof", ddof)
        metrics_as_rows = _cfg.get("metrics_as_rows", metrics_as_rows)
        raw = _cfg.get("raw", raw)

    if decimals < 0:
        raise ValueError(f"'decimals' must be >= 0, got {decimals}")
    if iqr_multiplier <= 0:
        raise ValueError(f"'iqr_multiplier' must be > 0, got {iqr_multiplier}")
    if ddof < 0:
        raise ValueError(f"'ddof' must be >= 0, got {ddof}")

    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    if isinstance(df, pd.DataFrame) and not df.columns.is_unique:
        _dups = sorted(set(map(str, df.columns[df.columns.duplicated()])))
        raise ValueError(
            f"describe_numeric: duplicate column name(s) {_dups} -- dextra needs "
            f"unique column labels. Run dx.tidycols(df) (standardize_columns) "
            f"to de-duplicate them first.")

    cols_resolved = resolve_columns(df, cols, numeric_only=True)
    num_col = to_numeric_frame(df[cols_resolved].copy())

    if num_col.empty or num_col.shape[1] == 0:
        raise ValueError("No numeric columns available to describe.")

    # Central tendency & dispersion
    mean_ = num_col.mean()
    std_ = num_col.std(ddof=ddof)
    var_ = num_col.var(ddof=ddof)
    cv_ = safe_divide(std_, mean_.replace(0, np.nan).abs()) * 100
    min_ = num_col.min()
    max_ = num_col.max()

    # Quartiles
    q1 = num_col.quantile(0.25)
    q2 = num_col.quantile(0.50)
    q3 = num_col.quantile(0.75)
    iqr = q3 - q1

    lb = q1 - iqr_multiplier * iqr
    ub = q3 + iqr_multiplier * iqr

    # Outliers (Tukey's rule)
    out_mask = num_col.lt(lb) | num_col.gt(ub)
    out_count = out_mask.sum()

    # Relative distance between mean and median (sign-safe)
    diff_percent = (mean_ - q2).abs() / q2.replace(0, np.nan).abs() * 100
    diff_percent = diff_percent.fillna(0.0)

    count_ = num_col.count()
    total_rows = pd.Series(len(num_col), index=num_col.columns)
    missing = total_rows - count_
    out_pct = safe_divide(out_count, count_) * 100

    skew_ = num_col.skew()
    kurt_ = num_col.kurt()
    nunique_ = num_col.nunique(dropna=True)

    # Mode can return 0..N values per column; keep raw list for the "raw" path
    # and a pre-joined string for display.
    modes_raw: dict = {}
    modes_str: dict = {}
    for c in num_col.columns:
        s = num_col[c].dropna()
        values = [] if s.empty else s.mode(dropna=True).tolist()
        modes_raw[c] = values
        n_unique_c = s.nunique(dropna=True)
        if not values or n_unique_c == len(s):
            # every value unique -> "mode" is meaningless; avoid table blow-up
            modes_str[c] = "-"
        elif len(values) > 5:
            head = " | ".join(f"{x:,.{decimals}f}" for x in values[:5])
            modes_str[c] = f"{head} | ... ({len(values) - 5} more)"
        else:
            modes_str[c] = " | ".join(f"{x:,.{decimals}f}" for x in values)

    summary = pd.DataFrame({
        "mean": mean_,
        "std": std_,
        "var": var_,
        "cv_%": cv_,
        "min": min_,
        "q1": q1,
        "median": q2,
        "diff_mean_median_%": diff_percent,
        "q3": q3,
        "max": max_,
        "IQR": iqr,
        "lower_bound": lb,
        "upper_bound": ub,
        "outliers_count": out_count,
        "outliers_%": out_pct,
        "count": count_,
        "missing": missing,
        "unique": nunique_,
        "skewness": skew_,
        "kurtosis": kurt_,
    })
    summary["modes"] = pd.Series(
        modes_raw if raw else modes_str, dtype=object
    )

    # Metrics as rows (common reading layout) → transpose and reorder.
    summary_t = summary.T
    summary_t = summary_t.loc[[m for m in _METRIC_ORDER if m in summary_t.index]]

    if raw:
        formatted: pd.DataFrame = summary_t
    else:
        # Cast every value to a display string column-by-column.
        formatted = summary_t.copy()
        for c in formatted.columns:
            new_vals = []
            for r, v in zip(formatted.index, formatted[c]):
                if r == "modes":
                    # modes_str is already a finished, length-capped display
                    # string; re-formatting it would erase it. Pass it through.
                    new_vals.append(v if isinstance(v, str)
                                    else format_value(v, "mode", decimals))
                else:
                    new_vals.append(format_value(v, "num", decimals))
            formatted[c] = new_vals

    if not metrics_as_rows:
        formatted = formatted.T

    if show:
        header = f"Summary for: {df_name}"
        print(header)
        print("-" * len(header))
        _display(formatted)

    config = {
        "cols": list(cols_resolved),
        "decimals": decimals,
        "iqr_multiplier": iqr_multiplier,
        "ddof": ddof,
        "metrics_as_rows": metrics_as_rows,
        "raw": raw,
    }
    audit_entry = {
        "stage": "phase1-eda",
        "function": "describe_numeric",
        "timestamp": now_iso(),
        "df_name": df_name,
        "params": config,
        "decision": (
            f"Summarised {len(cols_resolved)} numeric column(s) of "
            f"'{df_name}'."
        ),
    }
    append_audit(formatted, audit_entry)
    manifest = {
        "stage": "phase1-eda",
        "function": "describe_numeric",
        "df_name": df_name,
        "params": config,
        "summary": json_safe(summary_t.to_dict()),
        "dextra_audit": list(formatted.attrs.get("dextra_audit", [])),
    }
    # describe_numeric has no native figure; plot/return_fig are documented
    # no-ops kept only for unified-contract symmetry.
    packed = _ret_pack(formatted, manifest, None,
                       return_df, return_params, return_fig)
    if not (return_df or return_params or return_fig):
        return None
    return packed


# Backward-compatible short alias. Preserves the original public entry point
# from dextra 0.0.x so existing notebooks keep working.
numdesc = describe_numeric
