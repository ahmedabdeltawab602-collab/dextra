"""Internal helpers shared across dextra modules.

These utilities are *not* part of the public API and may change without
notice. They exist to keep the public functions (``describe_numeric``,
``plot_histograms``, ``plot_boxplots``) focused on intent rather than
plumbing.
"""

from __future__ import annotations

import inspect
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Public-ish constants
# ---------------------------------------------------------------------------

#: Default qualitative palette used by :func:`dextra.plot_boxplots` when the
#: caller does not pass an explicit ``colors`` argument. Colours come from the
#: well-known Matplotlib ``tab10`` palette.
DEFAULT_BOX_COLORS: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Column resolution & coercion
# ---------------------------------------------------------------------------

def resolve_columns(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    numeric_only: bool = True,
) -> List[str]:
    """Return the list of columns to operate on.

    If ``cols`` is ``None`` and ``numeric_only=True``, every numeric column
    of ``df`` is returned. If ``cols`` is provided, missing columns raise a
    :class:`KeyError`. Duplicates are removed while preserving the original
    order.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'df' must be a pandas DataFrame, got {type(df).__name__}")

    if cols is None:
        if numeric_only:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError(
                    "No numeric columns found in the DataFrame. "
                    "Pass `cols=` explicitly or convert columns to numeric."
                )
            return numeric_cols
        return df.columns.tolist()

    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Columns not found in DataFrame: {missing}")

    seen: set = set()
    ordered: List[str] = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def to_numeric_frame(df_subset: pd.DataFrame) -> pd.DataFrame:
    """Coerce every column to numeric, turning un-parseable values into NaN."""
    return df_subset.apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------------------------------------
# Variable-name sniffing (best-effort)
# ---------------------------------------------------------------------------

def get_variable_name(obj: object, depth: int = 2) -> str:
    """Best-effort retrieval of the caller-side variable name bound to ``obj``.

    Walks up ``depth`` frames and inspects local variables. Returns
    ``"DataFrame"`` if the search fails (which it often will — e.g. when the
    object was constructed inline). Never raises.
    """
    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is None:
                break
            frame = frame.f_back
        if frame is None:
            return "DataFrame"
        for name, value in frame.f_locals.items():
            if value is obj:
                return name
    except Exception:  # pragma: no cover - defensive
        pass
    return "DataFrame"


# ---------------------------------------------------------------------------
# Safe arithmetic
# ---------------------------------------------------------------------------

def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN instead of inf when dividing by 0.

    Also NaN out results where the denominator is already NaN.
    """
    den = denominator.replace(0, np.nan)
    return numerator / den


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _is_integer_like(values: Iterable) -> bool:
    for v in values:
        if pd.isna(v):
            continue
        if isinstance(v, (int, np.integer)):
            continue
        if isinstance(v, float) and float(v).is_integer():
            continue
        return False
    return True


def format_value(value, kind: str = "num", decimals: int = 2) -> str:
    """Format a single scalar for display.

    ``kind`` controls the rendering:

    * ``"num"``  — thousand-separated float with ``decimals`` digits.
    * ``"int"``  — thousand-separated integer.
    * ``"pct"``  — same as ``"num"`` but suffixed with ``%``.
    * ``"mode"`` — accepts a list/array and joins with ``" | "``.
    """
    if kind == "mode":
        if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
            vals = [x for x in list(value) if pd.notna(x)]
            if not vals:
                return "-"
            if _is_integer_like(vals):
                return " | ".join(f"{int(x):,}" for x in vals)
            return " | ".join(f"{float(x):,.{decimals}f}" for x in vals)
        return "-"

    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except (TypeError, ValueError):
        return str(value)

    if kind == "int":
        try:
            return f"{int(value):,}"
        except (TypeError, ValueError):
            return str(value)
    if kind == "pct":
        try:
            return f"{float(value):,.{decimals}f}%"
        except (TypeError, ValueError):
            return str(value)
    # default numeric
    try:
        return f"{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)
