"""Internal helpers shared across dextra modules.

These utilities are *not* part of the public API and may change without
notice. They exist to keep the public functions (``describe_numeric``,
``plot_histograms``, ``plot_boxplots``) focused on intent rather than
plumbing.
"""

from __future__ import annotations

import inspect
import warnings
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Sequence

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


def _ensure_pandas(df):
    """Accept a pandas / polars / pyarrow table and return a pandas DataFrame.

    pandas frames pass through unchanged (zero overhead). polars DataFrames and
    pyarrow Tables are converted via their ``.to_pandas()`` method (both are
    optional ``perf`` extras). Any other type raises :class:`TypeError`,
    preserving dextra's original input contract.
    """
    if isinstance(df, pd.DataFrame):
        return df
    to_pandas = getattr(df, "to_pandas", None)
    if callable(to_pandas):
        try:
            converted = to_pandas()
        except Exception as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"could not convert {type(df).__name__} to a pandas "
                f"DataFrame via .to_pandas(): {exc}") from exc
        if isinstance(converted, pd.DataFrame):
            return converted
    raise TypeError(
        f"'df' must be a pandas DataFrame (or a polars / pyarrow table exposing "
        f".to_pandas()), got {type(df).__name__}")


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

def resolve_name(df_name, name, func_name):
    """Back-compat shim: ``name=`` is the deprecated alias for ``df_name=``.

    Emits a ``DeprecationWarning`` when ``name=`` is used and maps it onto
    ``df_name`` only if the caller did not already pass ``df_name``.
    """
    if name is not None:
        warnings.warn(
            f"{func_name}: 'name=' is deprecated; use 'df_name=' instead.",
            DeprecationWarning, stacklevel=3)
        if df_name is None:
            df_name = name
    return df_name


def now_iso() -> str:
    """Current UTC time as a compact ISO-8601 string (shared contract helper)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_audit(out: pd.DataFrame, entry: dict) -> None:
    """Append an audit entry to ``out.attrs['dextra_audit']`` (copy-on-write safe).

    Shared, single source of truth for the dextra audit trail. New modules import
    this instead of re-defining it locally.
    """
    out.attrs.setdefault("dextra_audit", [])
    out.attrs["dextra_audit"] = list(out.attrs["dextra_audit"])
    out.attrs["dextra_audit"].append(entry)


def json_safe(value: Any) -> Any:
    """Recursively convert numpy scalars / arrays to plain JSON-serialisable types."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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
