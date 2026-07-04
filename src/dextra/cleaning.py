"""Data cleaning helpers for dextra - Phase 3 of the Roadmap.

Implements the 8-stage cleaning framework documented in CLEANING_PHILOSOPHY.md
at the project root. Every function in this module:

* Accepts a pandas DataFrame and returns a NEW DataFrame (immutable;
  no `inplace` argument).
* Prints a before/after summary table.
* Renders a multi-panel visual showing the change.
* Prints a one-line `Decision:` sentence.
* Appends an entry to `df.attrs['dextra_audit']` so the full cleaning trail
  is auditable via a final `clean_report(df_clean)` call.
* Is idempotent: applying the same function twice yields the same result.

Stage 3.1 - Profile + Structural Cleanup:
  - clean_report(df)         (Stage 0: profile & audit, all DAMA dimensions)
  - standardize_columns(df)  (Stage 1: structural cleanup, consistency)

Aliases:
  - cleanrep, stdcols
"""

from __future__ import annotations

import re
import unicodedata
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import _ensure_pandas, get_variable_name, json_safe, now_iso
from ._version import __version__

try:
    from IPython.display import display as _ipy_display
except ImportError:
    _ipy_display = None


sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

AUDIT_KEY = "dextra_audit"


def _display(frame: pd.DataFrame) -> None:
    if _ipy_display is not None:
        _ipy_display(frame)
    else:
        print(frame.to_string())


def _print_header(title: str) -> None:
    print(title)
    print("-" * len(title))





def _format_per_value(v, kind, decimals):
    if pd.isna(v):
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if kind == "int":
        try:
            return f"{int(v):,}"
        except (TypeError, ValueError):
            return str(v)
    if kind == "pct":
        try:
            return f"{float(v):,.{decimals}f}%"
        except (TypeError, ValueError):
            return str(v)
    try:
        return f"{float(v):,.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _format_summary(
    summary: pd.DataFrame,
    decimals: int,
    int_cols: Sequence[str] = (),
    pct_cols: Sequence[str] = (),
) -> pd.DataFrame:
    out = summary.copy()
    for c in out.columns:
        if c in int_cols:
            out[c] = out[c].map(lambda v: _format_per_value(v, "int", decimals))
        elif c in pct_cols:
            out[c] = out[c].map(lambda v: _format_per_value(v, "pct", decimals))
        else:
            out[c] = out[c].map(lambda v: _format_per_value(v, "num", decimals))
    return out


def _finalize_figure(fig, show, plot, return_fig):
    """Display the figure when one was created.

    ``show`` and ``plot`` are now INDEPENDENT:
      * plot=True   -> figure created (above this call), displayed here.
      * plot=False  -> no figure created, this function returns early.
      * show        -> only controls the printed tabular output (not here).
      * return_fig  -> if True, caller will display; we don't call plt.show().
    """
    if fig is None:
        return
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if not return_fig:
        plt.show()


# ===========================================================================
# STAGE 0 - PROFILE & AUDIT
# ===========================================================================

# Suggested action thresholds (calibrated to common practice)
_MISSING_THRESHOLDS = [
    (0.0,  "OK"),
    (0.05, "review"),         # <5%
    (0.30, "impute"),         # 5-30%
    (0.60, "consider_drop"),  # 30-60%
    (1.01, "drop_column"),    # >60%
]


def _suggested_action(pct_missing: float, n_unique: int, n_rows: int,
                     dtype: str) -> str:
    """Map a column profile to a recommended cleaning action."""
    # pct_missing comes in as a percentage (0-100); normalise to fraction.
    p = pct_missing / 100.0
    # Constant column -> drop regardless of missingness.
    if n_unique <= 1 and n_rows > 1:
        return "drop_constant"
    # Missing-value action thresholds (matches CLEANING_PHILOSOPHY.md):
    if p == 0:
        return "OK"
    if p < 0.05:
        return "review"
    if p < 0.30:
        return "impute_mode" if dtype in ("object", "str") else "impute_median"
    if p < 0.60:
        return "consider_drop"
    return "drop_column"


def _completeness_score(df: pd.DataFrame) -> float:
    """% of non-NA cells across the dataframe."""
    if df.size == 0:
        return 100.0
    return float(100.0 * (1.0 - df.isna().sum().sum() / df.size))


def _uniqueness_score(df: pd.DataFrame) -> float:
    """% of rows that are not exact duplicates."""
    if len(df) == 0:
        return 100.0
    return float(100.0 * (1.0 - df.duplicated().sum() / len(df)))


def _consistency_score(df: pd.DataFrame) -> float:
    """Heuristic: % of object/string columns that DO NOT contain
    leading/trailing whitespace or mixed case duplicates."""
    obj_cols = [c for c in df.columns if df[c].dtype == object or
                str(df[c].dtype) == "str"]
    if not obj_cols:
        return 100.0
    bad = 0
    for c in obj_cols:
        s = df[c].dropna().astype(str)
        if s.empty:
            continue
        # Whitespace issues
        has_ws = (s != s.str.strip()).any()
        # Case duplicates: 'cairo' and 'Cairo' present
        case_dups = s.str.lower().nunique() < s.nunique()
        if has_ws or case_dups:
            bad += 1
    return float(100.0 * (1.0 - bad / len(obj_cols)))


def _quality_score(df: pd.DataFrame) -> Tuple[float, dict]:
    """Weighted composite quality score (0-100)."""
    completeness = _completeness_score(df)
    uniqueness = _uniqueness_score(df)
    consistency = _consistency_score(df)
    # Weights: completeness is most actionable, then uniqueness, then consistency.
    overall = 0.50 * completeness + 0.30 * uniqueness + 0.20 * consistency
    return float(overall), {
        "completeness": completeness,
        "uniqueness": uniqueness,
        "consistency": consistency,
    }


def clean_report(
    df: pd.DataFrame,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 14.0,
    fig_height: float = 8.0,
    dpi: int = 110,
):
    """Stage 0 / Stage 7: comprehensive data-quality audit.

    Produces a per-column profile + global quality scores + cleaning trail
    review. Does NOT modify the input DataFrame.

    For each column reports:
        dtype, n_missing, pct_missing, n_unique, cardinality_pct,
        memory_kb, sample_value, suggested_action.

    Global metrics:
        n_rows, n_cols, n_complete_rows, n_duplicate_rows, memory_mb,
        completeness_score, uniqueness_score, consistency_score,
        quality_score (weighted 0-100).

    Visual: 4-panel figure (missing per column, type composition,
    cardinality histogram, quality score gauge).

    If df was previously cleaned by dextra (carries ``dextra_audit`` in
    ``df.attrs``), the cleaning trail is rendered at the bottom.

    Examples
    --------
    >>> dx.clean_report(df)
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)

    if df.shape[1] == 0:
        raise ValueError(
            "clean_report: nothing to audit -- the DataFrame has no columns. "
            "Pass a frame with at least one column.")
    if not df.columns.is_unique:
        _dups = sorted(set(map(str, df.columns[df.columns.duplicated()])))
        raise ValueError(
            f"clean_report: duplicate column name(s) {_dups} -- dextra needs "
            f"unique column labels. Run dx.tidycols(df) (standardize_columns) "
            f"to de-duplicate them first.")

    n_rows = len(df)
    n_cols = df.shape[1]

    rows = []
    for c in df.columns:
        s = df[c]
        dtype = str(s.dtype)
        n_missing = int(s.isna().sum())
        pct_missing = (n_missing / n_rows * 100) if n_rows else 0.0
        n_unique = int(s.nunique(dropna=True))
        cardinality_pct = (n_unique / n_rows * 100) if n_rows else 0.0
        memory_kb = float(s.memory_usage(deep=True)) / 1024.0
        non_null = s.dropna()
        sample_value = non_null.iloc[0] if len(non_null) else None
        action = _suggested_action(pct_missing, n_unique, n_rows, dtype)
        rows.append({
            "column": c,
            "dtype": dtype,
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "n_unique": n_unique,
            "cardinality_pct": cardinality_pct,
            "memory_kb": memory_kb,
            "sample_value": sample_value,
            "suggested_action": action,
        })
    summary = pd.DataFrame(rows).set_index("column")

    n_duplicate_rows = int(df.duplicated().sum())
    n_complete_rows = int(df.dropna().shape[0])
    memory_mb = float(df.memory_usage(deep=True).sum()) / (1024.0 * 1024.0)
    quality, parts = _quality_score(df)

    if show:
        _print_header(f"Cleaning audit for: {df_name}  "
                      f"(shape={n_rows}x{n_cols}, memory={memory_mb:.{decimals}f} MB)")
        print()
        _print_header("Per-column profile")
        _display(_format_summary(
            summary, decimals,
            int_cols=("n_missing", "n_unique"),
            pct_cols=("pct_missing", "cardinality_pct"),
        ))

        print()
        _print_header("Global quality scores")
        scores_df = pd.DataFrame({
            "score (0-100)": [
                parts["completeness"], parts["uniqueness"],
                parts["consistency"], quality,
            ]
        }, index=["completeness", "uniqueness", "consistency", "OVERALL"])
        _display(_format_summary(scores_df, decimals))

        # Audit trail (if any)
        audit_entries = df.attrs.get(AUDIT_KEY, [])
        if audit_entries:
            print()
            _print_header(f"Cleaning trail ({len(audit_entries)} step(s) applied)")
            trail_rows = []
            for i, e in enumerate(audit_entries, 1):
                trail_rows.append({
                    "#": i,
                    "stage": e.get("stage", "?"),
                    "function": e.get("function", "?"),
                    "timestamp": e.get("timestamp", "?"),
                    "decision": e.get("decision", "?"),
                })
            trail = pd.DataFrame(trail_rows).set_index("#")
            _display(trail)

        print()
        rec_drop = int((summary["suggested_action"] == "drop_column").sum())
        rec_review = int((summary["suggested_action"] == "review").sum())
        rec_impute = int(summary["suggested_action"].str.startswith("impute").sum())
        rec_const = int((summary["suggested_action"] == "drop_constant").sum())
        print(
            f"Decision: quality_score = {quality:.{decimals}f}/100.  "
            f"Suggested actions across {n_cols} column(s): "
            f"drop_column={rec_drop}, drop_constant={rec_const}, "
            f"impute={rec_impute}, review={rec_review}.  "
            f"Duplicate rows: {n_duplicate_rows}.  "
            f"Complete rows: {n_complete_rows}/{n_rows}.\n"
        )

    fig = None
    if plot:
        fig = _plot_clean_report(df, summary, quality, parts, n_duplicate_rows,
                                  fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_clean_report(df, summary, quality, parts, n_dupes,
                       fig_width, fig_height, dpi, decimals):
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=dpi)

    # Panel 1: missing % per column
    ax = axes[0, 0]
    s = summary["pct_missing"].sort_values(ascending=True)
    colors = []
    for v in s.values:
        if v == 0:
            colors.append("#2ca02c")
        elif v < 5:
            colors.append("#9bd09a")
        elif v < 30:
            colors.append("#f7c873")
        elif v < 60:
            colors.append("#ec7853")
        else:
            colors.append("#c73e3e")
    ax.barh(s.index.astype(str), s.values, color=colors,
            edgecolor="black", linewidth=0.4)
    ax.set_xlabel("% missing")
    ax.set_title("Missing per column", fontweight="bold")
    ax.axvline(5,  color="gray", linestyle=":",  linewidth=0.8)
    ax.axvline(30, color="gray", linestyle=":",  linewidth=0.8)
    ax.axvline(60, color="gray", linestyle="--", linewidth=0.8)
    for i, v in enumerate(s.values):
        if v > 0:
            ax.text(v + 0.5, i, f"{v:.{decimals}f}%",
                    va="center", fontsize=8)

    # Panel 2: dtype composition
    ax = axes[0, 1]
    type_groups = {"numeric": 0, "datetime": 0, "boolean": 0,
                   "text/category": 0, "other": 0}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            type_groups["numeric"] += 1
        elif pd.api.types.is_datetime64_any_dtype(s):
            type_groups["datetime"] += 1
        elif pd.api.types.is_bool_dtype(s):
            type_groups["boolean"] += 1
        elif (pd.api.types.is_object_dtype(s) or
              str(s.dtype) == "str" or
              isinstance(s.dtype, pd.CategoricalDtype)):
            type_groups["text/category"] += 1
        else:
            type_groups["other"] += 1
    labels = [k for k, v in type_groups.items() if v > 0]
    sizes = [v for v in type_groups.values() if v > 0]
    ax.pie(sizes, labels=labels, autopct="%.0f%%", startangle=90,
           wedgeprops=dict(width=0.4, edgecolor="white"),
           colors=["#4c8eda", "#ff9966", "#aa66cc", "#66cc99", "#a0a0a0"][:len(labels)])
    ax.set_title("Column type composition", fontweight="bold")

    # Panel 3: cardinality distribution
    ax = axes[1, 0]
    card = summary["cardinality_pct"].sort_values(ascending=True)
    ax.barh(card.index.astype(str), card.values, color="steelblue",
            edgecolor="black", linewidth=0.4, alpha=0.8)
    ax.axvline(50, color="orange", linestyle="--", linewidth=1.0,
               label="50% cardinality")
    ax.axvline(100, color="red", linestyle="--", linewidth=1.0,
               label="all unique")
    ax.set_xlabel("unique values / total rows (%)")
    ax.set_title("Cardinality per column", fontweight="bold")
    ax.legend(fontsize=8)

    # Panel 4: quality scores
    ax = axes[1, 1]
    score_names = ["completeness", "uniqueness", "consistency", "overall"]
    score_values = [parts["completeness"], parts["uniqueness"],
                    parts["consistency"], quality]
    bar_colors = ["#4caf50" if v >= 90 else
                  "#ffc107" if v >= 70 else
                  "#f44336" for v in score_values]
    bars = ax.barh(score_names, score_values, color=bar_colors,
                   edgecolor="black", linewidth=0.5)
    ax.set_xlim(0, 100)
    ax.axvline(70, color="orange", linestyle=":",  linewidth=1.0)
    ax.axvline(90, color="green",  linestyle=":",  linewidth=1.0)
    ax.set_xlabel("score (0-100)")
    ax.set_title(f"Quality scores  (overall = {quality:.1f})",
                 fontweight="bold")
    for bar, v in zip(bars, score_values):
        ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", fontsize=10)

    fig.suptitle("dextra cleaning audit", fontsize=14, fontweight="bold")
    return fig


# ===========================================================================
# STAGE 1 - STRUCTURAL CLEANUP
# ===========================================================================

# Regex for valid Python identifier-like column names
_NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9_]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def _normalize_column_name(name: Any, lowercase: bool = True) -> str:
    """Normalize a single column name to a clean snake_case-like form."""
    s = str(name).strip()
    # Unicode normalize (NFKC fixes e.g. fullwidth chars, ligatures)
    s = unicodedata.normalize("NFKC", s)
    # Replace any non-alphanumeric with underscore
    s = _NON_ALPHANUM.sub("_", s)
    # Collapse multiple underscores
    s = _MULTI_UNDERSCORE.sub("_", s)
    # Strip leading / trailing underscores
    s = s.strip("_")
    if lowercase:
        s = s.lower()
    if not s:
        s = "unnamed"
    return s


def _deduplicate_names(names: Sequence[str]) -> List[str]:
    """If a normalised name repeats, suffix _1, _2, ..."""
    seen: Dict[str, int] = {}
    result = []
    for n in names:
        if n in seen:
            seen[n] += 1
            result.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            result.append(n)
    return result


def standardize_columns(
    df: pd.DataFrame,
    lowercase: bool = True,
    strip_cells: bool = True,
    deduplicate_names: bool = True,
    name_map: Optional[Mapping[str, str]] = None,
    dry_run: bool = False,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 12.0,
    fig_height: float = 5.0,
    dpi: int = 110,
):
    """Stage 1: Structural cleanup.

    Operations applied (returns a NEW DataFrame; original is untouched):
        1. Column names are NFKC-normalised and re-spelled as snake_case-ish:
           every non-alphanumeric char becomes '_', multiple underscores
           collapse, leading/trailing underscores stripped, optional lowercase.
        2. If two columns would collide after normalisation, the second
           receives a numeric suffix (`_1`, `_2`, ...).
        3. Optional explicit `name_map` overrides the auto-naming on a per-key
           basis.
        4. For string/object columns: leading/trailing whitespace is stripped
           from every cell (if `strip_cells=True`).

    DAMA dimension: Consistency.

    Examples
    --------
    >>> dx.standardize_columns(df)
    >>> dx.standardize_columns(df, name_map={"CustID": "customer_id"})
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)

    original_names = list(df.columns)

    # Build the rename map.
    proposed = [_normalize_column_name(c, lowercase=lowercase) for c in original_names]
    if name_map:
        proposed = [name_map.get(orig, prop)
                    for orig, prop in zip(original_names, proposed)]
    if deduplicate_names:
        proposed = _deduplicate_names(proposed)
    name_changes = [(o, p) for o, p in zip(original_names, proposed) if o != p]

    # Build new DataFrame.
    out = df.copy()
    if not dry_run:
        out.columns = proposed

    # Cell-level stripping for string columns.
    n_cells_stripped_per_col: Dict[str, int] = {}
    if strip_cells and not dry_run:
        for new_name in proposed:
            s = out[new_name]
            if (s.dtype == object
                    or str(s.dtype) == "str"
                    or isinstance(s.dtype, pd.CategoricalDtype)):
                # Only apply to string-typed values
                mask = s.notna()
                str_vals = s[mask].astype(str)
                stripped = str_vals.str.strip()
                changed = (str_vals != stripped).sum()
                n_cells_stripped_per_col[new_name] = int(changed)
                if changed:
                    out.loc[mask, new_name] = stripped
            else:
                n_cells_stripped_per_col[new_name] = 0

    # Carry over and append to audit log.
    out.attrs = dict(df.attrs)  # shallow copy
    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])  # decouple from original list
    n_renamed = len(name_changes)
    n_stripped_total = sum(n_cells_stripped_per_col.values())
    if dry_run:
        decision = (f"DRY-RUN: would rename {n_renamed} column(s) and "
                    f"strip {n_stripped_total} cell(s). Set dry_run=False to execute.")
    else:
        decision = (f"Renamed {n_renamed} column(s); "
                    f"stripped whitespace in {n_stripped_total} cell(s).")
    out.attrs[AUDIT_KEY].append({
        "stage": "structural_cleanup",
        "function": "standardize_columns",
        "timestamp": now_iso(),
        "before": {"n_cols": len(original_names)},
        "after":  {"n_cols": len(proposed),
                   "n_renamed_columns": n_renamed,
                   "n_cells_stripped": n_stripped_total},
        "params": {"lowercase": lowercase,
                   "strip_cells": strip_cells,
                   "deduplicate_names": deduplicate_names,
                   "name_map_provided": name_map is not None},
        "decision": decision,
    })

    if show:
        _print_header(f"Structural cleanup for: {df_name}")
        if name_changes:
            changes_df = pd.DataFrame(name_changes, columns=["before", "after"])
            print(f"\nColumn renames ({len(name_changes)}):")
            _display(changes_df)
        else:
            print("\nNo column renames needed.")
        if strip_cells:
            stripped = {c: n for c, n in n_cells_stripped_per_col.items() if n > 0}
            if stripped:
                print("\nWhitespace stripped from cells:")
                strip_df = pd.DataFrame(
                    list(stripped.items()),
                    columns=["column", "n_cells_stripped"],
                ).set_index("column")
                _display(_format_summary(strip_df, decimals,
                                          int_cols=("n_cells_stripped",)))
            else:
                print("\nNo whitespace stripping needed.")
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_standardize_columns(name_changes, n_cells_stripped_per_col,
                                         fig_width, fig_height, dpi)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return out, fig
    if return_df: return out
    if return_fig: return fig
    return out  # Default: return the cleaned DataFrame (cleaning functions ALWAYS return data)


def _plot_standardize_columns(name_changes, n_cells_stripped_per_col,
                               fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)

    # Panel 1: rename table (rendered as text in axes)
    ax = axes[0]
    ax.set_axis_off()
    if name_changes:
        lines = [f"{'BEFORE':<25} -> {'AFTER':<25}"]
        lines.append("-" * 60)
        for o, n in name_changes[:20]:
            lines.append(f"{str(o)[:25]:<25} -> {str(n)[:25]:<25}")
        if len(name_changes) > 20:
            lines.append(f"... ({len(name_changes) - 20} more)")
        ax.text(0.02, 0.98, "\n".join(lines),
                ha="left", va="top", fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.95))
    else:
        ax.text(0.5, 0.5, "No column renames",
                ha="center", va="center", fontsize=12, color="gray")
    ax.set_title(f"Column renames ({len(name_changes)})",
                 fontsize=12, fontweight="bold")

    # Panel 2: cells stripped per column
    ax = axes[1]
    stripped = {c: n for c, n in n_cells_stripped_per_col.items() if n > 0}
    if stripped:
        s = pd.Series(stripped).sort_values(ascending=True)
        ax.barh(s.index, s.values, color="darkorange",
                edgecolor="black", linewidth=0.4)
        ax.set_xlabel("cells with whitespace stripped")
        ax.set_title("Cells stripped per column", fontweight="bold")
        for i, v in enumerate(s.values):
            ax.text(v + max(s.values) * 0.01, i, f"{v:,}",
                    va="center", fontsize=9)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No whitespace stripping needed",
                ha="center", va="center", fontsize=12, color="gray")
        ax.set_title("Cell whitespace", fontweight="bold")

    fig.suptitle("Structural cleanup (Stage 1)", fontsize=14, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Aliases (dextra naming tradition)
# ---------------------------------------------------------------------------

# (`cleanrep` / `stdcols` are deprecated -- see module __getattr__, audit #10.)


# ===========================================================================
# STAGE 2 - TYPE COERCION
# ===========================================================================

# Recognised boolean-like string values for auto-detection
_BOOL_TRUE  = {"true", "yes", "y", "t", "1", "1.0"}
_BOOL_FALSE = {"false", "no", "n", "f", "0", "0.0"}
_BOOL_ALL   = _BOOL_TRUE | _BOOL_FALSE


def _try_boolean_cast(s: pd.Series) -> Optional[pd.Series]:
    """If every non-null value looks boolean, return the casted Series."""
    sc = s.dropna()
    if sc.empty:
        return None
    lowered = sc.astype(str).str.strip().str.lower()
    if not lowered.isin(_BOOL_ALL).all():
        return None
    # Map back to True/False (using nullable BooleanDtype to preserve NA)
    mapped = lowered.map(lambda v: v in _BOOL_TRUE)
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    out.loc[sc.index] = mapped.astype("boolean").values
    return out


def _try_datetime_cast(s: pd.Series, success_threshold: float) -> Optional[pd.Series]:
    """If >= threshold of non-null values parse as datetime, return casted."""
    sc = s.dropna()
    if sc.empty:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(sc, errors="coerce")
    except (TypeError, ValueError):
        return None
    if parsed.notna().sum() / len(sc) < success_threshold:
        return None
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    out.loc[sc.index] = parsed.values
    return out


_CURRENCY_REGEX = re.compile(r"[\$€£¥,\s%]")


def _try_numeric_cast(s: pd.Series, success_threshold: float) -> Optional[pd.Series]:
    """Try direct to_numeric; if it fails for too many values, retry after
    stripping common locale symbols (currency, comma thousand-separators, %)."""
    sc = s.dropna()
    if sc.empty:
        return None
    # Direct attempt
    parsed = pd.to_numeric(sc, errors="coerce")
    success = parsed.notna().sum() / len(sc)
    if success < success_threshold:
        try:
            cleaned = sc.astype(str).str.replace(_CURRENCY_REGEX, "", regex=True)
            parsed = pd.to_numeric(cleaned, errors="coerce")
            success = parsed.notna().sum() / len(sc)
        except Exception:
            return None
    if success < success_threshold:
        return None
    # Decide integer vs float
    non_na = parsed.dropna()
    is_int = (non_na == non_na.astype(np.int64)).all() if len(non_na) else False
    if is_int:
        out = pd.Series(pd.NA, index=s.index, dtype="Int64")
        out.loc[sc.index] = parsed.astype("Int64").values
        return out
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    out.loc[sc.index] = parsed.values
    return out


def _maybe_categorical(s: pd.Series, threshold: int, ratio: float) -> Optional[pd.Series]:
    """If the column has fewer unique values than `threshold` and the
    cardinality ratio is below `ratio`, cast to category."""
    if not (pd.api.types.is_object_dtype(s) or str(s.dtype) == "str"):
        return None
    n = len(s)
    if n == 0:
        return None
    n_unique = s.nunique(dropna=True)
    if n_unique == 0:
        return None
    if n_unique < threshold and (n_unique / n) < ratio:
        return s.astype("category")
    return None


def cast_types(
    df: pd.DataFrame,
    schema: Optional[Mapping[str, str]] = None,
    auto_categorical: bool = True,
    categorical_threshold: int = 50,
    categorical_ratio: float = 0.5,
    parse_threshold: float = 0.90,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 12.0,
    fig_height: float = 5.5,
    dpi: int = 110,
):
    """Stage 2: smart dtype coercion.

    If ``schema`` is provided, casts each named column to the requested dtype.
    If ``schema`` is None, infers the best dtype per column:

    Detection order (for object / str columns only):
        1. Boolean   - every value matches a recognized true/false token.
        2. Datetime  - >= parse_threshold parses with pd.to_datetime.
        3. Numeric   - >= parse_threshold parses with pd.to_numeric
                       (with currency / comma stripping as a fallback).
        4. Category  - cardinality < categorical_threshold AND
                       cardinality / n_rows < categorical_ratio.

    Returns a NEW DataFrame; original is untouched. Memory before/after is
    reported.

    DAMA dimension: Validity.

    Examples
    --------
    >>> dx.cast_types(df)
    >>> dx.cast_types(df, schema={'price': 'float64', 'date': 'datetime64[ns]'})
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)
    if not (0 < parse_threshold <= 1):
        raise ValueError(f"'parse_threshold' must be in (0, 1], got {parse_threshold}")

    out = df.copy()
    memory_before = {c: int(df[c].memory_usage(deep=True)) for c in df.columns}
    dtypes_before = {c: str(df[c].dtype) for c in df.columns}
    changes: List[Dict[str, Any]] = []

    # 1) Apply explicit schema first if provided
    if schema:
        for c, target in schema.items():
            if c not in df.columns:
                raise KeyError(f"schema references missing column: {c!r}")
            try:
                if target.startswith("datetime"):
                    out[c] = pd.to_datetime(out[c], errors="coerce")
                else:
                    out[c] = out[c].astype(target)
            except Exception as e:
                # Try a permissive coercion path
                try:
                    out[c] = pd.to_numeric(out[c], errors="coerce").astype(target)
                except Exception:
                    raise ValueError(
                        f"Could not cast column {c!r} to {target!r}: {e}"
                    ) from e

    # 2) Auto-detection for object/string columns NOT covered by schema
    auto_cols = [c for c in out.columns
                 if (schema is None or c not in schema)
                 and (pd.api.types.is_object_dtype(out[c])
                      or str(out[c].dtype) == "str")]
    for c in auto_cols:
        s = out[c]

        # 2a) boolean
        new = _try_boolean_cast(s)
        if new is not None:
            out[c] = new
            continue
        # 2b) datetime
        new = _try_datetime_cast(s, parse_threshold)
        if new is not None:
            out[c] = new
            continue
        # 2c) numeric
        new = _try_numeric_cast(s, parse_threshold)
        if new is not None:
            out[c] = new
            continue
        # 2d) categorical
        if auto_categorical:
            new = _maybe_categorical(out[c], categorical_threshold,
                                     categorical_ratio)
            if new is not None:
                out[c] = new

    # 3) Record changes
    for c in out.columns:
        before = dtypes_before.get(c, "?")
        after = str(out[c].dtype)
        bytes_before = memory_before.get(c, 0)
        bytes_after = int(out[c].memory_usage(deep=True))
        saved = bytes_before - bytes_after
        if before != after:
            changes.append({
                "column": c,
                "before": before,
                "after": after,
                "memory_before_kb": bytes_before / 1024.0,
                "memory_after_kb": bytes_after / 1024.0,
                "saved_kb": saved / 1024.0,
            })

    # 4) Append to audit log
    out.attrs = dict(df.attrs)
    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])

    total_saved_kb = sum(ch["saved_kb"] for ch in changes)
    decision = (f"Coerced {len(changes)} column(s); "
                f"net memory delta = {total_saved_kb:+.{decimals}f} KB.")
    out.attrs[AUDIT_KEY].append({
        "stage": "type_coercion",
        "function": "cast_types",
        "timestamp": now_iso(),
        "before": {"dtypes": dtypes_before},
        "after":  {"dtypes": {c: str(out[c].dtype) for c in out.columns}},
        "params": {
            "schema_provided": schema is not None,
            "auto_categorical": auto_categorical,
            "categorical_threshold": categorical_threshold,
            "parse_threshold": parse_threshold,
        },
        "decision": decision,
    })

    changes_df = (pd.DataFrame(changes).set_index("column")
                  if changes else pd.DataFrame())

    if show:
        _print_header(f"Type coercion for: {df_name}  "
                      f"(schema={'provided' if schema else 'auto-detected'})")
        if changes:
            _display(_format_summary(changes_df, decimals))
        else:
            print("No type changes applied.")
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_cast_types(changes_df, fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return out, fig
    if return_df: return out
    if return_fig: return fig
    return out


def _plot_cast_types(changes_df, fig_width, fig_height, dpi, decimals):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    if changes_df.empty:
        for ax in axes:
            ax.set_axis_off()
        axes[0].text(0.5, 0.5, "No type changes applied",
                     ha="center", va="center", fontsize=13, color="gray")
        fig.suptitle("Type coercion", fontsize=14, fontweight="bold")
        return fig

    # Panel 1: before -> after table
    ax = axes[0]
    ax.set_axis_off()
    lines = [f"{'COLUMN':<20}  {'BEFORE':<15} -> {'AFTER':<15}"]
    lines.append("-" * 60)
    for col, row in changes_df.iterrows():
        lines.append(f"{str(col)[:20]:<20}  "
                     f"{row['before'][:15]:<15} -> {row['after'][:15]:<15}")
    ax.text(0.02, 0.98, "\n".join(lines),
            ha="left", va="top", fontsize=10, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95))
    ax.set_title(f"Type changes ({len(changes_df)})", fontweight="bold")

    # Panel 2: memory before/after
    ax = axes[1]
    cols = changes_df.index.tolist()
    x = np.arange(len(cols))
    w = 0.35
    ax.bar(x - w/2, changes_df["memory_before_kb"], w,
           label="before", color="steelblue", edgecolor="black")
    ax.bar(x + w/2, changes_df["memory_after_kb"], w,
           label="after", color="darkorange", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c)[:12] for c in cols], rotation=30, ha="right")
    ax.set_ylabel("memory (KB)")
    ax.set_title("Memory before vs after", fontweight="bold")
    ax.legend()

    total_saved = changes_df["saved_kb"].sum()
    fig.suptitle(f"Type coercion (Stage 2) - net delta {total_saved:+.{decimals}f} KB",
                 fontsize=14, fontweight="bold")
    return fig


# ===========================================================================
# STAGE 6 - CONSISTENCY RULES (BUSINESS-RULE VALIDATION)
# ===========================================================================

def validate_rules(
    df: pd.DataFrame,
    rules: Sequence[Dict[str, Any]],
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    return_violations: bool = False,
    fig_width: float = 12.0,
    fig_height: float = 5.0,
    dpi: int = 110,
):
    """Stage 6: business / consistency rules.

    Each rule is a dict with these keys:
        name:        str - unique identifier.
        check:       str expression (eval'd via df.eval) OR callable taking df
                     and returning a boolean Series.  True = passing row.
        description: optional human-readable description (str).
        severity:    optional 'error'|'warning' (default 'error').

    Trust assumption: string `check` rules are executed with `df.eval` and
    callables receive the full DataFrame -- rules are code, not data. Only
    use rules from a trusted source (your own code or configuration); never
    build them from untrusted end-user input.

    For each rule we report n_violations, pct_violations, status.

    `return_violations=True` returns the rows that failed at least one rule
    plus a `violated_rules` column listing the rule names.

    DAMA dimensions: Consistency, Validity.

    Examples
    --------
    >>> rules = [
    ...     {"name": "price_positive", "check": "price >= 0"},
    ...     {"name": "age_valid",     "check": "age.between(0, 120)"},
    ...     {"name": "valid_email",   "check": lambda d: d['email'].str.contains('@')},
    ... ]
    >>> dx.validate_rules(df, rules)
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)
    if not rules:
        raise ValueError("'rules' must be a non-empty sequence of rule dicts.")

    results: List[Dict[str, Any]] = []
    # per-rule violation mask, for later cross-rule reporting
    per_rule_mask: Dict[str, pd.Series] = {}

    for i, rule in enumerate(rules):
        name = rule.get("name", f"rule_{i+1}")
        check = rule.get("check")
        desc = rule.get("description", "")
        severity = rule.get("severity", "error")
        if check is None:
            raise ValueError(f"rule {name!r} missing required 'check' key.")

        try:
            if callable(check):
                passing = check(df)
            else:
                passing = df.eval(check)
            if not isinstance(passing, pd.Series):
                raise TypeError(f"check for {name!r} did not return a Series")
            passing = passing.astype("boolean")  # nullable
            # Violation = NOT passing (NA -> treated as violation)
            violations = ~passing.fillna(False)
            n_v = int(violations.sum())
            pct = (n_v / len(df) * 100) if len(df) else 0.0
            status = "PASS" if n_v == 0 else "FAIL"
            per_rule_mask[name] = violations
        except Exception as e:
            n_v = -1
            pct = float("nan")
            status = "ERROR"
            desc = f"{desc} | {e}"
            per_rule_mask[name] = pd.Series(False, index=df.index)

        results.append({
            "rule": name,
            "severity": severity,
            "n_violations": n_v,
            "pct_violations": pct,
            "status": status,
            "description": desc,
        })

    summary = pd.DataFrame(results).set_index("rule")

    # cross-rule violation mask
    any_violation = pd.Series(False, index=df.index)
    for m in per_rule_mask.values():
        any_violation = any_violation | m

    n_pass = int((summary["status"] == "PASS").sum())
    n_fail = int((summary["status"] == "FAIL").sum())
    n_error = int((summary["status"] == "ERROR").sum())
    n_total = len(rules)
    n_bad_rows = int(any_violation.sum())

    if n_error > 0:
        _errs = summary.index[summary["status"] == "ERROR"].tolist()
        warnings.warn(
            f"validate_rules: {n_error} rule(s) could not be evaluated and "
            f"were marked ERROR: {_errs}. Their check raised an exception; "
            f"treat the validation as incomplete.", UserWarning, stacklevel=2)

    if show:
        _print_header(f"Rule validation for: {df_name}  ({n_total} rule(s))")
        _display(_format_summary(
            summary, decimals,
            int_cols=("n_violations",),
            pct_cols=("pct_violations",),
        ))
        print(f"\nDecision: {n_pass} PASS, {n_fail} FAIL, {n_error} ERROR "
              f"out of {n_total} rule(s).  {n_bad_rows} row(s) violate at "
              f"least one rule.\n")

    fig = None
    if plot:
        fig = _plot_validate_rules(summary, fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    # audit log
    summary_attrs = dict(df.attrs)
    summary_attrs.setdefault(AUDIT_KEY, [])
    summary_attrs[AUDIT_KEY] = list(summary_attrs[AUDIT_KEY])
    summary_attrs[AUDIT_KEY].append({
        "stage": "consistency_rules",
        "function": "validate_rules",
        "timestamp": now_iso(),
        "before": {"n_rows": len(df)},
        "after":  {"n_pass": n_pass, "n_fail": n_fail, "n_error": n_error,
                   "n_violating_rows": n_bad_rows},
        "params": {"n_rules": n_total},
        "decision": f"{n_pass}/{n_total} pass, {n_bad_rows} row(s) flagged.",
    })

    if return_violations:
        if n_bad_rows:
            flagged = df.loc[any_violation].copy()
            flagged["violated_rules"] = [
                ", ".join(name for name, m in per_rule_mask.items() if m.loc[idx])
                for idx in flagged.index
            ]
            flagged.attrs = summary_attrs
        else:
            flagged = df.iloc[0:0].copy()
            flagged["violated_rules"] = pd.Series(dtype="object")
            flagged.attrs = summary_attrs
        if return_df and return_fig: return summary, flagged, fig
        if return_df: return summary, flagged
        return flagged

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_validate_rules(summary, fig_width, fig_height, dpi, decimals):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    # Sort by violations descending
    s = summary.sort_values("n_violations", ascending=True)
    colors = []
    for st, v in zip(s["status"], s["n_violations"]):
        if st == "PASS":
            colors.append("#2ca02c")
        elif st == "ERROR":
            colors.append("#7f7f7f")
        elif v == 0:
            colors.append("#9bd09a")
        elif v < 10:
            colors.append("#f7c873")
        elif v < 100:
            colors.append("#ec7853")
        else:
            colors.append("#c73e3e")
    bars = ax.barh(s.index.astype(str),
                   s["n_violations"].clip(lower=0).values,
                   color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("number of violations")
    ax.set_title(f"Business-rule validation ({len(summary)} rules)",
                 fontsize=13, fontweight="bold")
    for bar, n, st in zip(bars, s["n_violations"], s["status"]):
        label = f"{int(n):,} ({st})" if n >= 0 else st
        ax.text(max(bar.get_width(), 0) + 0.1, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=9)
    return fig


# (Stage 3.2 aliases `cast` / `vrules` are deprecated -- see __getattr__.)


# ===========================================================================
# STAGE 3 - MISSING VALUE HANDLING
# ===========================================================================

_VALID_MISSING_STRATEGIES = (
    "auto", "mean", "median", "mode", "ffill", "bfill",
    "constant", "drop_rows", "drop_cols",
    # Stage 3.5 additions:
    "keep", "random_uniform", "random_normal", "random_sample", "interpolate",
)


def _auto_strategy_for_column(s: pd.Series) -> str:
    """Pick the best imputation strategy for a single column."""
    if pd.api.types.is_bool_dtype(s):
        return "mode"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "ffill"
    if pd.api.types.is_numeric_dtype(s):
        non_na = s.dropna()
        if len(non_na) < 3:
            return "median"
        try:
            sk = float(non_na.skew())
            return "median" if abs(sk) > 0.5 else "mean"
        except (TypeError, ValueError):
            return "median"
    # categorical or string
    return "mode"


def _impute_column(s: pd.Series, strategy: str,
                   fill_value=None, rng=None) -> pd.Series:
    """Apply a single-column imputation strategy. Returns a NEW Series.

    ``rng`` is a numpy Generator used by random_* strategies for
    reproducibility.
    """
    out = s.copy()
    n_before = int(out.isna().sum())
    if n_before == 0:
        return out

    if strategy == "keep":
        # Explicit no-op: leave the NaN as is.
        return out

    if strategy == "mean":
        if pd.api.types.is_numeric_dtype(out):
            non_na = out.dropna()
            if not non_na.empty:
                out = out.fillna(non_na.mean())
    elif strategy == "median":
        if pd.api.types.is_numeric_dtype(out):
            non_na = out.dropna()
            if not non_na.empty:
                out = out.fillna(non_na.median())
    elif strategy == "mode":
        modes = out.dropna().mode()
        if len(modes):
            fill = modes.iloc[0]
            if isinstance(out.dtype, pd.CategoricalDtype):
                if fill not in out.cat.categories:
                    out = out.cat.add_categories([fill])
            out = out.fillna(fill)
    elif strategy == "ffill":
        out = out.ffill().bfill()
    elif strategy == "bfill":
        out = out.bfill().ffill()
    elif strategy == "constant":
        # Categorical columns reject fillna with a value that is not in their
        # category set.  Add the fill_value to the categories first if needed.
        if isinstance(out.dtype, pd.CategoricalDtype):
            if fill_value not in out.cat.categories:
                out = out.cat.add_categories([fill_value])
        out = out.fillna(fill_value)
    elif strategy == "interpolate":
        if pd.api.types.is_numeric_dtype(out):
            # Linear interpolation; backfill / forward fill cover the edges.
            out = out.interpolate(method="linear",
                                  limit_direction="both")
    elif strategy == "random_uniform":
        if pd.api.types.is_numeric_dtype(out):
            non_na = out.dropna()
            if not non_na.empty:
                lo, hi = float(non_na.min()), float(non_na.max())
                if rng is None:
                    rng = np.random.default_rng()
                draws = rng.uniform(lo, hi, n_before)
                out.loc[out.isna()] = draws
    elif strategy == "random_normal":
        if pd.api.types.is_numeric_dtype(out):
            non_na = out.dropna()
            if not non_na.empty:
                mu = float(non_na.mean())
                sd = float(non_na.std())
                if pd.isna(sd) or sd == 0:
                    sd = 1.0
                if rng is None:
                    rng = np.random.default_rng()
                draws = rng.normal(mu, sd, n_before)
                out.loc[out.isna()] = draws
    elif strategy == "random_sample":
        non_na = out.dropna()
        if not non_na.empty:
            if rng is None:
                rng = np.random.default_rng()
            draws = rng.choice(non_na.values, size=n_before, replace=True)
            out.loc[out.isna()] = draws
    else:
        raise ValueError(f"Unknown column strategy: {strategy!r}")
    return out


_FROZEN_FILL_STRATEGIES = ("mean", "median", "mode", "constant")


def _fill_value_kind(v):
    """Classify a fill value for JSON-safe storage and faithful replay."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None, None
    if isinstance(v, (bool, np.bool_)):
        return bool(v), "bool"
    if isinstance(v, (int, np.integer)):
        return int(v), "number"
    if isinstance(v, (float, np.floating)):
        return float(v), "number"
    if isinstance(v, pd.Timestamp):
        return v.isoformat(), "datetime"
    if isinstance(v, str):
        return v, "text"
    return str(v), "other"


def _fit_fill_entry(s: pd.Series, strategy: str, fill_value) -> dict:
    """Build one column's replayable fill entry, learned from the FIT data.

    mean/median/mode/constant freeze the fill value; random_uniform /
    random_normal freeze the fitted distribution; the order-based strategies
    (ffill/bfill/interpolate) and keep/random_sample carry no train
    statistic and are re-run on apply (disclosed there).
    """
    entry = {"strategy": strategy, "fill_value": None, "value_kind": None}
    non_na = s.dropna()
    if strategy == "mean" and pd.api.types.is_numeric_dtype(s) and len(non_na):
        entry["fill_value"], entry["value_kind"] = _fill_value_kind(s.mean())
    elif (strategy == "median" and pd.api.types.is_numeric_dtype(s)
          and len(non_na)):
        entry["fill_value"], entry["value_kind"] = _fill_value_kind(s.median())
    elif strategy == "mode":
        modes = non_na.mode()
        if len(modes):
            entry["fill_value"], entry["value_kind"] = (
                _fill_value_kind(modes.iloc[0]))
    elif strategy == "constant":
        entry["fill_value"], entry["value_kind"] = _fill_value_kind(fill_value)
    elif (strategy == "random_uniform"
          and pd.api.types.is_numeric_dtype(s) and len(non_na)):
        entry["bounds"] = [float(non_na.min()), float(non_na.max())]
    elif (strategy == "random_normal"
          and pd.api.types.is_numeric_dtype(s) and len(non_na)):
        sd = float(non_na.std())
        entry["dist"] = [float(non_na.mean()),
                         1.0 if (pd.isna(sd) or sd == 0) else sd]
    return entry


def _apply_fill_column(s: pd.Series, entry: dict, rng) -> pd.Series:
    """Fill one apply-side column from a fitted entry. Returns a NEW Series."""
    strategy = entry.get("strategy")
    value, kind = entry.get("fill_value"), entry.get("value_kind")
    if strategy == "keep":
        return s.copy()
    if value is not None:
        fill = pd.Timestamp(value) if kind == "datetime" else value
        out = s.copy()
        if isinstance(out.dtype, pd.CategoricalDtype):
            if fill not in out.cat.categories:
                out = out.cat.add_categories([fill])
        return out.fillna(fill)
    if strategy in _FROZEN_FILL_STRATEGIES:
        return s.copy()  # nothing was learnable at fit -> leave as is
    if strategy == "random_uniform":
        out = s.copy()
        n = int(out.isna().sum())
        if n and "bounds" in entry:
            lo, hi = entry["bounds"]
            out.loc[out.isna()] = rng.uniform(lo, hi, n)
        return out
    if strategy == "random_normal":
        out = s.copy()
        n = int(out.isna().sum())
        if n and "dist" in entry:
            mu, sd = entry["dist"]
            out.loc[out.isna()] = rng.normal(mu, sd, n)
        return out
    if strategy in ("ffill", "bfill", "interpolate", "random_sample"):
        # order-based / sampling strategies have no train statistic; re-run
        return _impute_column(s, strategy, rng=rng)
    return s.copy()


def _handle_missing_apply(df, params, random_state, decimals, df_name,
                          show, plot, return_df, return_params, return_fig,
                          fig_width, fig_height, dpi):
    """APPLY mode for handle_missing: replay fitted fills, never re-compute."""
    if (not isinstance(params, dict)
            or params.get("function") != "handle_missing"):
        got = params.get("function") if isinstance(params, dict) else params
        raise ValueError(
            f"params dict is not for 'handle_missing' (function={got!r}).")
    meta = params.get("metadata", {}) or {}
    mode_level = meta.get("mode_level")
    fitted_at = params.get("fit_at", "?")
    n_before_rows = len(df)

    if mode_level == "drop_rows":
        out = df.dropna(axis=0, how="any").copy()
        per_col_log = []
        decision = (f"Applied saved 'drop_rows' plan (fitted {fitted_at}): "
                    f"dropped {n_before_rows - len(out)} row(s) with any NaN "
                    f"(re-run; no train statistic involved).")
    elif mode_level == "drop_cols":
        fitted_drop = list(meta.get("dropped_cols", []))
        absent = [c for c in fitted_drop if c not in df.columns]
        out = df.drop(
            columns=[c for c in fitted_drop if c in df.columns]).copy()
        per_col_log = [{"column": c, "strategy": "drop_col (fitted)",
                        "filled": 0} for c in fitted_drop]
        decision = (f"Applied saved 'drop_cols' plan (fitted {fitted_at}): "
                    f"dropped the {len(fitted_drop)} column(s) chosen on the "
                    f"fit data; no re-decision -- leakage-safe."
                    + (f" {len(absent)} fitted column(s) absent here: "
                       f"{absent}." if absent else ""))
    else:
        col_params = params.get("columns", {}) or {}
        missing_cols = [c for c in col_params if c not in df.columns]
        if missing_cols:
            raise KeyError(
                f"handle_missing apply failed: params expects column(s) "
                f"{missing_cols} which are not present in this DataFrame. "
                f"The data does not match the fitted plan.")
        rng = np.random.default_rng(
            random_state if random_state is not None
            else meta.get("random_state"))
        out = df.copy()
        per_col_log, resampled = [], []
        for c, entry in col_params.items():
            n_missing = int(out[c].isna().sum())
            if n_missing == 0:
                continue
            filled_col = _apply_fill_column(out[c], entry, rng)
            n_after = int(filled_col.isna().sum())
            out[c] = filled_col
            if entry.get("strategy") == "random_sample":
                resampled.append(c)
            per_col_log.append({
                "column": c,
                "strategy": f"{entry.get('strategy')} (fitted)",
                "missing_before": n_missing,
                "filled": n_missing - n_after,
                "missing_after": n_after,
            })
        unfitted = [c for c in out.columns
                    if c not in col_params and int(out[c].isna().sum()) > 0]
        if unfitted:
            warnings.warn(
                f"handle_missing: {len(unfitted)} column(s) with missing "
                f"values had no fitted fill (not part of the fit plan): "
                f"{unfitted}; left as NaN.", UserWarning, stacklevel=3)
        if resampled:
            warnings.warn(
                f"handle_missing: column(s) {resampled} use 'random_sample', "
                f"which re-samples from THIS data on apply (no train "
                f"statistic is stored).", UserWarning, stacklevel=3)
        total = sum(e.get("filled", 0) for e in per_col_log)
        decision = (f"Applied saved missing-value fills (fitted {fitted_at}) "
                    f"to {len(per_col_log)} column(s), {total} cell(s) "
                    f"filled with fit-time values; no re-fit -- "
                    f"leakage-safe.")

    out.attrs = dict(df.attrs)
    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
    out.attrs[AUDIT_KEY].append({
        "stage": "missing_values",
        "function": "handle_missing",
        "timestamp": now_iso(),
        "mode": "apply",
        "params": {"strategy": params.get("strategy"), "fit_at": fitted_at},
        "decision": decision,
    })
    if show:
        _print_header(f"Missing handling for: {df_name}  "
                      f"(strategy={params.get('strategy')}, mode=apply)")
        if per_col_log:
            log_df = pd.DataFrame(per_col_log).set_index("column")
            int_cols = tuple(c for c in ("missing_before", "filled",
                                         "missing_after")
                             if c in log_df.columns)
            _display(_format_summary(log_df, decimals, int_cols=int_cols))
        else:
            print("No missing values to handle.")
        print(f"\nDecision: {decision}\n")
    fig = None
    if plot:
        fig = _plot_handle_missing(df, out, fig_width, fig_height, dpi,
                                   decimals)
    _finalize_figure(fig, show, plot, return_fig)
    return _ret_pack(out, fig, return_df, return_fig,
                     params=params, return_params=return_params)


def handle_missing(
    df: pd.DataFrame,
    strategy: Union[str, Mapping[str, str]] = "auto",
    drop_threshold: float = 0.60,
    fill_value: Any = None,
    random_state: Optional[int] = None,
    dry_run: bool = False,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 14.0,
    fig_height: float = 5.5,
    dpi: int = 110,
    params: Optional[dict] = None,
    return_params: bool = False,
):
    """Stage 3: handle missing values.

    Strategies (string, or dict for per-column):
        'auto'      - smart per-column: median for skewed numeric, mean for
                       symmetric numeric, mode for categorical, ffill for
                       datetime, mode for boolean.
        'mean'      - mean (numeric only).
        'median'    - median (numeric only).
        'mode'      - most frequent (any dtype).
        'ffill'     - forward fill then back fill.
        'bfill'     - back fill then forward fill.
        'constant'  - fill with `fill_value`.
        'drop_rows' - drop rows containing ANY NaN.
        'drop_cols' - drop columns with > `drop_threshold` missing.
        dict        - per-column: {'colname': strategy_str}.

    Leakage-safe fit/apply: in FIT mode (``params=None``) pass
    ``return_params=True`` to also get a replayable params dict
    holding the fill values learned from THIS data; in APPLY mode
    (``params=<dict>``) those values are applied verbatim -- never
    recomputed -- so held-out data is filled with train statistics.
    mean/median/mode/constant freeze the fill value;
    random_uniform / random_normal freeze the train distribution;
    ffill/bfill/interpolate are order-based re-runs (no train
    statistic exists); random_sample re-samples from the apply-side
    data (warned). 'drop_cols' replays the fitted column drop;
    'drop_rows' re-runs (no statistic).

    Columns that are entirely missing have no statistic to impute
    from and are left unchanged (no RuntimeWarning is emitted).

    DAMA dimension: Completeness.

    Examples
    --------
    >>> dx.handle_missing(df)                              # auto
    >>> dx.handle_missing(df, strategy={'price': 'median', 'name': 'mode'})
    >>> dx.handle_missing(df, strategy='drop_cols', drop_threshold=0.5)
    >>> tr, p = dx.handle_missing(train, strategy='mean', return_params=True)
    >>> te = dx.handle_missing(test, params=p)   # train means, no re-fit
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)

    if params is not None:
        if dry_run:
            raise ValueError(
                "handle_missing: dry_run is a fit-mode flag and cannot "
                "be combined with params= (apply mode).")
        return _handle_missing_apply(
            df, params, random_state, decimals, df_name, show, plot,
            return_df, return_params, return_fig,
            fig_width, fig_height, dpi)
    if dry_run and return_params:
        raise ValueError(
            "handle_missing: dry_run does not fit parameters; run with "
            "dry_run=False to get a replayable params dict.")

    n_before_rows, n_before_cols = df.shape
    missing_before = df.isna().sum()

    # ---------- dataset-level strategies ----------
    if strategy == "drop_rows":
        out = df.dropna(axis=0, how="any").copy()
        n_after_rows = len(out)
        n_dropped = n_before_rows - n_after_rows
        per_col_log: List[Dict[str, Any]] = []
        decision = (f"Dropped {n_dropped} row(s) with any NaN. "
                    f"Rows {n_before_rows} -> {n_after_rows}.")
        out.attrs = dict(df.attrs)
        out.attrs.setdefault(AUDIT_KEY, [])
        out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
        out.attrs[AUDIT_KEY].append({
            "stage": "missing_values",
            "function": "handle_missing",
            "timestamp": now_iso(),
            "before": {"n_rows": n_before_rows, "n_missing_cells": int(missing_before.sum())},
            "after":  {"n_rows": n_after_rows, "n_missing_cells": int(out.isna().sum().sum())},
            "params": {"strategy": "drop_rows"},
            "decision": decision,
        })
        _emit_missing_report(out, df, n_before_rows, n_after_rows,
                              per_col_log, decision, decimals,
                              df_name, "drop_rows",
                              show, plot, return_fig, fig_width, fig_height, dpi)
        params_out = None
        if return_params:
            params_out = {
                "function": "handle_missing", "strategy": "drop_rows",
                "version": __version__, "fit_at": now_iso(),
                "columns": {},
                "metadata": {"mode_level": "drop_rows"},
            }
        return _ret_pack(out, None, return_df, return_fig,
                         params=params_out, return_params=return_params)

    if strategy == "drop_cols":
        miss_pct = (missing_before / n_before_rows).fillna(0)
        cols_to_drop = miss_pct[miss_pct > drop_threshold].index.tolist()
        out = df.drop(columns=cols_to_drop).copy()
        per_col_log = [{"column": c, "strategy": "drop_col", "filled": 0,
                        "pct_missing_before": float(miss_pct[c]) * 100}
                       for c in cols_to_drop]
        decision = (f"Dropped {len(cols_to_drop)} column(s) with > "
                    f"{drop_threshold*100:.0f}% missing: "
                    f"{cols_to_drop if cols_to_drop else '(none)'}.")
        out.attrs = dict(df.attrs)
        out.attrs.setdefault(AUDIT_KEY, [])
        out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
        out.attrs[AUDIT_KEY].append({
            "stage": "missing_values",
            "function": "handle_missing",
            "timestamp": now_iso(),
            "before": {"n_cols": n_before_cols},
            "after":  {"n_cols": out.shape[1], "dropped": cols_to_drop},
            "params": {"strategy": "drop_cols", "drop_threshold": drop_threshold},
            "decision": decision,
        })
        _emit_missing_report(out, df, n_before_rows, len(out),
                              per_col_log, decision, decimals,
                              df_name, "drop_cols",
                              show, plot, return_fig, fig_width, fig_height, dpi)
        params_out = None
        if return_params:
            params_out = {
                "function": "handle_missing", "strategy": "drop_cols",
                "version": __version__, "fit_at": now_iso(),
                "columns": {},
                "metadata": {"mode_level": "drop_cols",
                             "dropped_cols": list(cols_to_drop),
                             "drop_threshold": drop_threshold},
            }
        return _ret_pack(out, None, return_df, return_fig,
                         params=params_out, return_params=return_params)

    # ---------- per-column imputation ----------
    if isinstance(strategy, str):
        if strategy not in _VALID_MISSING_STRATEGIES:
            raise ValueError(
                f"'strategy' must be one of {_VALID_MISSING_STRATEGIES} "
                f"or a dict, got {strategy!r}"
            )
        global_strategy = strategy
        per_col_strategy = None
    else:
        global_strategy = None
        per_col_strategy = dict(strategy)
        bad = [k for k in per_col_strategy if k not in df.columns]
        if bad:
            raise KeyError(f"strategy dict references missing columns: {bad}")

    if dry_run:
        # In dry-run we only count what WOULD be imputed; we do NOT change anything.
        out = df.copy()
        per_col_log = []
        for c in df.columns:
            n_missing = int(df[c].isna().sum())
            if n_missing == 0:
                continue
            if per_col_strategy is not None:
                col_strat = per_col_strategy.get(c, "(unspecified - skipped)")
            else:
                col_strat = (_auto_strategy_for_column(df[c])
                              if global_strategy == "auto" else global_strategy)
            per_col_log.append({
                "column": c, "strategy": col_strat,
                "missing_before": n_missing, "filled": 0,  # nothing actually filled
                "missing_after": n_missing,
            })
        total_would_fill = sum(int(df[c].isna().sum()) for c in df.columns)
        decision = (f"DRY-RUN: would impute {total_would_fill} cell(s) using "
                    f"strategy {strategy!r}. Set dry_run=False to execute.")
        out.attrs = dict(df.attrs)  # don't even append to audit log on dry-run
        if show:
            _print_header(f"Missing handling (dry-run) for: {df_name}  (strategy={strategy})")
            if per_col_log:
                _display(_format_summary(pd.DataFrame(per_col_log).set_index("column"),
                                          decimals,
                                          int_cols=("missing_before", "filled",
                                                     "missing_after")))
            print(f"\nDecision: {decision}\n")
        return _ret_pack(out, None, return_df, return_fig)

    out = df.copy()
    per_col_log = []
    _rng = (np.random.default_rng(random_state)
            if random_state is not None else np.random.default_rng())
    for c in out.columns:
        s = out[c]
        n_missing = int(s.isna().sum())
        if n_missing == 0:
            continue
        if per_col_strategy is not None:
            col_strat = per_col_strategy.get(c)
            if col_strat is None:
                continue  # not specified -> leave it
        else:
            col_strat = (_auto_strategy_for_column(s)
                          if global_strategy == "auto" else global_strategy)
        try:
            out[c] = _impute_column(s, col_strat,
                                     fill_value=fill_value, rng=_rng)
        except Exception as e:
            per_col_log.append({
                "column": c, "strategy": col_strat,
                "filled": 0, "error": str(e),
            })
            continue
        n_after = int(out[c].isna().sum())
        per_col_log.append({
            "column": c,
            "strategy": col_strat,
            "missing_before": n_missing,
            "filled": n_missing - n_after,
            "missing_after": n_after,
        })

    total_filled = sum(e.get("filled", 0) for e in per_col_log)
    n_cols_imputed = sum(1 for e in per_col_log if e.get("filled", 0) > 0)
    decision = (f"Imputed {total_filled} cell(s) across {n_cols_imputed} "
                f"column(s) using strategy '{strategy}'.")
    if return_params:
        decision += (" Fitted fill values saved; apply to held-out "
                     "data with handle_missing(df_test, params=...).")

    out.attrs = dict(df.attrs)
    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
    out.attrs[AUDIT_KEY].append({
        "stage": "missing_values",
        "function": "handle_missing",
        "timestamp": now_iso(),
        "before": {"n_missing_cells": int(missing_before.sum())},
        "after":  {"n_missing_cells": int(out.isna().sum().sum())},
        "params": {"strategy": str(strategy),
                   "drop_threshold": drop_threshold,
                   "fill_value": fill_value,
                   "random_state": random_state},
        "decision": decision,
    })

    params_out = None
    if return_params:
        out.attrs[AUDIT_KEY][-1]["mode"] = "fit"
        col_entries = {}
        for c in df.columns:
            if per_col_strategy is not None:
                strat_c = per_col_strategy.get(c)
                if strat_c is None:
                    continue
            else:
                strat_c = (_auto_strategy_for_column(df[c])
                           if global_strategy == "auto"
                           else global_strategy)
            entry = _fit_fill_entry(df[c], strat_c, fill_value)
            entry["missing_in_fit"] = int(df[c].isna().sum())
            col_entries[c] = entry
        params_out = json_safe({
            "function": "handle_missing",
            "strategy": (dict(per_col_strategy)
                         if per_col_strategy is not None
                         else global_strategy),
            "version": __version__,
            "fit_at": now_iso(),
            "columns": col_entries,
            "metadata": {"mode_level": None,
                         "drop_threshold": drop_threshold,
                         "random_state": random_state},
        })
    fig = _emit_missing_report(out, df, n_before_rows, len(out),
                                per_col_log, decision, decimals,
                                df_name, str(strategy),
                                show, plot, return_fig, fig_width, fig_height, dpi)
    return _ret_pack(out, fig, return_df, return_fig,
                     params=params_out, return_params=return_params)


def _emit_missing_report(out, df, n_before_rows, n_after_rows,
                         per_col_log, decision, decimals, df_name, strategy_label,
                         show, plot, return_fig, fig_width, fig_height, dpi):
    if show:
        _print_header(f"Missing handling for: {df_name}  (strategy={strategy_label})")
        if per_col_log:
            log_df = pd.DataFrame(per_col_log).set_index("column")
            int_cols = tuple(c for c in ("missing_before", "filled", "missing_after")
                              if c in log_df.columns)
            _display(_format_summary(log_df, decimals, int_cols=int_cols))
        else:
            print("No missing values to handle.")
        print(f"\nDecision: {decision}\n")
    fig = None
    if plot:
        fig = _plot_handle_missing(df, out, fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)
    return fig


def _ret_pack(out, fig, return_df, return_fig, params=None,
              return_params=False):
    if return_params:
        # unified contract order: dataframe, params, figure
        if return_fig:
            return out, params, fig
        return out, params
    if return_df and return_fig: return out, fig
    if return_df: return out
    if return_fig: return fig
    return out  # cleaning functions ALWAYS return data


def _plot_handle_missing(df_before, df_after, fig_width, fig_height, dpi, decimals):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    common_cols = [c for c in df_before.columns if c in df_after.columns]
    miss_before = df_before[common_cols].isna().sum()
    miss_after = df_after[common_cols].isna().sum()
    x = np.arange(len(common_cols))
    w = 0.4
    ax.bar(x - w/2, miss_before.values, w, label="before",
           color="#ec7853", edgecolor="black")
    ax.bar(x + w/2, miss_after.values, w, label="after",
           color="#2ca02c", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(common_cols, rotation=30, ha="right")
    ax.set_ylabel("# missing cells")
    ax.set_title("Missing values before vs after", fontsize=13, fontweight="bold")
    ax.legend()
    return fig


# ===========================================================================
# STAGE 4 - DUPLICATE RESOLUTION
# ===========================================================================

def dedupe(
    df: pd.DataFrame,
    subset: Optional[Sequence[str]] = None,
    keep: Union[str, bool] = "first",
    drop_indices: Optional[Sequence[int]] = None,
    dry_run: bool = False,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    return_dropped: bool = False,
    fig_width: float = 12.0,
    fig_height: float = 4.5,
    dpi: int = 110,
):
    """Stage 4: remove duplicate rows.

    `subset` is the list of columns considered when defining a duplicate.
    Default (None) uses all columns.

    `keep` follows pandas' convention:
        'first'  - keep the first occurrence (default)
        'last'   - keep the last occurrence
        False    - drop all duplicates (keep none)

    DAMA dimension: Uniqueness.

    Examples
    --------
    >>> dx.dedupe(df)
    >>> dx.dedupe(df, subset=['customer_id'])
    >>> dropped = dx.dedupe(df, return_dropped=True)
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)
    if keep not in ("first", "last", False):
        raise ValueError(f"'keep' must be 'first', 'last' or False, got {keep!r}")
    if subset is not None:
        missing_cols = [c for c in subset if c not in df.columns]
        if missing_cols:
            raise KeyError(f"subset references missing columns: {missing_cols}")

    n_before = len(df)
    dup_mask = df.duplicated(subset=subset, keep=False)   # marks ALL duplicates
    n_dup_rows = int(dup_mask.sum())
    n_unique_dup_groups = int(df.loc[dup_mask].drop_duplicates(subset=subset).shape[0])

    # If user supplied explicit indices, honor them.
    if drop_indices is not None:
        bad = [i for i in drop_indices if i not in df.index]
        if bad:
            raise KeyError(f"drop_indices references missing rows: {bad}")
        if dry_run:
            out = df.copy()
        else:
            out = df.drop(index=list(drop_indices)).copy()
    elif dry_run:
        out = df.copy()
    else:
        out = df.drop_duplicates(subset=subset, keep=keep).copy()
    n_after = len(out)
    n_removed = n_before - n_after

    # Sample of removed rows for visibility (max 10)
    if keep == "first":
        removed_mask = df.duplicated(subset=subset, keep="first")
    elif keep == "last":
        removed_mask = df.duplicated(subset=subset, keep="last")
    else:
        removed_mask = df.duplicated(subset=subset, keep=False)
    removed_rows = df.loc[removed_mask]

    if dry_run:
        n_would_drop = (len(drop_indices) if drop_indices is not None
                         else n_before - len(df.drop_duplicates(subset=subset, keep=keep)))
        decision = (f"DRY-RUN: would drop {n_would_drop} row(s). "
                    f"Set dry_run=False to actually execute. "
                    f"({n_dup_rows} rows in {n_unique_dup_groups} duplicate group(s)).")
    elif drop_indices is not None:
        decision = (f"Dropped {n_removed} row(s) at the user-supplied indices. "
                    f"Rows {n_before} -> {n_after}.")
    else:
        decision = (f"Removed {n_removed} duplicate row(s) "
                    f"({n_dup_rows} were part of {n_unique_dup_groups} duplicate "
                    f"group(s)).  Rows {n_before} -> {n_after}.")

    out.attrs = dict(df.attrs)
    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
    out.attrs[AUDIT_KEY].append({
        "stage": "duplicate_resolution",
        "function": "dedupe",
        "timestamp": now_iso(),
        "before": {"n_rows": n_before},
        "after":  {"n_rows": n_after, "n_removed": n_removed,
                   "n_duplicate_groups": n_unique_dup_groups},
        "params": {"subset": list(subset) if subset else None, "keep": keep},
        "decision": decision,
    })

    if show:
        _print_header(f"Duplicate resolution for: {df_name}")
        summary = pd.DataFrame({
            "value": [n_before, n_after, n_removed,
                      n_dup_rows, n_unique_dup_groups,
                      str(subset) if subset else "(all columns)",
                      str(keep)],
        }, index=[
            "rows_before", "rows_after", "rows_removed",
            "rows_in_duplicate_groups", "duplicate_groups",
            "subset", "keep",
        ])
        _display(_format_summary(summary, decimals,
                                  int_cols=("value",)))
        if n_removed and len(removed_rows):
            print("\nSample of removed duplicates (showing up to 5):")
            _display(removed_rows.head(5))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_dedupe(n_before, n_after, n_dup_rows, n_unique_dup_groups,
                            fig_width, fig_height, dpi)
    _finalize_figure(fig, show, plot, return_fig)

    if return_dropped:
        if return_df and return_fig: return out, removed_rows, fig
        if return_df: return out, removed_rows
        return removed_rows
    return _ret_pack(out, fig, return_df, return_fig)


def _plot_dedupe(n_before, n_after, n_dup_rows, n_groups, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    ax = axes[0]
    ax.bar(["before", "after"], [n_before, n_after],
           color=["#ec7853", "#2ca02c"], edgecolor="black")
    ax.set_ylabel("row count")
    ax.set_title("Rows before vs after dedup", fontweight="bold")
    for i, v in enumerate([n_before, n_after]):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=11)

    ax = axes[1]
    labels = ["unique\n(kept)", "duplicates\n(removed)", "groups"]
    values = [n_after, n_before - n_after, n_groups]
    colors = ["#2ca02c", "#c73e3e", "#f7c873"]
    ax.bar(labels, values, color=colors, edgecolor="black")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("count")
    ax.set_title("Duplicate breakdown", fontweight="bold")

    fig.suptitle("Duplicate resolution (Stage 4)", fontsize=14, fontweight="bold")
    return fig


# ===========================================================================
# STAGE 5 - OUTLIER TREATMENT
# ===========================================================================

_VALID_OUTLIER_METHODS_CLEAN = ("iqr", "zscore")
_VALID_OUTLIER_ACTIONS = ("clip", "drop")


def _clip_outliers_apply(df, params, decimals, df_name, show, plot,
                         return_df, return_params, return_fig,
                         fig_width, fig_height, dpi):
    """APPLY mode for clip_outliers: replay fitted bounds, never re-compute."""
    if (not isinstance(params, dict)
            or params.get("function") != "clip_outliers"):
        got = params.get("function") if isinstance(params, dict) else params
        raise ValueError(
            f"params dict is not for 'clip_outliers' (function={got!r}).")
    col_params = params.get("columns", {}) or {}
    method = params.get("method", "?")
    action = params.get("action", "clip")
    fitted_at = params.get("fit_at", "?")
    missing_cols = [c for c in col_params if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"clip_outliers apply failed: params expects column(s) "
            f"{missing_cols} which are not present in this DataFrame. "
            f"The data does not match the fitted bounds.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    n_before = len(out)
    per_col_log = []
    outlier_mask_total = pd.Series(False, index=out.index)
    for c, cp in col_params.items():
        s = pd.to_numeric(out[c], errors="coerce")
        lb, ub = cp["lower_bound"], cp["upper_bound"]
        col_mask = (s < lb) | (s > ub)
        n_out = int(col_mask.fillna(False).sum())
        if action == "clip":
            if n_out:
                out[c] = s.clip(lower=lb, upper=ub)
            n_clipped, n_dropped = n_out, 0
        else:
            outlier_mask_total = outlier_mask_total | col_mask.fillna(False)
            n_clipped, n_dropped = 0, n_out
        per_col_log.append({
            "column": c, "lower_bound": float(lb), "upper_bound": float(ub),
            "n_outliers": n_out, "n_clipped": n_clipped,
            "n_dropped_rows": n_dropped,
        })
    if action == "drop" and outlier_mask_total.any():
        out = out.loc[~outlier_mask_total].copy()
        out.attrs = dict(df.attrs)
    n_after = len(out)
    total_clipped = sum(e["n_clipped"] for e in per_col_log)
    rows_dropped = n_before - n_after
    did = (f"clipped {total_clipped} cell(s)" if action == "clip"
           else f"dropped {rows_dropped} row(s)")
    decision = (f"Applied saved {method} bounds (fitted {fitted_at}) to "
                f"{len(per_col_log)} column(s): {did} at fit-time bounds; "
                f"no re-fit -- leakage-safe.")
    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
    out.attrs[AUDIT_KEY].append({
        "stage": "outlier_treatment",
        "function": "clip_outliers",
        "timestamp": now_iso(),
        "mode": "apply",
        "before": {"n_rows": n_before},
        "after": {"n_rows": n_after, "cells_clipped": total_clipped,
                  "rows_dropped": rows_dropped},
        "params": {"method": method, "action": action,
                   "cols": list(col_params), "fit_at": fitted_at},
        "decision": decision,
    })
    if show:
        _print_header(f"Outlier treatment for: {df_name}  "
                      f"(method={method}, action={action}, mode=apply)")
        if per_col_log:
            log_df = pd.DataFrame(per_col_log).set_index("column")
            _display(_format_summary(
                log_df, decimals,
                int_cols=("n_outliers", "n_clipped", "n_dropped_rows")))
        print(f"\nDecision: {decision}\n")
    fig = None
    if plot:
        fig = _plot_clip_outliers(df, out, list(col_params), per_col_log,
                                  action, fig_width, fig_height, dpi,
                                  decimals)
    _finalize_figure(fig, show, plot, return_fig)
    return _ret_pack(out, fig, return_df, return_fig,
                     params=params, return_params=return_params)


def clip_outliers(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "iqr",
    k: float = 1.5,
    z_threshold: float = 3.0,
    action: str = "clip",
    dry_run: bool = False,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 14.0,
    fig_height: float = 6.0,
    dpi: int = 110,
    params: Optional[dict] = None,
    return_params: bool = False,
):
    """Stage 5: outlier treatment.

    Detection methods:
        'iqr'    - Tukey fence: LB = Q1 - k*IQR, UB = Q3 + k*IQR.
        'zscore' - |Z| > z_threshold.

    Actions:
        'clip'   - winsorization (replace with bound). Default; no row loss.
        'drop'   - drop rows where ANY analysed column is outlying.

    Leakage-safe fit/apply: in FIT mode (``params=None``) pass
    ``return_params=True`` to also get a replayable params dict with
    the per-column bounds computed from THIS data; in APPLY mode
    (``params=<dict>``) held-out data is clipped/dropped at those
    fit-time bounds verbatim -- never recomputed. Columns whose
    bounds could not be fitted (all-NaN, zero spread) are excluded
    from the params and left untouched on apply.

    DAMA dimension: Accuracy (correcting suspect values without inventing them).

    Examples
    --------
    >>> dx.clip_outliers(df)                              # IQR, k=1.5, clip
    >>> dx.clip_outliers(df, method='zscore', z_threshold=3)
    >>> dx.clip_outliers(df, action='drop')
    >>> tr, p = dx.clip_outliers(train, return_params=True)
    >>> te = dx.clip_outliers(test, params=p)   # train bounds, no re-fit
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)

    if params is not None:
        if dry_run:
            raise ValueError(
                "clip_outliers: dry_run is a fit-mode flag and cannot "
                "be combined with params= (apply mode).")
        return _clip_outliers_apply(
            df, params, decimals, df_name, show, plot, return_df,
            return_params, return_fig, fig_width, fig_height, dpi)
    if dry_run and return_params:
        raise ValueError(
            "clip_outliers: dry_run does not fit parameters; run with "
            "dry_run=False to get a replayable params dict.")

    if method not in _VALID_OUTLIER_METHODS_CLEAN:
        raise ValueError(f"'method' must be one of {_VALID_OUTLIER_METHODS_CLEAN}, got {method!r}")
    if action not in _VALID_OUTLIER_ACTIONS:
        raise ValueError(f"'action' must be one of {_VALID_OUTLIER_ACTIONS}, got {action!r}")
    if k <= 0 or z_threshold <= 0:
        raise ValueError("k and z_threshold must both be positive.")

    if cols is None:
        cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]
        if not cols:
            raise ValueError("No numeric columns to treat. Pass cols= explicitly.")
    else:
        cols = list(cols)
        bad = [c for c in cols if c not in df.columns]
        if bad:
            raise KeyError(f"cols references missing: {bad}")

    out = df.copy()
    out.attrs = dict(df.attrs)
    n_before = len(out)
    per_col_log = []
    outlier_mask_total = pd.Series(False, index=out.index)

    # Flag non-finite (inf) inputs with a named dextra warning, then suppress
    # the raw numpy overflow/invalid RuntimeWarnings while bounds are computed.
    # Clipping results are unchanged; this is a disclosure-only change.
    _inf_cols = []
    for _c in cols:
        _ser = df[_c]
        if isinstance(_ser, pd.DataFrame):
            continue
        if np.isinf(pd.to_numeric(_ser, errors="coerce")
                    .to_numpy(dtype="float64", na_value=np.nan)).any():
            _inf_cols.append(str(_c))
    if _inf_cols:
        warnings.warn(
            f"clip_outliers: column(s) {_inf_cols} contain non-finite (inf) "
            f"values; the computed bounds may overflow to inf. Replace them "
            f"first (e.g. df.replace([np.inf, -np.inf], np.nan)) if that is "
            f"not intended.", UserWarning, stacklevel=2)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for c in cols:
            s = pd.to_numeric(out[c], errors="coerce")
            non_na = s.dropna()
            if non_na.empty:
                continue
            if method == "iqr":
                q1 = non_na.quantile(0.25)
                q3 = non_na.quantile(0.75)
                iqr = q3 - q1
                lb = q1 - k * iqr
                ub = q3 + k * iqr
            else:
                mu, sigma = non_na.mean(), non_na.std()
                if sigma == 0 or pd.isna(sigma):
                    continue
                lb = mu - z_threshold * sigma
                ub = mu + z_threshold * sigma
            col_mask = (s < lb) | (s > ub)
            n_out = int(col_mask.fillna(False).sum())
            if action == "clip":
                if n_out:
                    out[c] = s.clip(lower=lb, upper=ub)
                n_clipped = n_out
                n_dropped = 0
            else:  # drop
                outlier_mask_total = outlier_mask_total | col_mask.fillna(False)
                n_clipped = 0
                n_dropped = n_out
            per_col_log.append({
                "column": c, "lower_bound": float(lb), "upper_bound": float(ub),
                "n_outliers": n_out, "n_clipped": n_clipped, "n_dropped_rows": n_dropped,
            })

    if dry_run:
        # Revert any clipping that was done in-place above
        out = df.copy()
        out.attrs = dict(df.attrs)
    elif action == "drop" and outlier_mask_total.any():
        out = out.loc[~outlier_mask_total].copy()
        out.attrs = dict(df.attrs)

    n_after = len(out)
    total_clipped = sum(e["n_clipped"] for e in per_col_log)
    rows_dropped = n_before - n_after

    if dry_run:
        decision = (f"DRY-RUN: would {action} {total_clipped if action=='clip' else rows_dropped} "
                    f"{'cell' if action=='clip' else 'row'}(s). Set dry_run=False to execute.")
    elif action == "clip":
        decision = (f"Clipped {total_clipped} cell(s) across {len(per_col_log)} "
                    f"column(s) using {method} (k={k} / z={z_threshold}).")
    else:
        decision = (f"Dropped {rows_dropped} row(s) flagged by {method} "
                    f"across {len(per_col_log)} column(s).")
    if return_params:
        decision += (" Fitted bounds saved; apply to held-out data "
                     "with clip_outliers(df_test, params=...).")

    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
    out.attrs[AUDIT_KEY].append({
        "stage": "outlier_treatment",
        "function": "clip_outliers",
        "timestamp": now_iso(),
        "before": {"n_rows": n_before},
        "after":  {"n_rows": n_after, "cells_clipped": total_clipped,
                   "rows_dropped": rows_dropped},
        "params": {"method": method, "k": k, "z_threshold": z_threshold,
                   "action": action, "cols": list(cols)},
        "decision": decision,
    })
    params_out = None
    if return_params:
        out.attrs[AUDIT_KEY][-1]["mode"] = "fit"
        params_out = json_safe({
            "function": "clip_outliers",
            "method": method, "k": k, "z_threshold": z_threshold,
            "action": action,
            "version": __version__,
            "fit_at": now_iso(),
            "columns": {e["column"]: {
                "lower_bound": e["lower_bound"],
                "upper_bound": e["upper_bound"]} for e in per_col_log},
            "metadata": {"cols": list(cols)},
        })

    if show:
        _print_header(f"Outlier treatment for: {df_name}  "
                      f"(method={method}, action={action})")
        log_df = pd.DataFrame(per_col_log).set_index("column")
        _display(_format_summary(
            log_df, decimals,
            int_cols=("n_outliers", "n_clipped", "n_dropped_rows"),
        ))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_clip_outliers(df, out, cols, per_col_log, action,
                                    fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    return _ret_pack(out, fig, return_df, return_fig,
                     params=params_out, return_params=return_params)


def _plot_clip_outliers(df_before, df_after, cols, per_col_log, action,
                        fig_width, fig_height, dpi, decimals):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)

    # Panel 1: cells changed / rows dropped per column
    ax = axes[0]
    s_log = pd.DataFrame(per_col_log).set_index("column")
    metric_col = "n_clipped" if action == "clip" else "n_dropped_rows"
    s = s_log[metric_col].sort_values(ascending=True)
    ax.barh(s.index.astype(str), s.values,
            color="#ec7853", edgecolor="black", linewidth=0.4)
    ax.set_xlabel(f"# {metric_col}")
    ax.set_title(f"Outliers per column ({action})", fontweight="bold")
    for i, v in enumerate(s.values):
        if v > 0:
            ax.text(v + max(s.values) * 0.01, i, f"{int(v):,}",
                    va="center", fontsize=9)

    # Panel 2: before-vs-after distribution for the first numeric column
    ax = axes[1]
    if cols:
        c = cols[0]
        before = pd.to_numeric(df_before[c], errors="coerce").dropna()
        after = pd.to_numeric(df_after[c], errors="coerce").dropna()
        bins = max(20, int(np.sqrt(len(before))))
        ax.hist(before, bins=bins, alpha=0.5, label="before",
                color="#ec7853", edgecolor="black")
        ax.hist(after, bins=bins, alpha=0.5, label="after",
                color="#2ca02c", edgecolor="black")
        ax.set_title(f"Distribution of '{c}' before vs after",
                     fontweight="bold")
        ax.set_xlabel(c)
        ax.set_ylabel("count")
        ax.legend()
    else:
        ax.set_axis_off()

    fig.suptitle("Outlier treatment (Stage 5)", fontsize=14, fontweight="bold")
    return fig


# Stage 3.3: `dedup` is the official short alias; `fillna_smart` /
# `clipout` are deprecated -- see module __getattr__ (audit #10).
dedup = dedupe


# ===========================================================================
# v2 INSPECTORS — Stage 3.4
# ===========================================================================
# All inspectors are READ-ONLY: they never mutate the input, never write to
# df.attrs, and always return a diagnostic DataFrame for the user to filter
# further with native pandas.
# ===========================================================================

# --- na_show: missing-value inspector ------------------------------------

def na_show(
    df: pd.DataFrame,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    max_rows_shown: int = 10,
    fig_width: float = 14.0,
    fig_height: float = 5.5,
    dpi: int = 110,
):
    """Inspector: show rows with missing values without modifying the data.

    Pure / read-only. Returns the rows that contain at least one NaN with
    two extra columns:
      - ``which_cols_missing``: comma-separated list of columns with NaN.
      - ``n_missing_in_row``: count of NaN cells in the row.

    Also prints a per-column profile with a recommended strategy.

    DAMA dimension: Completeness.

    Examples
    --------
    >>> view = dx.na_show(df)
    >>> # decide a strategy from the suggestions, then:
    >>> df_clean = dx.impute(df, strategy='median')
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)

    n_total = len(df)
    col_rows = []
    for c in df.columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        pct = (n_missing / n_total * 100) if n_total else 0.0
        if n_missing == 0:
            suggested = "keep"
        else:
            suggested = _auto_strategy_for_column(s)
        col_rows.append({
            "column": c, "dtype": str(s.dtype),
            "n_missing": n_missing, "pct_missing": pct,
            "suggested_strategy": suggested,
        })
    col_summary = pd.DataFrame(col_rows).set_index("column")

    row_mask = df.isna().any(axis=1)
    if row_mask.any():
        diag = df.loc[row_mask].copy()
        diag.insert(
            len(diag.columns), "which_cols_missing",
            df.loc[row_mask].apply(
                lambda r: ", ".join(c for c in df.columns if pd.isna(r[c])),
                axis=1,
            ).values,
        )
        diag.insert(
            len(diag.columns), "n_missing_in_row",
            df.loc[row_mask].isna().sum(axis=1).astype(int).values,
        )
    else:
        diag = df.iloc[0:0].copy()

    n_bad_rows = int(row_mask.sum())
    n_bad_cols = int((df.isna().sum() > 0).sum())

    if show:
        _print_header(f"Missing-value inspection for: {df_name}")
        print("\nPer-column profile:")
        _display(_format_summary(col_summary, decimals,
                                  int_cols=("n_missing",),
                                  pct_cols=("pct_missing",)))
        if n_bad_rows:
            n_show = min(max_rows_shown, n_bad_rows)
            print(f"\nFirst {n_show} of {n_bad_rows} rows with missing values:")
            _display(diag.head(n_show))
            recs = col_summary.loc[col_summary["n_missing"] > 0,
                                    "suggested_strategy"].to_dict()
            print(f"\nDecision: {n_bad_rows} row(s) across {n_bad_cols} column(s) "
                  f"have missing values. Suggested per-column strategies: {recs}. "
                  f"Call `dx.impute(df, strategy=...)` once you have decided.\n")
        else:
            print("\nDecision: No missing values detected. Data is complete.\n")

    fig = None
    if plot:
        fig = _plot_na_show(df, col_summary, fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return diag, fig
    if return_df: return diag
    if return_fig: return fig
    return None


def _plot_na_show(df, col_summary, fig_width, fig_height, dpi, decimals):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi,
                              gridspec_kw={"width_ratios": [1, 2]})
    # Panel 1: bar chart of pct_missing
    ax = axes[0]
    s = col_summary["pct_missing"].sort_values(ascending=True)
    colors = ["#2ca02c" if v == 0 else
              "#9bd09a" if v < 5 else
              "#f7c873" if v < 30 else
              "#ec7853" if v < 60 else "#c73e3e" for v in s.values]
    ax.barh(s.index.astype(str), s.values, color=colors,
            edgecolor="black", linewidth=0.4)
    ax.set_xlabel("% missing")
    ax.set_title("Missing per column", fontweight="bold")
    for i, v in enumerate(s.values):
        if v > 0:
            ax.text(v + 0.5, i, f"{v:.{decimals}f}%",
                    va="center", fontsize=8)

    # Panel 2: missingness pattern heatmap (subsampled)
    ax = axes[1]
    mask = df.isna()
    if mask.shape[0] > 200:
        step = max(1, mask.shape[0] // 200)
        mask = mask.iloc[::step]
    sns.heatmap(mask.values.T, ax=ax, cmap="Greys", cbar=False,
                yticklabels=df.columns, xticklabels=False)
    ax.set_title("Missingness pattern (rows → cols)", fontweight="bold")
    ax.set_xlabel("rows (subsampled if >200)")

    fig.suptitle("na_show — Missing-value inspection", fontsize=13, fontweight="bold")
    return fig


# --- dup_show: duplicate inspector ----------------------------------------

def dup_show(
    df: pd.DataFrame,
    subset: Optional[Sequence[str]] = None,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    max_rows_shown: int = 20,
    fig_width: float = 12.0,
    fig_height: float = 5.0,
    dpi: int = 110,
):
    """Inspector: show duplicate rows grouped side-by-side.

    Pure / read-only. Returns the duplicate rows with three extra columns:
      - ``dup_group_id``: integer ID per duplicate group.
      - ``group_size``: number of rows in the group.
      - ``is_first_in_group``: True for the first occurrence.

    The result is sorted by group so the user can decide which rows to
    drop and call ``dx.dedup(df, drop_indices=[...])``.

    DAMA dimension: Uniqueness.

    Examples
    --------
    >>> view = dx.dup_show(df, subset=['customer_id'])
    >>> # inspect, decide, then:
    >>> df_clean = dx.dedup(df, drop_indices=view.index[view['is_first_in_group']==False])
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)
    if subset is not None:
        bad = [c for c in subset if c not in df.columns]
        if bad:
            raise KeyError(f"subset references missing columns: {bad}")

    dup_mask = df.duplicated(subset=subset, keep=False)
    n_dup_rows = int(dup_mask.sum())

    if n_dup_rows == 0:
        diag = df.iloc[0:0].copy()
        n_groups = 0
        group_sizes = pd.Series(dtype=int)
    else:
        if subset is None:
            key_series = df.apply(tuple, axis=1)
        else:
            key_series = df[list(subset)].apply(tuple, axis=1)
        key_in_dup = key_series.loc[dup_mask]
        group_ids = pd.factorize(key_in_dup)[0] + 1
        diag = df.loc[dup_mask].copy()
        diag.insert(len(diag.columns), "dup_group_id", group_ids)
        sizes = pd.Series(group_ids).value_counts().sort_index()
        diag.insert(len(diag.columns), "group_size",
                    diag["dup_group_id"].map(sizes).values)
        # is_first_in_group flag
        first_idx_per_group = diag.groupby("dup_group_id").head(1).index
        diag.insert(len(diag.columns), "is_first_in_group",
                    diag.index.isin(first_idx_per_group))
        diag = diag.sort_values("dup_group_id")
        n_groups = int(diag["dup_group_id"].nunique())
        group_sizes = sizes

    if show:
        suf = f" (subset={list(subset)})" if subset else ""
        _print_header(f"Duplicate inspection for: {df_name}{suf}")
        if n_dup_rows:
            n_show = min(max_rows_shown, n_dup_rows)
            print(f"\nFirst {n_show} of {n_dup_rows} duplicate row(s) in {n_groups} group(s):")
            _display(diag.head(n_show))
            print(f"\nDecision: {n_dup_rows} duplicate row(s) in {n_groups} group(s). "
                  f"Either call `dx.dedup(df, drop_indices=[...])` after choosing rows, "
                  f"or `dx.dedup(df, keep='first'|'last')` for a bulk action.\n")
        else:
            print("\nDecision: No duplicate rows found.\n")

    fig = None
    if plot and n_dup_rows > 0:
        fig = _plot_dup_show(group_sizes, fig_width, fig_height, dpi)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return diag, fig
    if return_df: return diag
    if return_fig: return fig
    return None


def _plot_dup_show(group_sizes, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    # Panel 1: distribution of group sizes
    ax = axes[0]
    counts = group_sizes.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values,
           color="#ec7853", edgecolor="black")
    ax.set_xlabel("group size (# rows per duplicate)")
    ax.set_ylabel("# groups")
    ax.set_title("Distribution of duplicate group sizes", fontweight="bold")

    # Panel 2: top largest groups
    ax = axes[1]
    top = group_sizes.sort_values(ascending=False).head(10)
    ax.barh([f"group {gid}" for gid in top.index][::-1],
            top.values[::-1], color="#c73e3e", edgecolor="black")
    ax.set_xlabel("group size")
    ax.set_title("Top 10 largest duplicate groups", fontweight="bold")

    fig.suptitle("dup_show — Duplicate inspection",
                 fontsize=13, fontweight="bold")
    return fig


# --- out_show: outlier inspector ------------------------------------------

def out_show(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "iqr",
    k: float = 1.5,
    z_threshold: float = 3.0,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    max_rows_shown: int = 10,
    fig_width: float = 14.0,
    fig_height: float = 5.5,
    dpi: int = 110,
):
    """Inspector: show rows containing outlying values.

    Pure / read-only. Returns the outlying rows with three extra columns:
      - ``outlier_in_columns``: comma-separated list of columns where the
        row is outlying.
      - ``severity_z``: maximum |z| across analysed columns.
      - ``severity_iqr``: maximum normalised distance beyond the IQR fence.

    Sorted by ``severity_z`` descending so the user sees the worst offenders
    first.

    DAMA dimension: Accuracy.

    Examples
    --------
    >>> view = dx.out_show(df, cols=['price', 'age'])
    >>> # decide and apply:
    >>> df_clean = dx.clip_outliers(df, cols=['price'], action='clip')
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    df = _ensure_pandas(df)
    if method not in ("iqr", "zscore"):
        raise ValueError(f"method must be 'iqr' or 'zscore', got {method!r}")

    if cols is None:
        cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])]
    else:
        cols = list(cols)
        bad = [c for c in cols if c not in df.columns]
        if bad:
            raise KeyError(f"cols references missing: {bad}")
    if not cols:
        if show:
            _print_header(f"Outlier inspection for: {df_name}")
            print("\nDecision: No numeric columns to inspect.\n")
        return None

    col_rows = []
    outlier_mask_total = pd.Series(False, index=df.index)
    z_per_col = {}
    iqr_dist_per_col = {}
    bounds_per_col = {}

    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        non_na = s.dropna()
        if non_na.empty:
            continue
        if method == "iqr":
            q1, q3 = non_na.quantile(0.25), non_na.quantile(0.75)
            iqr = q3 - q1
            lb = q1 - k * iqr
            ub = q3 + k * iqr
            iqr_dist = pd.Series(0.0, index=s.index)
            if iqr > 0:
                iqr_dist = ((s - q3).clip(lower=0) + (q1 - s).clip(lower=0)) / iqr
        else:
            mu, sigma = non_na.mean(), non_na.std()
            if sigma == 0 or pd.isna(sigma):
                continue
            lb = mu - z_threshold * sigma
            ub = mu + z_threshold * sigma
            iqr_dist = pd.Series(0.0, index=s.index)
        mask_c = (s < lb) | (s > ub)
        mu, sigma = non_na.mean(), non_na.std()
        z_per_col[c] = ((s - mu) / sigma) if sigma > 0 else pd.Series(0.0, index=s.index)
        iqr_dist_per_col[c] = iqr_dist
        bounds_per_col[c] = (float(lb), float(ub))
        outlier_mask_total = outlier_mask_total | mask_c.fillna(False)
        col_rows.append({
            "column": c, "lower_bound": float(lb),
            "upper_bound": float(ub),
            "n_outliers": int(mask_c.fillna(False).sum()),
        })

    col_summary = (pd.DataFrame(col_rows).set_index("column")
                    if col_rows else pd.DataFrame())

    if not outlier_mask_total.any():
        if show:
            _print_header(f"Outlier inspection for: {df_name}  (method={method})")
            print("\nPer-column bounds:")
            if not col_summary.empty:
                _display(_format_summary(col_summary, decimals,
                                          int_cols=("n_outliers",)))
            print("\nDecision: No outliers detected with the current bounds.\n")
        if return_df: return df.iloc[0:0].copy()
        return None

    diag = df.loc[outlier_mask_total].copy()
    out_cols_per_row = []
    sev_z = []
    sev_iqr = []
    for idx in diag.index:
        outlying_here = []
        for c, (lb, ub) in bounds_per_col.items():
            v = pd.to_numeric(pd.Series([df.loc[idx, c]]),
                               errors="coerce").iloc[0]
            if pd.notna(v) and (v < lb or v > ub):
                outlying_here.append(c)
        out_cols_per_row.append(", ".join(outlying_here))
        zs = [abs(z_per_col[c].loc[idx]) for c in z_per_col
              if pd.notna(z_per_col[c].loc[idx])]
        iqrs = [iqr_dist_per_col[c].loc[idx] for c in iqr_dist_per_col
                if pd.notna(iqr_dist_per_col[c].loc[idx])]
        sev_z.append(max(zs) if zs else float("nan"))
        sev_iqr.append(max(iqrs) if iqrs else float("nan"))
    diag.insert(len(diag.columns), "outlier_in_columns", out_cols_per_row)
    diag.insert(len(diag.columns), "severity_z", sev_z)
    diag.insert(len(diag.columns), "severity_iqr", sev_iqr)
    diag = diag.sort_values("severity_z", ascending=False, na_position="last")
    n_outlier_rows = len(diag)

    if show:
        _print_header(f"Outlier inspection for: {df_name}  (method={method})")
        print("\nPer-column bounds:")
        _display(_format_summary(col_summary, decimals,
                                  int_cols=("n_outliers",)))
        n_show = min(max_rows_shown, n_outlier_rows)
        print(f"\nTop {n_show} of {n_outlier_rows} outlier rows (sorted by |z| desc):")
        _display(diag.head(n_show))
        print(f"\nDecision: {n_outlier_rows} outlier row(s) detected across "
              f"{len(col_summary)} column(s). Options: "
              f"`dx.clip_outliers(df, action='clip')` to winsorise, "
              f"`dx.clip_outliers(df, action='drop')` to remove the rows, or leave as is.\n")

    fig = None
    if plot:
        fig = _plot_out_show(df, cols, bounds_per_col,
                              fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return diag, fig
    if return_df: return diag
    if return_fig: return fig
    return None


def _plot_out_show(df, cols, bounds_per_col, fig_width, fig_height, dpi, decimals):
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, fig_height), dpi=dpi)
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, cols):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        lb, ub = bounds_per_col.get(c, (None, None))
        if lb is None:
            ax.set_axis_off(); continue
        is_out = (s < lb) | (s > ub)
        jitter_in = np.random.uniform(-0.02, 0.02, (~is_out).sum())
        jitter_out = np.random.uniform(-0.02, 0.02, is_out.sum())
        ax.scatter(s[~is_out], jitter_in, color="steelblue", s=18,
                   alpha=0.5, label="inlier")
        ax.scatter(s[is_out], jitter_out, color="red", s=28,
                   alpha=0.9, edgecolor="black", label="outlier")
        ax.axvline(lb, color="orange", linestyle="--", linewidth=1.3)
        ax.axvline(ub, color="orange", linestyle="--", linewidth=1.3)
        ax.set_title(f"{c}", fontweight="bold")
        ax.set_xlabel(c); ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("out_show — Outlier inspection",
                 fontsize=13, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# v2 short aliases (alongside v1 long names which still work)
# ---------------------------------------------------------------------------

# Inspectors (new, no v1 equivalent)
# na_show, dup_show, out_show — already defined above

# (v2 actor aliases `na_fix` / `dup_fix` / `out_fix` / `rule_check` /
# `type_fix` / `col_clean` / `col_fix` / `clean_rep` are deprecated --
# see module __getattr__, audit #10.)

# ---------------------------------------------------------------------------
# Professional underscore-free aliases (polars/tidyverse competitive)
# ---------------------------------------------------------------------------
# Each name is a single word or compound-no-underscore, chosen for clarity
# against famous data-science libraries.

# Inspectors
audit    = clean_report          # master audit
nascan   = na_show               # scan for missing values
dupscan  = dup_show              # scan for duplicates
outscan  = out_show              # scan for outliers

# Actors
tidycols = standardize_columns   # tidyverse-inspired column cleanup
recast   = cast_types            # recast dtypes
impute   = handle_missing        # academic term for filling missing
# dedup is already defined above as v1 alias
winsor   = clip_outliers         # winsorize outliers (academic term)
verify   = validate_rules        # generic verify


# ---------------------------------------------------------------------------
# Deprecated aliases (audit #10): ONE official short alias per function
# (`audit`, `tidycols`, `recast`, `verify`, `impute`, `dedup`, `winsor`);
# the older synonyms below still resolve -- to the SAME function object, so
# identity checks keep holding -- but emit a DeprecationWarning (PEP 562).
# ---------------------------------------------------------------------------

_DEPRECATED_ALIASES = {
    "cleanrep": "clean_report", "clean_rep": "clean_report",
    "stdcols": "standardize_columns", "col_clean": "standardize_columns",
    "col_fix": "standardize_columns",
    "cast": "cast_types", "type_fix": "cast_types",
    "vrules": "validate_rules", "rule_check": "validate_rules",
    "fillna_smart": "handle_missing", "na_fix": "handle_missing",
    "dup_fix": "dedupe",
    "clipout": "clip_outliers", "out_fix": "clip_outliers",
}

_OFFICIAL_SHORT = {
    "clean_report": "audit", "standardize_columns": "tidycols",
    "cast_types": "recast", "validate_rules": "verify",
    "handle_missing": "impute", "dedupe": "dedup",
    "clip_outliers": "winsor",
}


def __getattr__(name):
    canonical = _DEPRECATED_ALIASES.get(name)
    if canonical is not None:
        warnings.warn(
            f"dextra: '{name}' is a deprecated alias of '{canonical}'; use "
            f"'{_OFFICIAL_SHORT[canonical]}' (or '{canonical}') instead.",
            DeprecationWarning, stacklevel=2)
        return globals()[canonical]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
