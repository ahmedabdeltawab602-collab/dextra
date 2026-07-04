"""Plotting helpers built on top of :mod:`matplotlib`, :mod:`seaborn` and
:mod:`plotly`.

Two public entry points live here:

* :func:`plot_histograms` — one row per column, histogram + KDE on the left
  and a tidy statistical summary on the right. Matplotlib / Seaborn based.
* :func:`plot_boxplots` — stacked, horizontal box-plots with annotated
  summary statistics. Plotly based, interactive.

The legacy aliases ``hister`` and ``boxpl`` remain exported for backward
compatibility.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Mapping, Optional, Sequence, Tuple, Union

# Matplotlib / seaborn are used lazily so importing dextra doesn't open a
# GUI backend unexpectedly.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import (
    DEFAULT_BOX_COLORS,
    append_audit,
    format_value,
    get_variable_name,
    json_safe,
    now_iso,
    resolve_columns,
    safe_divide,
    to_numeric_frame,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, not a runtime dependency
    import plotly.graph_objects as go

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Shared summary used by both plot functions
# ---------------------------------------------------------------------------

def _require_plotly():
    """Import Plotly lazily so it is an optional (viz) dependency, not core.

    Only :func:`plot_boxplots` needs Plotly; importing dextra (or using the
    matplotlib-based helpers) must not require it. Raises a helpful error if
    the optional dependency is missing.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:  # pragma: no cover - exercised only without plotly
        raise ImportError(
            "plot_boxplots requires Plotly, an optional dependency. Install it "
            "with `pip install plotly` (or `pip install pydextra[viz]`)."
        ) from exc
    return go, make_subplots


def _summary_frame(
    sub: pd.DataFrame, iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """Return one row per column with the metrics needed by the plots."""
    mean_ = sub.mean()
    std_ = sub.std()
    var_ = sub.var()
    cv_ = safe_divide(std_, mean_.replace(0, np.nan).abs()) * 100
    min_ = sub.min()
    max_ = sub.max()

    q1 = sub.quantile(0.25)
    q2 = sub.quantile(0.50)
    q3 = sub.quantile(0.75)
    iqr = q3 - q1
    lb = q1 - iqr_multiplier * iqr
    ub = q3 + iqr_multiplier * iqr

    mask = sub.lt(lb) | sub.gt(ub)
    out_count = mask.sum()
    count_ = sub.count()
    missing = pd.Series(len(sub), index=sub.columns) - count_
    out_pct = safe_divide(out_count, count_) * 100

    skew_ = sub.skew()
    kurt_ = sub.kurt()
    nunique_ = sub.nunique(dropna=True)

    modes: dict = {}
    for c in sub.columns:
        s = sub[c].dropna()
        modes[c] = [] if s.empty else s.mode(dropna=True).tolist()

    out = pd.DataFrame({
        "mean": mean_,
        "std": std_,
        "variance": var_,
        "cv_pct": cv_,
        "min": min_,
        "q1": q1,
        "median": q2,
        "q3": q3,
        "max": max_,
        "iqr": iqr,
        "lb": lb,
        "ub": ub,
        "outliers_count": out_count,
        "outliers_pct": out_pct,
        "non_na_count": count_,
        "missing_count": missing,
        "unique_count": nunique_,
        "skew": skew_,
        "kurt": kurt_,
    })
    out["mode"] = pd.Series(modes, dtype=object)
    return out.sort_index()


# ---------------------------------------------------------------------------
# Histograms with adjacent stats
# ---------------------------------------------------------------------------

def plot_histograms(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    bins: Union[int, str] = "auto",
    decimals: int = 2,
    iqr_multiplier: float = 1.5,
    fig_width: float = 17.0,
    fig_row_height: float = 4.8,
    width_ratios: Sequence[float] = (3, 1),
    dpi: int = 120,
    hist_color: str = "skyblue",
    hist_edgecolor: str = "black",
    alpha: float = 0.85,
    kde: bool = True,
    kde_color: str = "blue",
    kde_linewidth: float = 2.2,
    title: str = "Histograms with adjacent statistical summary",
    save: bool = False,
    output_dir: str = "plots",
    filename: str = "histograms_with_summary.png",
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_fig: bool = False,
    return_df: bool = False,
    return_params: bool = False,
    params: Optional[dict] = None,
) -> Union[None, plt.Figure, pd.DataFrame, Tuple[plt.Figure, pd.DataFrame]]:
    """Draw a histogram + KDE for each selected column with a side-panel summary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    cols : sequence of str, optional
        Columns to plot. Defaults to every numeric column in ``df``.
    bins : int, default ``20``
        Number of histogram bins.
    decimals : int, default ``2``
        Number of fractional digits in the side-panel summary.
    iqr_multiplier : float, default ``1.5``
        Multiplier on IQR for the outlier bounds reported in the side panel.
    fig_width, fig_row_height : float
        Figure dimensions, in inches. Total height = ``fig_row_height * n_rows``.
    width_ratios : sequence of float, default ``(3, 1)``
        Relative widths of the plot column and the text column.
    dpi : int, default ``120``
        Figure resolution.
    hist_color, hist_edgecolor, alpha : str, str, float
        Histogram style.
    kde : bool, default ``True``
        Whether to overlay a kernel-density estimate.
    kde_color, kde_linewidth : str, float
        KDE style.
    title : str
        Figure-level super-title.
    save : bool, default ``False``
        If ``True``, the figure is written to ``<output_dir>/<filename>``.
    output_dir, filename : str
        Path used when ``save=True``.
    show : bool, default ``True``
        If ``True``, the figure is displayed with ``plt.show()``.
    return_fig, return_df : bool, default ``False``
        Whether to return the Matplotlib ``Figure``, the summary DataFrame,
        both, or neither.

    Returns
    -------
    Figure, DataFrame, tuple of (Figure, DataFrame), or None
        Depends on ``return_fig`` and ``return_df``.

    Examples
    --------
    >>> dx.plot_histograms(df)
    >>> dx.hister(df, cols=["income"], bins=30)
    """
    if params is not None:
        _cfg = params.get("params", params)
        cols = _cfg.get("cols", cols)
        bins = _cfg.get("bins", bins)
        decimals = _cfg.get("decimals", decimals)
        iqr_multiplier = _cfg.get("iqr_multiplier", iqr_multiplier)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    if isinstance(bins, int) and bins <= 0:
        raise ValueError(f"'bins' must be a positive integer, got {bins}")
    if decimals < 0:
        raise ValueError(f"'decimals' must be >= 0, got {decimals}")

    cols_resolved = resolve_columns(df, cols, numeric_only=True)
    if not cols_resolved:
        raise ValueError("No columns to plot.")

    sub = to_numeric_frame(df[cols_resolved].copy())
    summary = _summary_frame(sub, iqr_multiplier=iqr_multiplier)

    n_rows = len(cols_resolved)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(fig_width, fig_row_height * n_rows),
        gridspec_kw={"width_ratios": list(width_ratios)},
        dpi=dpi,
    )
    # Normalise axes to 2D array so we can always index axes[i, 0] / axes[i, 1]
    if n_rows == 1:
        axes = np.atleast_2d(axes)

    for i, col in enumerate(cols_resolved):
        ax_hist, ax_text = axes[i, 0], axes[i, 1]
        series = sub[col].dropna()

        if series.empty:
            ax_hist.text(
                0.5, 0.5, f"No numeric data for '{col}'",
                ha="center", va="center", fontsize=12,
            )
            ax_hist.set_axis_off()
            ax_text.set_axis_off()
            continue

        sns.histplot(
            x=series,
            bins=bins,
            color=hist_color,
            edgecolor=hist_edgecolor,
            stat="density",
            alpha=alpha,
            ax=ax_hist,
        )
        if kde and len(series) > 1:
            sns.kdeplot(
                series,
                ax=ax_hist,
                linewidth=kde_linewidth,
                color=kde_color,
            )

        mean_val = summary.loc[col, "mean"]
        median_val = summary.loc[col, "median"]

        if pd.notna(mean_val):
            ax_hist.axvline(
                mean_val, color="red", linestyle="--",
                linewidth=2.0, label="Mean",
            )
        if pd.notna(median_val):
            ax_hist.axvline(
                median_val, color="green", linestyle="-.",
                linewidth=1.8, label="Median",
            )

        ax_hist.set_title(f"Histogram of {col}", fontsize=13, fontweight="bold")
        ax_hist.set_xlabel(col, fontsize=11)
        ax_hist.set_ylabel("Density", fontsize=11)
        ax_hist.legend(loc="upper right", fontsize=9)

        lines: List[str] = []

        def _add(label: str, key: str, kind: str, col=col, lines=lines) -> None:
            val = summary.loc[col, key] if key in summary.columns else None
            lines.append(f"{label:<15}: {format_value(val, kind, decimals)}")

        _add("count", "non_na_count", "int")
        _add("unique_count", "unique_count", "int")
        _add("missing_count", "missing_count", "int")
        _add("mean", "mean", "num")
        _add("std", "std", "num")
        _add("variance", "variance", "num")
        _add("cv_pct", "cv_pct", "pct")
        _add("min", "min", "num")
        _add("q1", "q1", "num")
        _add("median", "median", "num")
        _add("q3", "q3", "num")
        _add("max", "max", "num")
        _add("IQR", "iqr", "num")
        _add("outliers_count", "outliers_count", "int")
        _add("outliers_pct", "outliers_pct", "pct")
        _add("skew", "skew", "num")
        _add("kurt", "kurt", "num")

        ax_text.set_axis_off()
        ax_text.text(
            0.02, 0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    # Suptitle FIRST so tight_layout can reserve space for it.
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"saved: {out_path}")

    if show and plot:
        plt.show()
    elif not return_fig:
        plt.close(fig)

    config = {
        "cols": list(cols_resolved),
        "bins": bins,
        "decimals": decimals,
        "iqr_multiplier": iqr_multiplier,
    }
    audit_entry = {
        "stage": "phase1-eda",
        "function": "plot_histograms",
        "timestamp": now_iso(),
        "df_name": df_name,
        "params": config,
        "decision": (
            f"Plotted histograms for {len(cols_resolved)} column(s) of "
            f"'{df_name}'."
        ),
    }
    append_audit(summary, audit_entry)
    manifest = {
        "stage": "phase1-eda",
        "function": "plot_histograms",
        "df_name": df_name,
        "params": config,
        "summary": json_safe(summary.to_dict()),
        "dextra_audit": list(summary.attrs.get("dextra_audit", [])),
    }
    # Plotters are figure-first historically; keep (fig, df) order and append
    # the params manifest last so existing unpacking stays valid.
    results = []
    if return_fig:
        results.append(fig)
    if return_df:
        results.append(summary)
    if return_params:
        results.append(manifest)
    if not results:
        return None
    return results[0] if len(results) == 1 else tuple(results)


# ---------------------------------------------------------------------------
# Horizontal box-plots (Plotly)
# ---------------------------------------------------------------------------

def _build_color_map(
    cols_resolved: Sequence[str],
    colors: Optional[Union[Sequence[str], Mapping[str, str]]],
) -> dict:
    """Resolve the colour argument into a ``{column: colour}`` mapping."""
    n = len(cols_resolved)
    palette = list(DEFAULT_BOX_COLORS)
    if n > len(palette):
        repeats = (n + len(palette) - 1) // len(palette)
        palette = (palette * repeats)[:n]

    if colors is None:
        return {col: palette[i] for i, col in enumerate(cols_resolved)}
    if isinstance(colors, Mapping):
        return {
            col: colors.get(col, palette[i])
            for i, col in enumerate(cols_resolved)
        }

    # Treat anything else as an iterable of colours; repeat if too short.
    palette_in = list(colors)
    if not palette_in:
        return {col: palette[i] for i, col in enumerate(cols_resolved)}
    if n > len(palette_in):
        repeats = (n + len(palette_in) - 1) // len(palette_in)
        palette_in = (palette_in * repeats)[:n]
    return {col: palette_in[i] for i, col in enumerate(cols_resolved)}


def plot_boxplots(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    decimals: int = 2,
    iqr_multiplier: float = 1.5,
    width: int = 1400,
    row_height: int = 350,
    opacity: float = 0.7,
    line_color: str = "orange",
    template: str = "plotly_white",
    show_grid: bool = True,
    title: str = "Boxplots",
    colors: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_fig: bool = False,
    return_df: bool = False,
    return_params: bool = False,
    params: Optional[dict] = None,
) -> Union[None, go.Figure, pd.DataFrame, Tuple[go.Figure, pd.DataFrame]]:
    """Stacked horizontal box-plots with annotated statistics (Plotly).

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.
    cols : sequence of str, optional
        Columns to plot. Defaults to every numeric column in ``df``.
    decimals : int, default ``2``
        Fractional digits used in hover text and annotation labels.
    iqr_multiplier : float, default ``1.5``
        Multiplier used to derive the dashed outlier bounds.
    width, row_height : int
        Figure dimensions, in pixels. Total height = ``row_height * n_rows``.
    opacity : float
        Marker / box opacity.
    line_color : str
        Colour of the dashed outlier-bound lines.
    template : str
        Any Plotly figure template (e.g. ``"plotly_white"``, ``"plotly_dark"``).
    show_grid : bool
        Whether to draw the x-axis grid.
    title : str
        Top-level figure title.
    colors : sequence or mapping of colour strings, optional
        Either a list that will be zipped against ``cols``, or a mapping
        ``{column: colour}`` with an entry per column. Unspecified columns
        fall back to :data:`dextra._utils.DEFAULT_BOX_COLORS`.
    show, return_fig, return_df : bool
        Same semantics as :func:`plot_histograms`.

    Returns
    -------
    Figure, DataFrame, tuple of (Figure, DataFrame), or None

    Examples
    --------
    >>> dx.plot_boxplots(df)
    >>> fig, stats = dx.boxpl(df, return_fig=True, return_df=True, show=False)
    """
    if params is not None:
        _cfg = params.get("params", params)
        cols = _cfg.get("cols", cols)
        decimals = _cfg.get("decimals", decimals)
        iqr_multiplier = _cfg.get("iqr_multiplier", iqr_multiplier)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    if decimals < 0:
        raise ValueError(f"'decimals' must be >= 0, got {decimals}")

    cols_resolved = resolve_columns(df, cols, numeric_only=True)
    if not cols_resolved:
        raise ValueError("No columns to plot.")

    sub = to_numeric_frame(df[cols_resolved].copy())
    n_rows = len(cols_resolved)

    go, make_subplots = _require_plotly()
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=tuple(cols_resolved),
        vertical_spacing=0.12,
    )

    color_map = _build_color_map(cols_resolved, colors)
    rows_summary: List[dict] = []

    for i, col in enumerate(cols_resolved, start=1):
        s = sub[col].dropna()
        if s.empty:
            fig.add_annotation(
                text=f"No numeric data for '{col}'",
                xref="x domain", yref="y domain",
                x=0.5, y=0.5,
                showarrow=False,
                row=i, col=1,
            )
            continue

        mean_val = s.mean()
        median_val = s.median()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lb = q1 - iqr_multiplier * iqr
        ub = q3 + iqr_multiplier * iqr
        min_val = s.min()
        max_val = s.max()
        mask = (s < lb) | (s > ub)
        out_count = int(mask.sum())
        total = int(s.count())
        out_pct = (out_count / total * 100) if total > 0 else 0.0

        rows_summary.append({
            "column": col,
            "mean": mean_val,
            "median": median_val,
            "q1": q1,
            "q3": q3,
            "lb": lb,
            "ub": ub,
            "min": min_val,
            "max": max_val,
            "outliers_count": out_count,
            "outliers_pct": out_pct,
            "count": total,
        })

        fig.add_trace(
            go.Box(
                x=s,
                name=col,
                orientation="h",
                boxmean="sd",
                boxpoints="outliers",
                marker=dict(
                    color=color_map[col],
                    opacity=opacity,
                    line=dict(color=color_map[col], width=1),
                ),
                hovertemplate=(
                    f"<b>{col}</b><br>"
                    f"count: {total:,}<br>"
                    f"outliers: {out_count:,} ({out_pct:.{decimals}f}%)"
                    "<extra></extra>"
                ),
            ),
            row=i,
            col=1,
        )

        for bound in (lb, ub):
            if pd.notna(bound):
                fig.add_vline(
                    x=bound,
                    line=dict(color=line_color, width=1.2, dash="dash"),
                    row=i, col=1,
                )

        txt = "<br>".join([
            f"mean = {format_value(mean_val, 'num', decimals)}",
            f"median = {format_value(median_val, 'num', decimals)}",
            f"Q1 = {format_value(q1, 'num', decimals)}",
            f"Q3 = {format_value(q3, 'num', decimals)}",
            f"LB = {format_value(lb, 'num', decimals)}",
            f"UB = {format_value(ub, 'num', decimals)}",
            f"Min = {format_value(min_val, 'num', decimals)}",
            f"Max = {format_value(max_val, 'num', decimals)}",
            f"Outliers count = {format_value(out_count, 'int', decimals)}",
            f"Outliers % = {format_value(out_pct, 'pct', decimals)}",
        ])

        fig.add_annotation(
            text=txt,
            xref="x domain", yref="y domain",
            x=0.98, y=0.9,
            showarrow=False,
            align="right",
            font=dict(size=12, color="black"),
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.9,
            row=i, col=1,
        )

        fig.update_yaxes(showticklabels=False, row=i, col=1)
        fig.update_xaxes(showgrid=show_grid, zeroline=False, row=i, col=1)

    fig.update_layout(
        height=row_height * n_rows,
        width=width,
        title_text=title,
        showlegend=False,
        margin=dict(l=60, r=100, t=60, b=60),
        template=template,
    )

    summary_df = (
        pd.DataFrame(rows_summary).set_index("column")
        if rows_summary else None
    )

    if show and plot:
        fig.show()

    config = {
        "cols": list(cols_resolved),
        "decimals": decimals,
        "iqr_multiplier": iqr_multiplier,
    }
    audit_entry = {
        "stage": "phase1-eda",
        "function": "plot_boxplots",
        "timestamp": now_iso(),
        "df_name": df_name,
        "params": config,
        "decision": (
            f"Plotted boxplots for {len(cols_resolved)} column(s) of "
            f"'{df_name}'."
        ),
    }
    if isinstance(summary_df, pd.DataFrame):
        append_audit(summary_df, audit_entry)
        _summary_payload = json_safe(summary_df.to_dict())
        _audit_payload = list(summary_df.attrs.get("dextra_audit", []))
    else:
        _summary_payload = None
        _audit_payload = [audit_entry]
    manifest = {
        "stage": "phase1-eda",
        "function": "plot_boxplots",
        "df_name": df_name,
        "params": config,
        "summary": _summary_payload,
        "dextra_audit": _audit_payload,
    }
    results = []
    if return_fig:
        results.append(fig)
    if return_df:
        results.append(summary_df)
    if return_params:
        results.append(manifest)
    if not results:
        return None
    return results[0] if len(results) == 1 else tuple(results)


# Backward-compatible short aliases.
hister = plot_histograms
boxpl = plot_boxplots
