from typing import Optional, Sequence, Mapping
import os

import numpy as np
import pandas as pd
import inspect
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


try:
    from IPython.display import display
except ImportError:
    display = print

sns.set_style("whitegrid")

DEFAULT_BOX_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]




def _get_variable_name(df: pd.DataFrame) -> str:
    try:
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        for var_name, var_val in callers_local_vars:
            if var_val is df:
                return var_name
    except Exception:
        pass
    return "DataFrame"


def numdesc(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    return_df: bool = False,
    vertical: bool = True
):
    if df_name is None:
        df_name = _get_variable_name(df)

    if cols is None:
        num_col = df.select_dtypes(include=[np.number]).copy()
    else:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{missing}")
        num_col = df[cols].copy()

    num_col = num_col.apply(pd.to_numeric, errors="coerce")
    if num_col.empty:
        raise ValueError("No numeric columns.")

    mean_ = num_col.mean(numeric_only=True)
    std_ = num_col.std(numeric_only=True)
    var_ = num_col.var(numeric_only=True)
    cv_ = (std_ / mean_) * 100
    min_ = num_col.min(numeric_only=True)
    max_ = num_col.max(numeric_only=True)

    q1 = num_col.quantile(0.25, numeric_only=True)
    q2 = num_col.quantile(0.50, numeric_only=True)
    q3 = num_col.quantile(0.75, numeric_only=True)
    iqr = q3 - q1

    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr

    out_mask = (num_col.lt(lb)) | (num_col.gt(ub))
    out_count = out_mask.sum(numeric_only=True)

    safe_median = q2.replace(0, np.nan)
    diff_percent = (mean_ - safe_median).abs() / safe_median * 100
    diff_percent = diff_percent.fillna(0)

    count_ = num_col.count(numeric_only=True)
    total_rows = pd.Series(len(num_col), index=num_col.columns)
    missing = total_rows - count_

    out_pct = (out_count / count_.replace({0: np.nan})) * 100

    skew_ = num_col.skew(numeric_only=True)
    kurt_ = num_col.kurt(numeric_only=True)
    nunique_ = num_col.nunique(dropna=True)

    modes_dict = {}
    for c in num_col.columns:
        s = num_col[c].dropna()
        m = [] if s.empty else s.mode(dropna=True).tolist()
        modes_dict[c] = " | ".join([f"{x:,.{decimals}f}" for x in m]) if m else ""

    summary_df = pd.DataFrame({
        'mean'              : mean_,
        'std'               : std_,
        'var'               : var_,
        'cv_%'              : cv_,
        'min'               : min_,
        'q1'                : q1,
        'median'            : q2,
        'diff_mean_median_%': diff_percent,
        'q3'                : q3,
        'max'               : max_,
        'IQR'               : iqr,
        'lower_bound'       : lb,
        'upper_bound'       : ub,
        'outliers_count'    : out_count,
        'outliers_%'        : out_pct,
        'count'             : count_,
        'missing'           : missing,
        'unique'            : nunique_,
        'skewness'          : skew_,
        'kurtosis'          : kurt_,
        'modes'             : pd.Series(modes_dict),
    })

    def fmt(v, d=decimals):
        try:
            if pd.isna(v):
                return ""
            return f"{float(v):,.{d}f}"
        except:
            return v

    formatted = summary_df.copy()
    for c in formatted.columns:
        formatted[c] = [fmt(v, decimals) for v in formatted[c]]

    order = [
        'mean','std','var','cv_%','min','q1','median','diff_mean_median_%','q3','max',
        'IQR','lower_bound','upper_bound','outliers_count','outliers_%','count',
        'missing','unique','skewness','kurtosis','modes'
    ]

    formatted = formatted.T
    formatted = formatted.loc[[r for r in order if r in formatted.index]]

    if not vertical:
        formatted = formatted.T

    if show:
        print(f"Summary for: {df_name}")
        display(formatted)

    if return_df:
        return formatted
    
def _resolve_cols(df: pd.DataFrame, cols: Optional[List[str]]) -> List[str]:
    if cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns.")
        return numeric_cols
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{missing}")
    out = []
    seen = set()
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _to_numeric(df_subset: pd.DataFrame) -> pd.DataFrame:
    return df_subset.apply(pd.to_numeric, errors="coerce")


def _hister_summary(df_subset: pd.DataFrame) -> pd.DataFrame:
    mean_ = df_subset.mean(numeric_only=True)
    std_ = df_subset.std(numeric_only=True)
    cv_ = (std_ / mean_) * 100
    min_ = df_subset.min(numeric_only=True)
    max_ = df_subset.max(numeric_only=True)

    q1 = df_subset.quantile(0.25, numeric_only=True)
    q2 = df_subset.quantile(0.50, numeric_only=True)
    q3 = df_subset.quantile(0.75, numeric_only=True)
    iqr = q3 - q1
    lb = q1 - 1.5 * iqr
    ub = q3 + 1.5 * iqr

    out_mask = (df_subset.lt(lb)) | (df_subset.gt(ub))
    out_count = out_mask.sum(numeric_only=True)
    count_ = df_subset.count(numeric_only=True)
    total_rows = pd.Series(len(df_subset), index=df_subset.columns)
    missing = total_rows - count_
    out_pct = (out_count / count_.replace({0: np.nan})) * 100

    skew_ = df_subset.skew(numeric_only=True)
    kurt_ = df_subset.kurt(numeric_only=True)

    modes_dict = {}
    for c in df_subset.columns:
        s = df_subset[c].dropna()
        m = [] if s.empty else s.mode(dropna=True).tolist()
        modes_dict[c] = m

    nunique_ = df_subset.nunique(dropna=True)

    summary = pd.DataFrame({
        "mean": mean_,
        "std": std_,
        "variance": std_ ** 2,
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
        "mode": pd.Series(modes_dict, dtype=object),
    })

    return summary.sort_index()


def _hister_fmt(val, kind: str, decimals: int) -> str:
    if kind == "mode":
        if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
            vals = [x for x in val if pd.notna(x)]
            if not vals:
                return "-"
            all_int = all(
                isinstance(x, (int, np.integer))
                or (isinstance(x, float) and float(x).is_integer())
                for x in vals
            )
            return " | ".join(
                f"{int(x):,}" if all_int else f"{float(x):,.{decimals}f}"
                for x in vals
            )
        return "-"
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "-"
    if kind == "int":
        return f"{int(val):,}"
    if kind == "pct":
        return f"{float(val):,.{decimals}f}%"
    return f"{float(val):,.{decimals}f}"

def hister(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    bins: int = 20,
    decimals: int = 2,
    fig_width: float = 17.0,
    fig_row_height: float = 4.8,
    width_ratios: Sequence[float] = (3, 1),
    dpi: int = 120,
    hist_color: str = "skyblue",
    hist_edgecolor: str = "black",
    alpha: float = 0.85,
    kde_color: str = "blue",
    kde_linewidth: float = 2.2,
    title: str = "Histograms with adjacent statistical summary",
    save: bool = False,
    output_dir: str = "plots",
    filename: str = "histograms_with_summary.png",
    show: bool = True,
    return_df: bool = False,
):
    cols_resolved = _resolve_cols(df, cols)
    sub_raw = df[cols_resolved].copy()
    sub = _to_numeric(sub_raw)
    summary = _hister_summary(sub)

    n_rows = len(cols_resolved)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(fig_width, fig_row_height * n_rows),
        gridspec_kw={"width_ratios": width_ratios},
        dpi=dpi,
    )
    if n_rows == 1:
        axes = np.atleast_2d(axes)

    for i, col in enumerate(cols_resolved):
        ax_hist, ax_text = axes[i, 0], axes[i, 1]
        series = sub[col].dropna()

        if series.empty:
            ax_hist.text(0.5, 0.5, f"No numeric data for '{col}'",
                         ha="center", va="center", fontsize=12)
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
        if len(series) > 1:
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
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2.0,
                label="Mean",
            )
        if pd.notna(median_val):
            ax_hist.axvline(
                median_val,
                color="green",
                linestyle="-.",
                linewidth=1.8,
                label="Median",
            )

        ax_hist.set_title(f"Histogram of {col}", fontsize=13, fontweight="bold")
        ax_hist.set_xlabel(col, fontsize=11)
        ax_hist.set_ylabel("Density", fontsize=11)
        ax_hist.legend(loc="upper right", fontsize=9)

        lines = []

        def add(label, key, kind):
            val = summary.loc[col, key] if key in summary.columns else None
            lines.append(f"{label:<15}: {_hister_fmt(val, kind, decimals)}")

        add("count", "non_na_count", "int")
        add("unique_count", "unique_count", "int")
        add("missing_count", "missing_count", "int")
        add("mean", "mean", "num")
        add("std", "std", "num")
        add("variance", "variance", "num")
        add("cv_pct", "cv_pct", "pct")
        add("min", "min", "num")
        add("q1", "q1", "num")
        add("median", "median", "num")
        add("q3", "q3", "num")
        add("max", "max", "num")
        add("IQR", "iqr", "num")
        add("outliers_count", "outliers_count", "int")
        add("outliers_pct", "outliers_pct", "pct")
        add("skew", "skew", "num")
        add("kurt", "kurt", "num")

        ax_text.set_axis_off()
        ax_text.text(
            0.02,
            0.98,
            "\n".join(lines),
            ha="left",
            va="top",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle(title, fontsize=16, fontweight="bold")

    if save:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, filename)
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_df:
        return summary

def boxpl(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    decimals: int = 2,
    width: int = 1400,
    row_height: int = 350,
    opacity: float = 0.7,
    line_color: str = "orange",
    template: str = "plotly_white",
    show_grid: bool = True,
    title: str = "Boxplots",
    colors: Optional[Sequence[str] | Mapping[str, str]] = None,
    show: bool = True,
    return_fig: bool = False,
    return_df: bool = False,
):
    cols_resolved = _resolve_cols(df, cols)
    sub_raw = df[cols_resolved].copy()
    sub = _to_numeric(sub_raw)

    n_rows = len(cols_resolved)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        subplot_titles=tuple(cols_resolved),
        vertical_spacing=0.12,
    )

    if colors is None:
        palette = list(DEFAULT_BOX_COLORS)
        if len(palette) < n_rows:
            repeats = (n_rows + len(palette) - 1) // len(palette)
            palette = (palette * repeats)[:n_rows]
        color_map = {col: palette[i] for i, col in enumerate(cols_resolved)}
    elif isinstance(colors, Mapping):
        palette = list(DEFAULT_BOX_COLORS)
        if len(palette) < n_rows:
            repeats = (n_rows + len(palette) - 1) // len(palette)
            palette = (palette * repeats)[:n_rows]
        color_map = {}
        for i, col in enumerate(cols_resolved):
            color_map[col] = colors.get(col, palette[i])
    else:
        palette = list(colors)
        if len(palette) < n_rows:
            repeats = (n_rows + len(palette) - 1) // len(palette)
            palette = (palette * repeats)[:n_rows]
        color_map = {col: palette[i] for i, col in enumerate(cols_resolved)}

    rows_summary: list[dict] = []

    for i, col in enumerate(cols_resolved, start=1):
        s = sub[col].dropna()
        if s.empty:
            continue

        mean_val = s.mean()
        median_val = s.median()
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lb = q1 - 1.5 * iqr
        ub = q3 + 1.5 * iqr
        min_val = s.min()
        max_val = s.max()
        mask = (s < lb) | (s > ub)
        out_count = int(mask.sum())
        total = int(s.count())
        out_pct = (out_count / total * 100) if total > 0 else 0.0

        rows_summary.append(
            {
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
            }
        )

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
                    f"outliers: {out_count:,} ({out_pct:.2f}%)"
                    "<extra></extra>"
                ),
            ),
            row=i,
            col=1,
        )

        if pd.notna(lb):
            fig.add_vline(
                x=lb,
                line=dict(color=line_color, width=1.2, dash="dash"),
                row=i,
                col=1,
            )
        if pd.notna(ub):
            fig.add_vline(
                x=ub,
                line=dict(color=line_color, width=1.2, dash="dash"),
                row=i,
                col=1,
            )

        txt = (
            f"mean = {mean_val:,.{decimals}f}<br>"
            f"median = {median_val:,.{decimals}f}<br>"
            f"Q1 = {q1:,.{decimals}f}<br>"
            f"Q3 = {q3:,.{decimals}f}<br>"
            f"LB = {lb:,.{decimals}f}<br>"
            f"UB = {ub:,.{decimals}f}<br>"
            f"Min = {min_val:,.{decimals}f}<br>"
            f"Max = {max_val:,.{decimals}f}<br>"
            f"Outliers count = {out_count:,}<br>"
            f"Outliers %   = {out_pct:.2f}%"
        )

        fig.add_annotation(
            text=txt,
            xref="x domain",
            yref="y domain",
            x=0.98,
            y=0.9,
            showarrow=False,
            align="right",
            font=dict(size=12, color="black"),
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.9,
            row=i,
            col=1,
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

    summary_df = pd.DataFrame(rows_summary).set_index("column") if rows_summary else None

    if show:
        fig.show()

    if return_fig and return_df:
        return fig, summary_df
    if return_fig:
        return fig
    if return_df:
        return summary_df
    return None
