"""Advanced statistical helpers for dextra.

Stage 1 - Descriptive extensions (DONE):
  - z_scores             (F-M03-L05-02)
  - pearson_skewness     (F-M03-L07-01)
  - empirical_rule_check (F-M03-L05-01)
  - outliers_report      (F-M03-L03-03)

Stage 2 - Bivariate analysis (DONE):
  - correlation_matrix      (F-M03-L10-01)
  - simple_linear_regression (F-M03-L11-01/02/03)

Each public function returns a rich DataFrame, draws a comprehensive
multi-panel figure with a side annotation panel, and prints an English
'Decision:' sentence. All formulas are cross-referenced with
``Statistics Course Build/formulas.json``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as _sst

from ._utils import (
    format_value,
    get_variable_name,
    resolve_columns,
    safe_divide,
    to_numeric_frame,
)

try:
    from IPython.display import display as _ipy_display
except ImportError:
    _ipy_display = None


sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _display(frame: pd.DataFrame) -> None:
    if _ipy_display is not None:
        _ipy_display(frame)
    else:
        print(frame.to_string())


def _print_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def _format_summary(
    summary: pd.DataFrame,
    decimals: int,
    int_cols: Sequence[str] = (),
    pct_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Per-value formatter that tolerates mixed-type columns (pandas 3.x safe)."""
    def _fmt(v, kind):
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
        # default: try numeric first, fall back to string
        try:
            return f"{float(v):,.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)

    out = summary.copy()
    for c in out.columns:
        if c in int_cols:
            out[c] = out[c].map(lambda v: _fmt(v, "int"))
        elif c in pct_cols:
            out[c] = out[c].map(lambda v: _fmt(v, "pct"))
        else:
            out[c] = out[c].map(lambda v: _fmt(v, "num"))
    return out


def _finalize_figure(
    fig: Optional[plt.Figure],
    show: bool,
    plot: bool,
    return_fig: bool,
) -> None:
    """Display the figure when one was created. 'show' & 'plot' are independent."""
    if fig is None:
        return
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if not return_fig:
        plt.show()


def _significance_mark(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ===========================================================================
# STAGE 1
# ===========================================================================

# --- 1) Z-SCORES (F-M03-L05-02) -------------------------------------------

def z_scores(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    threshold: float = 3.0,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    return_zscores: bool = False,
    fig_width: float = 14.0,
    fig_row_height: float = 3.5,
    dpi: int = 110,
):
    """Compute Z-scores per column and report extreme-value counts.

    Z = (x - mu) / sigma   (source: F-M03-L05-02).
    """
    if threshold <= 0:
        raise ValueError(f"'threshold' must be > 0, got {threshold}")
    if decimals < 0:
        raise ValueError(f"'decimals' must be >= 0, got {decimals}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    cols_resolved = resolve_columns(df, cols, numeric_only=True)
    num = to_numeric_frame(df[cols_resolved].copy())
    if num.empty or num.shape[1] == 0:
        raise ValueError("No numeric columns available to compute Z-scores.")

    mu = num.mean()
    sigma = num.std()
    z = num.subtract(mu, axis=1).div(sigma.replace(0, np.nan), axis=1)

    rows = []
    for c in cols_resolved:
        zc = z[c].dropna()
        if zc.empty:
            continue
        abs_z = zc.abs()
        n_extreme = int((abs_z > threshold).sum())
        rows.append({
            "n": int(zc.count()),
            "mean_z": zc.mean(),
            "std_z": zc.std(),
            "max_abs_z": abs_z.max(),
            "n_above_1sigma": int((abs_z > 1).sum()),
            "n_above_2sigma": int((abs_z > 2).sum()),
            f"n_above_{threshold:g}sigma": n_extreme,
            f"pct_above_{threshold:g}sigma": (n_extreme / zc.count() * 100),
        })
    summary = pd.DataFrame(rows, index=[c for c in cols_resolved if not z[c].dropna().empty])
    summary.index.name = "column"

    total_extreme = int(summary[f"n_above_{threshold:g}sigma"].sum())
    n_cols_with_extremes = int((summary[f"n_above_{threshold:g}sigma"] > 0).sum())

    if show:
        _print_header(f"Z-scores for: {df_name}  (extreme threshold |Z| > {threshold})")
        int_cols = ("n", "n_above_1sigma", "n_above_2sigma",
                    f"n_above_{threshold:g}sigma")
        pct_cols = (f"pct_above_{threshold:g}sigma",)
        _display(_format_summary(summary, decimals, int_cols, pct_cols))
        print(
            f"\nDecision: {total_extreme} extreme value(s) across "
            f"{n_cols_with_extremes} column(s) with |Z| > {threshold}sigma.\n"
        )

    fig = None
    if plot:
        fig = _plot_z_scores(z, cols_resolved, threshold, fig_width, fig_row_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_zscores:
        return z
    if return_df and return_fig:
        return summary, fig
    if return_df:
        return summary
    if return_fig:
        return fig
    return None


def _plot_z_scores(z, cols, threshold, fig_width, row_height, dpi, decimals):
    n = len(cols)
    fig, axes = plt.subplots(
        n, 2, figsize=(fig_width, row_height * n), dpi=dpi,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    if n == 1:
        axes = np.atleast_2d(axes)

    for i, c in enumerate(cols):
        ax_plot, ax_text = axes[i, 0], axes[i, 1]
        zc = z[c].dropna()
        if zc.empty:
            ax_plot.text(0.5, 0.5, f"No numeric data for '{c}'", ha="center", va="center")
            ax_plot.set_axis_off()
            ax_text.set_axis_off()
            continue
        sns.histplot(zc, bins=40, color="steelblue", edgecolor="black",
                     alpha=0.85, ax=ax_plot, stat="count")
        ax_plot.axvline(0, color="black", linewidth=1.0, alpha=0.8)
        for k, ls in [(1, ":"), (2, "--"), (threshold, "-")]:
            ax_plot.axvline(k,  color="red", linestyle=ls, linewidth=1.3, alpha=0.7)
            ax_plot.axvline(-k, color="red", linestyle=ls, linewidth=1.3, alpha=0.7)
        ax_plot.set_title(f"Z-scores: {c}", fontsize=12, fontweight="bold")
        ax_plot.set_xlabel("Z"); ax_plot.set_ylabel("count")

        abs_z = zc.abs()
        n_total = int(zc.count())
        n1, n2, nt = int((abs_z > 1).sum()), int((abs_z > 2).sum()), int((abs_z > threshold).sum())
        lines = [
            f"{'n':<15}: {n_total:,}",
            f"{'mean(Z)':<15}: {zc.mean():.{decimals}f}",
            f"{'std(Z)':<15}: {zc.std():.{decimals}f}",
            f"{'max|Z|':<15}: {abs_z.max():.{decimals}f}",
            f"{'|Z| > 1s':<15}: {n1:,}  ({n1/n_total*100:.{decimals}f}%)",
            f"{'|Z| > 2s':<15}: {n2:,}  ({n2/n_total*100:.{decimals}f}%)",
            f"{f'|Z| > {threshold:g}s':<15}: {nt:,}  ({nt/n_total*100:.{decimals}f}%)",
        ]
        ax_text.set_axis_off()
        ax_text.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
                     fontsize=10, family="monospace",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.suptitle("Z-scores (red lines: +/-1s +/-2s +/-threshold)",
                 fontsize=14, fontweight="bold")
    return fig


# --- 2) PEARSON SKEWNESS (F-M03-L07-01) -----------------------------------

def pearson_skewness(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 14.0,
    fig_row_height: float = 3.5,
    dpi: int = 110,
):
    """Karl Pearson's skewness coefficient: 3*(mean - median)/sigma."""
    if decimals < 0:
        raise ValueError(f"'decimals' must be >= 0, got {decimals}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    cols_resolved = resolve_columns(df, cols, numeric_only=True)
    num = to_numeric_frame(df[cols_resolved].copy())
    if num.empty or num.shape[1] == 0:
        raise ValueError("No numeric columns available.")

    mean_ = num.mean()
    median_ = num.median()
    std_ = num.std()
    skew_pearson = 3.0 * (mean_ - median_) / std_.replace(0, np.nan)
    skew_fisher = num.skew()

    def _direction(s):
        if pd.isna(s): return "-"
        if s > 0.1: return "right"
        if s < -0.1: return "left"
        return "symmetric"

    def _magnitude(s):
        if pd.isna(s): return "-"
        a = abs(s)
        if a < 0.5: return "low"
        if a < 1.0: return "moderate"
        return "high"

    summary = pd.DataFrame({
        "mean": mean_,
        "median": median_,
        "std": std_,
        "skew_pearson": skew_pearson,
        "skew_fisher": skew_fisher,
        "direction": skew_pearson.map(_direction),
        "magnitude": skew_pearson.map(_magnitude),
    })

    if show:
        _print_header(f"Pearson skewness for: {df_name}  (formula F-M03-L07-01)")
        _display(_format_summary(summary, decimals))
        worst = summary["skew_pearson"].abs().idxmax()
        s_worst = summary.loc[worst, "skew_pearson"]
        print(f"\nDecision: most skewed column is '{worst}' "
              f"(skew_pearson = {s_worst:+.{decimals}f}, "
              f"{summary.loc[worst, 'magnitude']} {summary.loc[worst, 'direction']}).\n")

    fig = None
    if plot:
        fig = _plot_pearson_skewness(num, summary, cols_resolved,
                                     fig_width, fig_row_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_pearson_skewness(num, summary, cols, fig_width, row_height, dpi, decimals):
    n = len(cols)
    fig, axes = plt.subplots(
        n, 2, figsize=(fig_width, row_height * n), dpi=dpi,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    if n == 1:
        axes = np.atleast_2d(axes)
    for i, c in enumerate(cols):
        ax_plot, ax_text = axes[i, 0], axes[i, 1]
        s = num[c].dropna()
        if s.empty:
            ax_plot.set_axis_off(); ax_text.set_axis_off(); continue
        sns.histplot(s, bins=30, color="lightcoral", edgecolor="black",
                     alpha=0.8, ax=ax_plot, stat="density")
        if len(s) > 1:
            sns.kdeplot(s, ax=ax_plot, color="darkred", linewidth=2.0)
        m = summary.loc[c, "mean"]
        md = summary.loc[c, "median"]
        ax_plot.axvline(m,  color="red", linestyle="--",  linewidth=2.0, label="mean")
        ax_plot.axvline(md, color="green", linestyle="-.", linewidth=2.0, label="median")
        ax_plot.set_title(f"{c}", fontsize=12, fontweight="bold")
        ax_plot.legend(fontsize=9); ax_plot.set_xlabel(c); ax_plot.set_ylabel("density")

        sp = summary.loc[c, "skew_pearson"]
        sf = summary.loc[c, "skew_fisher"]
        lines = [
            f"{'mean':<14}: {m:.{decimals}f}",
            f"{'median':<14}: {md:.{decimals}f}",
            f"{'std':<14}: {summary.loc[c, 'std']:.{decimals}f}",
            f"{'skew (Pearson)':<14}: {sp:+.{decimals}f}",
            f"{'skew (Fisher)':<14}: {sf:+.{decimals}f}",
            f"{'direction':<14}: {summary.loc[c, 'direction']}",
            f"{'magnitude':<14}: {summary.loc[c, 'magnitude']}",
        ]
        ax_text.set_axis_off()
        ax_text.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
                     fontsize=10, family="monospace",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.suptitle("Pearson skewness  (mean=red dashed, median=green dash-dot)",
                 fontsize=14, fontweight="bold")
    return fig


# --- 3) EMPIRICAL RULE CHECK (F-M03-L05-01) -------------------------------

_EMPIRICAL_TARGETS = {1: 68.2689, 2: 95.4500, 3: 99.7300}


def empirical_rule_check(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    decimals: int = 2,
    df_name: Optional[str] = None,
    tolerance: float = 3.0,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 14.0,
    fig_row_height: float = 3.8,
    dpi: int = 110,
):
    """Check actual vs theoretical 68/95/99.7 coverage."""
    if tolerance <= 0:
        raise ValueError(f"'tolerance' must be > 0, got {tolerance}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    cols_resolved = resolve_columns(df, cols, numeric_only=True)
    num = to_numeric_frame(df[cols_resolved].copy())
    if num.empty or num.shape[1] == 0:
        raise ValueError("No numeric columns available.")

    rows = []
    for c in cols_resolved:
        s = num[c].dropna()
        if s.empty:
            continue
        mu, sd = s.mean(), s.std()
        if sd == 0 or pd.isna(sd):
            rows.append({"column": c, "n": int(s.count()),
                         "pct_1sigma": np.nan, "pct_2sigma": np.nan,
                         "pct_3sigma": np.nan, "diff_1sigma": np.nan,
                         "diff_2sigma": np.nan, "diff_3sigma": np.nan,
                         "max_diff": np.nan, "looks_normal": False})
            continue
        within = {}; diffs = {}
        for k, target in _EMPIRICAL_TARGETS.items():
            actual = (s.between(mu - k * sd, mu + k * sd)).mean() * 100.0
            within[k] = actual
            diffs[k] = actual - target
        max_diff = max(abs(d) for d in diffs.values())
        rows.append({
            "column": c, "n": int(s.count()),
            "pct_1sigma": within[1], "pct_2sigma": within[2], "pct_3sigma": within[3],
            "diff_1sigma": diffs[1], "diff_2sigma": diffs[2], "diff_3sigma": diffs[3],
            "max_diff": max_diff, "looks_normal": max_diff < tolerance,
        })
    summary = pd.DataFrame(rows).set_index("column")

    if show:
        _print_header(f"Empirical-rule check for: {df_name}  "
                      f"(targets 68.27 / 95.45 / 99.73 %, tolerance {tolerance}%)")
        int_cols = ("n",)
        pct_cols = ("pct_1sigma", "pct_2sigma", "pct_3sigma",
                    "diff_1sigma", "diff_2sigma", "diff_3sigma", "max_diff")
        _display(_format_summary(summary, decimals, int_cols, pct_cols))
        n_normal = int(summary["looks_normal"].sum())
        n_total = len(summary)
        print(f"\nDecision: {n_normal} of {n_total} column(s) look approximately Normal "
              f"(all bands within +/-{tolerance}% of theoretical coverage).\n")

    fig = None
    if plot:
        fig = _plot_empirical_rule(num, summary, cols_resolved,
                                   fig_width, fig_row_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_empirical_rule(num, summary, cols, fig_width, row_height, dpi, decimals):
    n = len(cols)
    fig, axes = plt.subplots(
        n, 2, figsize=(fig_width, row_height * n), dpi=dpi,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    if n == 1:
        axes = np.atleast_2d(axes)
    band_colors = {1: "#a8d5a2", 2: "#f3d6a0", 3: "#f4a8a0"}
    for i, c in enumerate(cols):
        ax_plot, ax_text = axes[i, 0], axes[i, 1]
        s = num[c].dropna()
        if s.empty:
            ax_plot.set_axis_off(); ax_text.set_axis_off(); continue
        mu, sd = s.mean(), s.std()
        sns.histplot(s, bins=40, color="steelblue", edgecolor="black",
                     alpha=0.7, ax=ax_plot, stat="density")
        if len(s) > 1:
            sns.kdeplot(s, ax=ax_plot, color="navy", linewidth=2.0)
        for k in (3, 2, 1):
            ax_plot.axvspan(mu - k * sd, mu + k * sd, color=band_colors[k],
                            alpha=0.25, label=f"+/-{k}s")
        ax_plot.axvline(mu, color="black", linestyle="--", linewidth=1.5)
        ax_plot.set_title(f"{c}", fontsize=12, fontweight="bold")
        ax_plot.set_xlabel(c)
        ax_plot.legend(loc="upper right", fontsize=8)

        looks = "yes" if summary.loc[c, "looks_normal"] else "no"
        lines = [
            f"{'n':<10}: {int(summary.loc[c, 'n']):,}",
            f"{'1s actual':<10}: {summary.loc[c, 'pct_1sigma']:.{decimals}f}%",
            f"{'  target':<10}: {_EMPIRICAL_TARGETS[1]:.{decimals}f}%",
            f"{'2s actual':<10}: {summary.loc[c, 'pct_2sigma']:.{decimals}f}%",
            f"{'  target':<10}: {_EMPIRICAL_TARGETS[2]:.{decimals}f}%",
            f"{'3s actual':<10}: {summary.loc[c, 'pct_3sigma']:.{decimals}f}%",
            f"{'  target':<10}: {_EMPIRICAL_TARGETS[3]:.{decimals}f}%",
            f"{'max diff':<10}: {summary.loc[c, 'max_diff']:.{decimals}f}%",
            f"{'normal?':<10}: {looks}",
        ]
        ax_text.set_axis_off()
        ax_text.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
                     fontsize=10, family="monospace",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.suptitle("Empirical Rule Check (68 / 95 / 99.7)", fontsize=14, fontweight="bold")
    return fig


# --- 4) OUTLIERS REPORT (F-M03-L03-03) ------------------------------------

_VALID_OUTLIER_METHODS = ("iqr", "zscore")


def outliers_report(
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
    return_rows: bool = False,
    fig_width: float = 14.0,
    fig_row_height: float = 3.5,
    dpi: int = 110,
):
    """Detect outlier rows by IQR fence or Z-score."""
    if method not in _VALID_OUTLIER_METHODS:
        raise ValueError(f"'method' must be one of {_VALID_OUTLIER_METHODS}, got {method!r}")
    if k <= 0: raise ValueError(f"'k' must be > 0, got {k}")
    if z_threshold <= 0: raise ValueError(f"'z_threshold' must be > 0, got {z_threshold}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    cols_resolved = resolve_columns(df, cols, numeric_only=True)
    num = to_numeric_frame(df[cols_resolved].copy())
    if num.empty or num.shape[1] == 0:
        raise ValueError("No numeric columns available.")

    rows = []
    outlier_mask = pd.DataFrame(False, index=num.index, columns=cols_resolved)
    for c in cols_resolved:
        s = num[c].dropna()
        if s.empty: continue
        if method == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lb = q1 - k * iqr; ub = q3 + k * iqr
            mask_c = (num[c] < lb) | (num[c] > ub)
            extra = {"q1": q1, "q3": q3, "iqr": iqr,
                     "lower_bound": lb, "upper_bound": ub}
        else:
            mu, sd = s.mean(), s.std()
            if sd == 0 or pd.isna(sd):
                mask_c = pd.Series(False, index=num.index)
            else:
                z = (num[c] - mu) / sd
                mask_c = z.abs() > z_threshold
            extra = {"mean": mu, "std": sd,
                     "lower_bound": mu - z_threshold * sd,
                     "upper_bound": mu + z_threshold * sd}
        outlier_mask[c] = mask_c.fillna(False)
        n_out = int(mask_c.sum())
        pct = n_out / s.count() * 100 if s.count() else 0.0
        rows.append({"column": c, "method": method, "n_valid": int(s.count()),
                     "n_outliers": n_out, "pct_outliers": pct, **extra})
    summary = pd.DataFrame(rows).set_index("column")

    any_outlier_per_row = outlier_mask.any(axis=1)
    n_rows_flagged = int(any_outlier_per_row.sum())
    pct_rows_flagged = n_rows_flagged / len(num) * 100 if len(num) else 0.0

    if show:
        suffix = f"k={k}" if method == "iqr" else f"|Z|>{z_threshold}"
        _print_header(f"Outliers report for: {df_name}  (method={method}, {suffix})")
        _display(_format_summary(summary, decimals,
                                 int_cols=("n_valid", "n_outliers"),
                                 pct_cols=("pct_outliers",)))
        print(f"\nDecision: {n_rows_flagged} row(s) ({pct_rows_flagged:.{decimals}f}%) "
              f"are outlying in at least one column.\n")

    fig = None
    if plot:
        fig = _plot_outliers(num, summary, cols_resolved, method, k, z_threshold,
                             fig_width, fig_row_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_rows:
        if not n_rows_flagged:
            return df.iloc[0:0].assign(outlier_in=pd.Series(dtype="object"))
        flagged_rows = df.loc[any_outlier_per_row].copy()
        flagged_rows["outlier_in"] = outlier_mask.loc[any_outlier_per_row].apply(
            lambda r: ", ".join(c for c in cols_resolved if r[c]), axis=1)
        return flagged_rows
    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_outliers(num, summary, cols, method, k, z_threshold, fig_width, row_height, dpi, decimals):
    n = len(cols)
    fig, axes = plt.subplots(
        n, 2, figsize=(fig_width, row_height * n), dpi=dpi,
        gridspec_kw={"width_ratios": [3, 1]},
    )
    if n == 1:
        axes = np.atleast_2d(axes)
    for i, c in enumerate(cols):
        ax_plot, ax_text = axes[i, 0], axes[i, 1]
        s = num[c].dropna()
        if s.empty or c not in summary.index:
            ax_plot.set_axis_off(); ax_text.set_axis_off(); continue
        lb = summary.loc[c, "lower_bound"]
        ub = summary.loc[c, "upper_bound"]
        is_out = (s < lb) | (s > ub)
        jitter_in  = np.random.uniform(-0.02, 0.02, (~is_out).sum())
        jitter_out = np.random.uniform(-0.02, 0.02, is_out.sum())
        ax_plot.scatter(s[~is_out], jitter_in,  color="steelblue", s=18, alpha=0.5, label="inlier")
        ax_plot.scatter(s[is_out],  jitter_out, color="red", s=28, alpha=0.9,
                        label="outlier", edgecolor="black")
        ax_plot.axvline(lb, color="orange", linestyle="--", linewidth=1.3)
        ax_plot.axvline(ub, color="orange", linestyle="--", linewidth=1.3)
        ax_plot.set_title(f"{c}: {int(summary.loc[c, 'n_outliers'])} outliers "
                          f"({summary.loc[c, 'pct_outliers']:.{decimals}f}%)",
                          fontsize=11, fontweight="bold")
        ax_plot.set_xlabel(c); ax_plot.set_yticks([])
        ax_plot.legend(loc="upper right", fontsize=8)

        if method == "iqr":
            lines = [
                f"{'method':<10}: IQR (k={k})",
                f"{'Q1':<10}: {summary.loc[c, 'q1']:.{decimals}f}",
                f"{'Q3':<10}: {summary.loc[c, 'q3']:.{decimals}f}",
                f"{'IQR':<10}: {summary.loc[c, 'iqr']:.{decimals}f}",
                f"{'LB':<10}: {lb:.{decimals}f}",
                f"{'UB':<10}: {ub:.{decimals}f}",
                f"{'n_outl':<10}: {int(summary.loc[c, 'n_outliers']):,}",
                f"{'pct':<10}: {summary.loc[c, 'pct_outliers']:.{decimals}f}%",
            ]
        else:
            lines = [
                f"{'method':<10}: |Z|>{z_threshold}",
                f"{'mean':<10}: {summary.loc[c, 'mean']:.{decimals}f}",
                f"{'std':<10}: {summary.loc[c, 'std']:.{decimals}f}",
                f"{'LB':<10}: {lb:.{decimals}f}",
                f"{'UB':<10}: {ub:.{decimals}f}",
                f"{'n_outl':<10}: {int(summary.loc[c, 'n_outliers']):,}",
                f"{'pct':<10}: {summary.loc[c, 'pct_outliers']:.{decimals}f}%",
            ]
        ax_text.set_axis_off()
        ax_text.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
                     fontsize=10, family="monospace",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    title = (f"Outliers -- IQR fence (k={k})"
             if method == "iqr"
             else f"Outliers -- Z-score (|Z|>{z_threshold})")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    return fig


# ===========================================================================
# STAGE 2 - Bivariate analysis
# ===========================================================================

# --- 5) CORRELATION MATRIX (F-M03-L10-01) ---------------------------------

def correlation_matrix(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "pearson",
    decimals: int = 2,
    alpha: float = 0.05,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    return_p: bool = False,
    annot: bool = True,
    mask_lower: bool = False,
    cmap: str = "RdBu_r",
    fig_width: float = 12.0,
    fig_height: float = 8.5,
    dpi: int = 110,
):
    """Correlation matrix with p-values and a labelled heatmap.

    Pearson source: F-M03-L10-01.
    Spearman/Kendall are rank-based robust alternatives.

    Parameters
    ----------
    method : {'pearson', 'spearman', 'kendall'}, default 'pearson'.
    alpha  : significance threshold for the decision text.
    return_p : if True, return (r_matrix, p_matrix) instead of r alone.

    Examples
    --------
    >>> dx.correlation_matrix(df)
    >>> dx.correlation_matrix(df, method='spearman')
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"'method' must be one of pearson/spearman/kendall, got {method!r}")
    if not (0 < alpha < 1):
        raise ValueError(f"'alpha' must be in (0, 1), got {alpha}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    cols_resolved = resolve_columns(df, cols, numeric_only=True)
    if len(cols_resolved) < 2:
        raise ValueError("Need at least 2 numeric columns to compute a correlation matrix.")
    num = to_numeric_frame(df[cols_resolved].copy())

    n = len(cols_resolved)
    r_mat = pd.DataFrame(np.eye(n), index=cols_resolved, columns=cols_resolved, dtype=float)
    p_mat = pd.DataFrame(np.zeros((n, n)), index=cols_resolved, columns=cols_resolved, dtype=float)
    n_mat = pd.DataFrame(np.zeros((n, n)), index=cols_resolved, columns=cols_resolved, dtype=int)

    for i, ci in enumerate(cols_resolved):
        for j, cj in enumerate(cols_resolved):
            if j < i:
                continue
            if i == j:
                r_mat.iloc[i, j] = 1.0
                p_mat.iloc[i, j] = 0.0
                n_mat.iloc[i, j] = int(num[ci].count())
                continue
            valid = num[[ci, cj]].dropna()
            if len(valid) < 3:
                r_mat.iloc[i, j] = r_mat.iloc[j, i] = np.nan
                p_mat.iloc[i, j] = p_mat.iloc[j, i] = np.nan
                n_mat.iloc[i, j] = n_mat.iloc[j, i] = len(valid)
                continue
            if method == "pearson":
                rr, pp = _sst.pearsonr(valid[ci], valid[cj])
            elif method == "spearman":
                res = _sst.spearmanr(valid[ci], valid[cj])
                rr, pp = float(res.statistic), float(res.pvalue)
            else:
                res = _sst.kendalltau(valid[ci], valid[cj])
                rr, pp = float(res.statistic), float(res.pvalue)
            r_mat.iloc[i, j] = r_mat.iloc[j, i] = float(rr)
            p_mat.iloc[i, j] = p_mat.iloc[j, i] = float(pp)
            n_mat.iloc[i, j] = n_mat.iloc[j, i] = len(valid)

    # Upper-triangle summary
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            r = r_mat.iloc[i, j]
            if pd.isna(r):
                continue
            pairs.append({
                "var1": cols_resolved[i],
                "var2": cols_resolved[j],
                "r": float(r),
                "p_value": float(p_mat.iloc[i, j]),
                "n": int(n_mat.iloc[i, j]),
            })
    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.reindex(pairs_df["r"].abs().sort_values(ascending=False).index).reset_index(drop=True)

    n_pairs = len(pairs_df)
    n_sig = int((pairs_df["p_value"] < alpha).sum()) if n_pairs else 0
    avg_abs_r = float(pairs_df["r"].abs().mean()) if n_pairs else float("nan")

    if show:
        _print_header(f"{method.title()} correlation matrix for: {df_name}  (alpha={alpha})")
        # Build display matrix with stars
        disp = pd.DataFrame("", index=cols_resolved, columns=cols_resolved)
        for i in range(n):
            for j in range(n):
                r = r_mat.iloc[i, j]
                if pd.isna(r):
                    disp.iloc[i, j] = "-"
                elif i == j:
                    disp.iloc[i, j] = f"{r:+.{decimals}f}"
                else:
                    disp.iloc[i, j] = f"{r:+.{decimals}f}{_significance_mark(p_mat.iloc[i, j])}"
        _display(disp)
        print("\nTop 5 correlations:")
        top5 = pairs_df.head(5).copy()
        if len(top5):
            top5["r"] = top5["r"].map(lambda x: f"{x:+.{decimals}f}")
            top5["p_value"] = top5["p_value"].map(lambda x: f"{x:.4f}")
            _display(top5)
        print(f"\nDecision: {n_sig}/{n_pairs} pair(s) significant at alpha={alpha}.  "
              f"Avg |r| = {avg_abs_r:.{decimals}f}.\n")

    fig = None
    if plot:
        fig = _plot_correlation_matrix(r_mat, p_mat, pairs_df, method, alpha,
                                       annot, mask_lower, cmap,
                                       fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_p:
        if return_fig: return r_mat, p_mat, fig
        return r_mat, p_mat
    if return_df and return_fig: return r_mat, fig
    if return_df: return r_mat
    if return_fig: return fig
    return None


def _plot_correlation_matrix(r_mat, p_mat, pairs_df, method, alpha, annot, mask_lower, cmap,
                              fig_width, fig_height, dpi, decimals):
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    mask = np.triu(np.ones_like(r_mat, dtype=bool)) if mask_lower else None

    if annot:
        annot_mat = r_mat.copy().astype(object)
        for i in range(r_mat.shape[0]):
            for j in range(r_mat.shape[1]):
                v = r_mat.iloc[i, j]
                if pd.isna(v):
                    annot_mat.iloc[i, j] = "-"
                elif i == j:
                    annot_mat.iloc[i, j] = f"{v:.{decimals}f}"
                else:
                    annot_mat.iloc[i, j] = f"{v:+.{decimals}f}\n{_significance_mark(p_mat.iloc[i, j])}"
        annot_data = annot_mat.values
    else:
        annot_data = False

    sns.heatmap(r_mat, ax=ax_heat, vmin=-1, vmax=1, center=0,
                cmap=cmap, annot=annot_data, fmt="", square=True,
                cbar_kws={"shrink": 0.7, "label": f"{method} r"},
                mask=mask, linewidths=0.5,
                annot_kws={"fontsize": 9})
    ax_heat.set_title(f"{method.title()} Correlation Matrix", fontsize=13, fontweight="bold")

    ax_info.set_axis_off()
    n_cols = r_mat.shape[0]
    n_pairs = len(pairs_df)
    n_sig = int((pairs_df["p_value"] < alpha).sum()) if n_pairs else 0
    avg_abs = float(pairs_df["r"].abs().mean()) if n_pairs else float("nan")

    lines = [
        f"Method     : {method}",
        f"Variables  : {n_cols}",
        f"Pairs      : {n_pairs}",
        f"Significant: {n_sig} / {n_pairs}",
        f"  (alpha = {alpha})",
        f"Avg |r|    : {avg_abs:.{decimals}f}",
        "",
        "Top 5 |r|:",
    ]
    for k, row in enumerate(pairs_df.head(5).itertuples(), 1):
        sig = _significance_mark(row.p_value)
        lines.append(f"  {k}. {row.var1} ~ {row.var2}")
        lines.append(f"     r = {row.r:+.{decimals}f}{sig}")
    lines += ["",
              "Significance:",
              "  ***  p < 0.001",
              "  **   p < 0.01",
              "  *    p < 0.05"]
    ax_info.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
                 fontsize=10, family="monospace",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    fig.suptitle(f"Correlation analysis ({method})", fontsize=14, fontweight="bold")
    return fig


# --- 6) SIMPLE LINEAR REGRESSION (F-M03-L11-01/02/03) --------------------

def simple_linear_regression(
    df: pd.DataFrame,
    x: str,
    y: str,
    alpha: float = 0.05,
    decimals: int = 4,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    return_residuals: bool = False,
    ci_band: bool = True,
    fig_width: float = 16.0,
    fig_height: float = 5.0,
    dpi: int = 110,
):
    """Simple linear regression Y = m*X + b with full diagnostics.

    Source: course formulas
        F-M03-L11-01 : Y = m*X + b
        F-M03-L11-02 : m = r * (s_y / s_x)
        F-M03-L11-03 : b = mean(y) - m * mean(x)

    Visual: 3-panel plot
        (1) scatter + regression line + (1-alpha) CI band
        (2) residuals vs fitted (homoscedasticity check)
        (3) Q-Q plot of residuals (normality check)

    Examples
    --------
    >>> dx.simple_linear_regression(df, x='age', y='income')
    """
    if x not in df.columns:
        raise KeyError(f"x column not found: {x!r}")
    if y not in df.columns:
        raise KeyError(f"y column not found: {y!r}")
    if not (0 < alpha < 1):
        raise ValueError(f"'alpha' must be in (0, 1), got {alpha}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    valid = df[[x, y]].dropna()
    if len(valid) < 3:
        raise ValueError(f"Need >= 3 valid (non-NaN) rows; got {len(valid)}.")

    xa = pd.to_numeric(valid[x], errors="coerce").to_numpy()
    ya = pd.to_numeric(valid[y], errors="coerce").to_numpy()
    mask = ~(np.isnan(xa) | np.isnan(ya))
    xa = xa[mask]; ya = ya[mask]
    n = len(xa)
    if n < 3:
        raise ValueError(f"Need >= 3 valid (non-NaN) rows; got {n}.")

    res = _sst.linregress(xa, ya)
    m = float(res.slope)
    b = float(res.intercept)
    r = float(res.rvalue)
    r2 = r ** 2
    p_value = float(res.pvalue)
    se_slope = float(res.stderr) if res.stderr is not None else float("nan")
    se_intercept = float(res.intercept_stderr) if res.intercept_stderr is not None else float("nan")
    df_resid = n - 2

    y_pred = m * xa + b
    residuals = ya - y_pred
    ss_tot = float(((ya - ya.mean()) ** 2).sum())
    ss_res = float((residuals ** 2).sum())
    mse = ss_res / df_resid if df_resid > 0 else float("nan")
    rmse = float(np.sqrt(mse)) if not np.isnan(mse) else float("nan")
    t_crit = float(_sst.t.ppf(1 - alpha / 2, df_resid)) if df_resid > 0 else float("nan")
    slope_ci_low = m - t_crit * se_slope
    slope_ci_high = m + t_crit * se_slope

    # Residual normality
    if 3 <= n <= 5000:
        sh = _sst.shapiro(residuals)
        normality_p = float(sh.pvalue)
        normality_test = "Shapiro-Wilk"
    else:
        ks = _sst.kstest(residuals, "norm", args=(residuals.mean(), residuals.std(ddof=0)))
        normality_p = float(ks.pvalue)
        normality_test = "Kolmogorov-Smirnov"

    summary = pd.DataFrame({
        "value": [
            n, m, b, r, r2, r2 * 100, p_value, se_slope, se_intercept,
            slope_ci_low, slope_ci_high, rmse,
            float(residuals.mean()), float(residuals.std(ddof=1)),
            normality_p,
        ]},
        index=[
            "n", "slope (m)", "intercept (b)", "r", "r_squared", "variance_explained_%",
            "p_value_slope", "SE_slope", "SE_intercept",
            f"slope_CI_low_{int((1-alpha)*100)}", f"slope_CI_high_{int((1-alpha)*100)}",
            "RMSE", "residuals_mean", "residuals_std",
            f"normality_p_{normality_test}",
        ])
    summary.index.name = "metric"

    sign = "+" if b >= 0 else "-"
    equation = f"{y} = {m:.{decimals}f} * {x} {sign} {abs(b):.{decimals}f}"

    if show:
        _print_header(f"Simple Linear Regression for: {df_name}  ({y} ~ {x}, alpha={alpha})")
        _display(_format_summary(summary, decimals))
        sig = "Significant" if p_value < alpha else "Not significant"
        normality_status = "OK" if normality_p > alpha else "VIOLATED"
        print(f"\nEquation:  {equation}")
        print(f"Decision:  Slope is {sig.lower()} (p={p_value:.4f}).  "
              f"R^2 = {r2 * 100:.{decimals}f}% variance explained.  "
              f"Residuals normality: {normality_status} ({normality_test} p={normality_p:.4f}).\n")

    fig = None
    if plot:
        fig = _plot_slr(xa, ya, y_pred, residuals, m, b, r2, p_value, se_slope, df_resid,
                        alpha, x, y, ci_band, fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    residuals_series = pd.Series(residuals, index=valid.index[mask], name="residuals")
    if return_residuals:
        if return_df and return_fig: return summary, fig, residuals_series
        if return_df: return summary, residuals_series
        return residuals_series
    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_slr(xa, ya, y_pred, residuals, m, b, r2, p_value, se_slope, df_resid,
              alpha, x_name, y_name, ci_band, fig_width, fig_height, dpi, decimals):
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=dpi)

    # Panel 1: scatter + regression line (+ CI band)
    ax = axes[0]
    ax.scatter(xa, ya, color="steelblue", alpha=0.5, s=25,
               edgecolor="black", linewidth=0.4)
    idx = np.argsort(xa)
    x_sorted = xa[idx]; y_pred_sorted = y_pred[idx]
    ax.plot(x_sorted, y_pred_sorted, color="red", linewidth=2.0,
            label=f"y = {m:.{decimals}f}*x + {b:.{decimals}f}")
    if ci_band and df_resid > 0:
        n = len(xa)
        x_mean = xa.mean()
        s_xx = ((xa - x_mean) ** 2).sum()
        mse = (residuals ** 2).sum() / df_resid
        se_pred = np.sqrt(mse * (1.0 / n + (x_sorted - x_mean) ** 2 / s_xx))
        t_crit = float(_sst.t.ppf(1 - alpha / 2, df_resid))
        band = t_crit * se_pred
        ax.fill_between(x_sorted, y_pred_sorted - band, y_pred_sorted + band,
                        color="red", alpha=0.18, label=f"{int((1-alpha)*100)}% CI")
    ax.set_xlabel(x_name); ax.set_ylabel(y_name)
    ax.set_title(f"Fit  (R^2 = {r2:.{decimals}f})", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)

    # Panel 2: residuals vs fitted
    ax = axes[1]
    ax.scatter(y_pred, residuals, color="darkgreen", alpha=0.5, s=25,
               edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"Fitted {y_name}"); ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted\n(homoscedasticity)", fontsize=11, fontweight="bold")

    # Panel 3: Q-Q plot of residuals
    ax = axes[2]
    (osm, osr), (slope_qq, intercept_qq, r_qq) = _sst.probplot(residuals, dist="norm")
    ax.scatter(osm, osr, color="purple", alpha=0.5, s=25,
               edgecolor="black", linewidth=0.4)
    ax.plot(osm, slope_qq * osm + intercept_qq, color="red", linewidth=1.5)
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles (residuals)")
    ax.set_title("Q-Q plot of residuals\n(normality)", fontsize=11, fontweight="bold")

    fig.suptitle(f"Simple Linear Regression: {y_name} ~ {x_name}",
                 fontsize=14, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Backward-compatible short aliases (dextra naming tradition)
# ---------------------------------------------------------------------------

# Stage 1
zsc      = z_scores
pskew    = pearson_skewness
emprule  = empirical_rule_check
outrep   = outliers_report

# Stage 2
corrmat  = correlation_matrix
slr      = simple_linear_regression


# ===========================================================================
# STAGE 3 - EDA market tools
# ===========================================================================

# --- 7) MISSING REPORT ----------------------------------------------------

def missing_report(
    df: pd.DataFrame,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 14.0,
    fig_height: float = 6.5,
    dpi: int = 110,
):
    """Comprehensive missing-values report per column.

    Returns a per-column summary with: dtype, n_total, n_missing, pct_missing,
    sample_value, and a heuristic ``recommendation`` based on pct_missing:
        0%             -> 'OK'
        <  5%          -> 'impute_mean/median/mode' (depending on dtype)
        5 - 30%        -> 'review'
        30 - 60%       -> 'consider_drop'
        > 60%          -> 'drop_column'

    Visual: bar chart of pct_missing per column + missingness pattern heatmap.

    Examples
    --------
    >>> dx.missing_report(df)
    """
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'df' must be a pandas DataFrame, got {type(df).__name__}")
    n_total = len(df)
    rows = []
    for c in df.columns:
        s = df[c]
        n_missing = int(s.isna().sum())
        pct = n_missing / n_total * 100 if n_total else 0.0
        dtype = str(s.dtype)
        is_numeric = pd.api.types.is_numeric_dtype(s)
        # Recommendation
        if pct == 0:
            rec = "OK"
        elif pct < 5:
            rec = "impute_mean" if is_numeric else "impute_mode"
            if is_numeric and abs(s.skew()) > 1:
                rec = "impute_median"
        elif pct < 30:
            rec = "review"
        elif pct < 60:
            rec = "consider_drop"
        else:
            rec = "drop_column"
        sample = s.dropna().head(1)
        sample_val = sample.iloc[0] if len(sample) else None
        rows.append({
            "column": c,
            "dtype": dtype,
            "n_total": n_total,
            "n_missing": n_missing,
            "pct_missing": pct,
            "sample_value": sample_val,
            "recommendation": rec,
        })
    summary = pd.DataFrame(rows).set_index("column")

    total_missing = int(summary["n_missing"].sum())
    n_cols_missing = int((summary["n_missing"] > 0).sum())
    rows_complete = int(df.dropna().shape[0])

    if show:
        _print_header(f"Missing values report for: {df_name}")
        int_cols = ("n_total", "n_missing")
        pct_cols = ("pct_missing",)
        _display(_format_summary(summary, decimals, int_cols, pct_cols))
        if n_total:
            pct_complete = rows_complete / n_total * 100
            print(f"\nDecision: {total_missing} missing cell(s) across "
                  f"{n_cols_missing}/{len(df.columns)} column(s).  "
                  f"{rows_complete:,} of {n_total:,} rows are complete "
                  f"({pct_complete:.{decimals}f}%).\n")
        else:
            print("\nDecision: empty DataFrame.\n")

    fig = None
    if plot:
        fig = _plot_missing_report(df, summary, fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_missing_report(df, summary, fig_width, fig_height, dpi, decimals):
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])

    # Bar chart: pct missing per column (sorted)
    s = summary["pct_missing"].sort_values(ascending=True)
    colors = []
    for v in s.values:
        if v == 0:
            colors.append("#2ca02c")  # green
        elif v < 5:
            colors.append("#9bd09a")  # light green
        elif v < 30:
            colors.append("#f7c873")  # amber
        elif v < 60:
            colors.append("#ec7853")  # orange
        else:
            colors.append("#c73e3e")  # red
    ax_bar.barh(s.index.astype(str), s.values, color=colors, edgecolor="black", linewidth=0.4)
    ax_bar.set_xlabel("% missing")
    ax_bar.set_title("Missing per column", fontsize=12, fontweight="bold")
    ax_bar.axvline(5, color="gray", linestyle=":", linewidth=0.8)
    ax_bar.axvline(30, color="gray", linestyle=":", linewidth=0.8)
    ax_bar.axvline(60, color="gray", linestyle=":", linewidth=0.8)
    for i, v in enumerate(s.values):
        ax_bar.text(v + 0.5, i, f"{v:.{decimals}f}%", va="center", fontsize=8)

    # Missingness heatmap
    mask = df.isna()
    if mask.shape[0] > 200:
        # Subsample rows for clarity on big data
        step = max(1, mask.shape[0] // 200)
        mask = mask.iloc[::step]
    sns.heatmap(mask.values.T, ax=ax_heat, cmap="Greys",
                cbar=False, yticklabels=df.columns, xticklabels=False)
    ax_heat.set_title("Missingness pattern (rows -> columns)", fontsize=12, fontweight="bold")
    ax_heat.set_xlabel("rows (subsampled if >200)")

    fig.suptitle("Missing Values Report", fontsize=14, fontweight="bold")
    return fig


# --- 8) FREQUENCY TABLE ---------------------------------------------------

def frequency_table(
    df: pd.DataFrame,
    col: str,
    top_n: Optional[int] = None,
    ascending: bool = False,
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
    """Frequency table for a categorical column with Pareto chart.

    Columns returned: category, count, pct, cumulative_count, cumulative_pct.

    Examples
    --------
    >>> dx.frequency_table(df, 'category')
    >>> dx.frequency_table(df, 'category', top_n=10)
    """
    if col not in df.columns:
        raise KeyError(f"Column not found: {col!r}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    s = df[col].dropna()
    n_total_valid = len(s)
    n_missing = int(df[col].isna().sum())
    counts = s.value_counts(ascending=ascending)
    if top_n is not None and top_n > 0:
        other_count = int(counts.iloc[top_n:].sum())
        counts = counts.iloc[:top_n]
        if other_count > 0:
            counts = pd.concat([counts, pd.Series({"<other>": other_count})])

    pct = counts / counts.sum() * 100 if counts.sum() else counts * 0
    cum_count = counts.cumsum()
    cum_pct = pct.cumsum()

    summary = pd.DataFrame({
        "count": counts.astype(int),
        "pct": pct,
        "cumulative_count": cum_count.astype(int),
        "cumulative_pct": cum_pct,
    })
    summary.index.name = col

    n_unique = int(df[col].nunique(dropna=True))
    top_cat = summary.index[0] if len(summary) else None
    top_pct = float(summary.iloc[0]["pct"]) if len(summary) else 0.0

    if show:
        _print_header(f"Frequency table for: {df_name}[{col}]"
                      + (f"  (top {top_n})" if top_n else ""))
        _display(_format_summary(summary, decimals,
                                 int_cols=("count", "cumulative_count"),
                                 pct_cols=("pct", "cumulative_pct")))
        print(f"\nDecision: {n_unique} unique value(s).  "
              f"Top category '{top_cat}' covers {top_pct:.{decimals}f}%.  "
              f"Missing: {n_missing}.\n")

    fig = None
    if plot:
        fig = _plot_frequency_table(summary, col, fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_frequency_table(summary, col, fig_width, fig_height, dpi, decimals):
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    x = np.arange(len(summary))
    labels = [str(i) for i in summary.index]
    ax1.bar(x, summary["count"].values, color="steelblue",
            edgecolor="black", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.set_ylabel("count", color="steelblue")
    ax1.set_xlabel(col)
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(x, summary["cumulative_pct"].values, color="darkorange",
             marker="o", linewidth=2.0, label="cumulative %")
    ax2.set_ylabel("cumulative %", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")
    ax2.axhline(80, color="red", linestyle=":", linewidth=1.0)
    ax2.set_ylim(0, 105)
    ax2.text(len(summary) - 1, 81, " 80%", fontsize=8, color="red")

    # Annotate counts above bars
    for xi, c in zip(x, summary["count"].values):
        ax1.text(xi, c, f"{int(c):,}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Frequency / Pareto of '{col}'", fontsize=13, fontweight="bold")
    return fig


# --- 9) CROSS TAB ---------------------------------------------------------

def cross_tab(
    df: pd.DataFrame,
    row: str,
    col: str,
    normalize: Optional[str] = None,
    alpha: float = 0.05,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    return_test: bool = False,
    cmap: str = "Blues",
    fig_width: float = 12.0,
    fig_height: float = 6.0,
    dpi: int = 110,
):
    """Bivariate categorical: contingency table + chi-square + Cramér's V.

    Cramér's V interpretation (Cohen):
        < 0.1 : negligible
        0.1 - 0.3 : weak
        0.3 - 0.5 : moderate
        >= 0.5 : strong

    Parameters
    ----------
    normalize : {'index', 'columns', 'all', None}
        Forwarded to ``pandas.crosstab`` for the displayed table.
    return_test : bool
        If True, return a dict with chi2, dof, p, cramers_v alongside the table.

    Examples
    --------
    >>> dx.cross_tab(df, row='gender', col='product')
    """
    if row not in df.columns:
        raise KeyError(f"row column not found: {row!r}")
    if col not in df.columns:
        raise KeyError(f"col column not found: {col!r}")
    if normalize not in (None, "index", "columns", "all"):
        raise ValueError(f"'normalize' must be None/'index'/'columns'/'all', got {normalize!r}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    valid = df[[row, col]].dropna()
    if len(valid) == 0:
        raise ValueError("No valid (non-NaN) rows for cross-tabulation.")

    observed = pd.crosstab(valid[row], valid[col])
    n_total = int(observed.values.sum())
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        raise ValueError("Cross-tab needs at least 2 categories in each axis.")

    chi2, p_value, dof, expected = _sst.chi2_contingency(observed.values)
    expected_df = pd.DataFrame(expected, index=observed.index, columns=observed.columns)
    cramers_v = float(np.sqrt(chi2 / (n_total * min(observed.shape[0] - 1, observed.shape[1] - 1))))

    if cramers_v < 0.1: strength = "negligible"
    elif cramers_v < 0.3: strength = "weak"
    elif cramers_v < 0.5: strength = "moderate"
    else: strength = "strong"

    display_table = observed if normalize is None else pd.crosstab(
        valid[row], valid[col], normalize=normalize)

    if show:
        _print_header(f"Cross-tab for: {df_name}[{row}] x [{col}]")
        if normalize:
            print(f"(normalized by '{normalize}')")
        if normalize is None:
            _display(observed)
        else:
            disp = display_table.copy()
            for c in disp.columns:
                disp[c] = disp[c].map(lambda v: f"{v * 100:.{decimals}f}%")
            _display(disp)
        sig = "Significant association" if p_value < alpha else "No significant association"
        print(f"\nChi-square test:  chi2 = {chi2:.{decimals}f}, dof = {dof}, p = {p_value:.4f}")
        print(f"Cramér's V:       {cramers_v:.{decimals}f}  ({strength})")
        print(f"Decision: {sig} (alpha={alpha}).  Effect size: {strength}.\n")

    fig = None
    if plot:
        fig = _plot_cross_tab(observed, expected_df, row, col, chi2, p_value, cramers_v,
                              strength, cmap, fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    test_result = {
        "chi2": float(chi2), "dof": int(dof), "p_value": float(p_value),
        "cramers_v": cramers_v, "strength": strength, "n": n_total,
        "expected": expected_df,
    }
    if return_test:
        if return_fig: return display_table, test_result, fig
        return display_table, test_result
    if return_df and return_fig: return display_table, fig
    if return_df: return display_table
    if return_fig: return fig
    return None


def _plot_cross_tab(observed, expected, row, col, chi2, p, v, strength,
                    cmap, fig_width, fig_height, dpi, decimals):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    sns.heatmap(observed, ax=axes[0], cmap=cmap, annot=True, fmt="d",
                cbar=False, linewidths=0.5, annot_kws={"fontsize": 9})
    axes[0].set_title(f"Observed counts", fontsize=12, fontweight="bold")
    axes[0].set_xlabel(col); axes[0].set_ylabel(row)

    sns.heatmap(expected, ax=axes[1], cmap=cmap, annot=True, fmt=".1f",
                cbar=False, linewidths=0.5, annot_kws={"fontsize": 9})
    axes[1].set_title(f"Expected counts (under H0 of independence)",
                      fontsize=12, fontweight="bold")
    axes[1].set_xlabel(col); axes[1].set_ylabel(row)

    fig.suptitle(f"{row} x {col}    chi2={chi2:.{decimals}f}, p={p:.4f}, "
                 f"V={v:.{decimals}f} ({strength})", fontsize=13, fontweight="bold")
    return fig


# --- 10) GROUP COMPARE ----------------------------------------------------

def group_compare(
    df: pd.DataFrame,
    group_col: str,
    value_cols: Optional[Union[str, Sequence[str]]] = None,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 14.0,
    fig_row_height: float = 4.0,
    dpi: int = 110,
):
    """Compare numeric metrics across groups defined by a categorical column.

    For each (group, value_col) pair, computes: n, mean, std, median, min, max, IQR.

    Examples
    --------
    >>> dx.group_compare(df, group_col='region', value_cols='sales')
    """
    if group_col not in df.columns:
        raise KeyError(f"group_col not found: {group_col!r}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    if value_cols is None:
        value_cols = [c for c in df.columns
                      if pd.api.types.is_numeric_dtype(df[c]) and c != group_col]
        if not value_cols:
            raise ValueError("No numeric columns to compare. Pass value_cols= explicitly.")
    elif isinstance(value_cols, str):
        value_cols = [value_cols]
    else:
        value_cols = list(value_cols)

    missing = [c for c in value_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Value columns not found: {missing}")

    parts = []
    for c in value_cols:
        sub = df[[group_col, c]].dropna()
        g = sub.groupby(group_col)[c]
        out = pd.DataFrame({
            "n": g.count().astype(int),
            "mean": g.mean(),
            "std": g.std(),
            "median": g.median(),
            "min": g.min(),
            "max": g.max(),
            "iqr": g.quantile(0.75) - g.quantile(0.25),
        })
        out.columns = pd.MultiIndex.from_product([[c], out.columns])
        parts.append(out)
    summary = pd.concat(parts, axis=1)
    summary.index.name = group_col

    if show:
        _print_header(f"Group comparison for: {df_name}  by '{group_col}'")
        # Flatten for printing
        disp = summary.copy()
        disp.columns = [f"{a}.{b}" for a, b in disp.columns]
        int_cols = [c for c in disp.columns if c.endswith(".n")]
        _display(_format_summary(disp, decimals, int_cols=int_cols))
        # Decision per value column
        for c in value_cols:
            means = summary[c]["mean"].dropna()
            if len(means):
                hi = means.idxmax(); lo = means.idxmin()
                print(f"Decision ({c}): highest mean in '{hi}' "
                      f"({means.max():.{decimals}f}), lowest in '{lo}' "
                      f"({means.min():.{decimals}f}).")
        print()

    fig = None
    if plot:
        fig = _plot_group_compare(df, group_col, value_cols, fig_width,
                                  fig_row_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_group_compare(df, group_col, value_cols, fig_width, row_height, dpi, decimals):
    n = len(value_cols)
    fig, axes = plt.subplots(n, 1, figsize=(fig_width, row_height * n), dpi=dpi)
    if n == 1:
        axes = [axes]
    for ax, c in zip(axes, value_cols):
        sub = df[[group_col, c]].dropna()
        groups = list(sub[group_col].unique())
        data = [sub[sub[group_col] == g][c].values for g in groups]
        # Violin + box overlay
        parts = ax.violinplot(data, showmeans=False, showmedians=False,
                              showextrema=False, widths=0.85)
        for body in parts["bodies"]:
            body.set_facecolor("lightblue")
            body.set_edgecolor("steelblue")
            body.set_alpha(0.7)
        bp = ax.boxplot(data, widths=0.25, patch_artist=True,
                        showfliers=True, medianprops={"color": "red", "linewidth": 2})
        for patch in bp["boxes"]:
            patch.set_facecolor("white")
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([str(g) for g in groups], rotation=20, ha="right")
        ax.set_ylabel(c)
        # Overlay mean labels
        for i, g in enumerate(groups, start=1):
            m = sub[sub[group_col] == g][c].mean()
            ax.text(i, m, f" mean={m:.{decimals}f}", color="darkred",
                    fontsize=8, va="center")
        ax.set_title(f"{c} by {group_col}", fontsize=12, fontweight="bold")

    fig.suptitle(f"Group comparison ({group_col})", fontsize=14, fontweight="bold")
    return fig


# Stage 3 aliases
missrep  = missing_report
freqtab  = frequency_table
xtab     = cross_tab
gcmp     = group_compare


# ===========================================================================
# STAGE 4 - Inference: confidence intervals + sample size
# ===========================================================================

def _to_array(data):
    """Coerce list-like/Series/array to a 1D float array, NaN-stripped."""
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D data; got shape {arr.shape}")
    return arr[~np.isnan(arr)]


# --- 11) CONFIDENCE INTERVAL FOR THE MEAN  (F-M04-L09-02 uses SE = s/sqrt(n))

def confidence_interval_mean(
    data,
    confidence: float = 0.95,
    decimals: int = 4,
    name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 10.0,
    fig_height: float = 3.0,
    dpi: int = 110,
):
    """t-based confidence interval for the population mean.

    Uses the Student-t distribution with n-1 degrees of freedom.
    CI = mean +/- t_{alpha/2, n-1} * SE,   SE = s / sqrt(n).
    Standard error formula: F-M04-L09-02.

    Examples
    --------
    >>> dx.confidence_interval_mean([12, 14, 11, 15, 13, 16])
    """
    if not (0 < confidence < 1):
        raise ValueError(f"'confidence' must be in (0, 1), got {confidence}")
    arr = _to_array(data)
    n = len(arr)
    if n < 2:
        raise ValueError(f"Need n >= 2; got {n}")

    alpha = 1.0 - confidence
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    t_crit = float(_sst.t.ppf(1 - alpha / 2, n - 1))
    margin = t_crit * se
    lower = mean - margin
    upper = mean + margin

    pct = int(round(confidence * 100))
    summary = pd.DataFrame(
        {"value": [n, mean, sd, se, t_crit, margin, lower, upper]},
        index=["n", "mean", "std", "SE", "t_critical", "margin_of_error",
               f"CI_lower_{pct}", f"CI_upper_{pct}"],
    )
    summary.index.name = "metric"

    label = name or "data"
    if show:
        _print_header(f"Confidence interval for the mean of: {label}  (confidence={confidence})")
        _display(_format_summary(summary, decimals))
        print(f"\nDecision: with {pct}% confidence, the population mean lies in "
              f"[{lower:.{decimals}f}, {upper:.{decimals}f}] "
              f"(point estimate = {mean:.{decimals}f}, margin = +/- {margin:.{decimals}f}).\n")

    fig = None
    if plot:
        fig = _plot_ci_mean(label, mean, lower, upper, n, confidence,
                            fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_ci_mean(label, mean, lower, upper, n, conf, fig_width, fig_height, dpi, decimals):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    y = 0
    ax.errorbar([mean], [y], xerr=[[mean - lower], [upper - mean]],
                fmt="o", color="steelblue", markersize=12, capsize=10,
                capthick=2.5, elinewidth=3.0)
    ax.scatter([mean], [y], s=160, color="darkblue", zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.axvline(mean, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(mean, y + 0.10, f"mean = {mean:.{decimals}f}",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(lower, y - 0.13, f"{lower:.{decimals}f}",
            ha="center", fontsize=9, color="steelblue")
    ax.text(upper, y - 0.13, f"{upper:.{decimals}f}",
            ha="center", fontsize=9, color="steelblue")
    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xlabel(str(label))
    ax.set_title(f"{int(round(conf * 100))}% Confidence interval for {label}  (n={n})",
                 fontsize=12, fontweight="bold")
    return fig


# --- 12) CONFIDENCE INTERVAL FOR A PROPORTION -----------------------------

def confidence_interval_proportion(
    successes: int,
    n: int,
    confidence: float = 0.95,
    method: str = "wilson",
    decimals: int = 4,
    name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 10.0,
    fig_height: float = 3.0,
    dpi: int = 110,
):
    """Confidence interval for a population proportion.

    Methods:
        'wald'   - normal approximation, classical. Fails for small n / extreme p_hat.
        'wilson' - score interval. Recommended for small n or extreme p_hat.
    """
    if not (0 < confidence < 1):
        raise ValueError(f"'confidence' must be in (0, 1), got {confidence}")
    if n <= 0:
        raise ValueError(f"'n' must be > 0, got {n}")
    if not (0 <= successes <= n):
        raise ValueError(f"Need 0 <= successes <= n, got successes={successes}, n={n}")
    if method not in ("wald", "wilson"):
        raise ValueError(f"'method' must be 'wald' or 'wilson', got {method!r}")

    alpha = 1.0 - confidence
    z = float(_sst.norm.ppf(1 - alpha / 2))
    p_hat = successes / n

    if method == "wald":
        se = float(np.sqrt(p_hat * (1 - p_hat) / n))
        margin = z * se
        lower = max(0.0, p_hat - margin)
        upper = min(1.0, p_hat + margin)
    else:
        z2 = z * z
        denom = 1.0 + z2 / n
        center = (p_hat + z2 / (2 * n)) / denom
        margin = (z / denom) * float(np.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n)))
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)

    pct = int(round(confidence * 100))
    summary = pd.DataFrame(
        {"value": [successes, n, p_hat, z, lower, upper, upper - lower]},
        index=["successes", "n", "p_hat", "z_critical",
               f"CI_lower_{pct}", f"CI_upper_{pct}", "width"],
    )
    summary.index.name = "metric"

    label = name or "proportion"
    if show:
        _print_header(f"Confidence interval for {label}  (method={method}, confidence={confidence})")
        _display(_format_summary(summary, decimals))
        print(f"\nDecision: with {pct}% confidence, the population proportion lies in "
              f"[{lower:.{decimals}f}, {upper:.{decimals}f}] "
              f"(p_hat = {p_hat:.{decimals}f}, method = {method}).\n")

    fig = None
    if plot:
        fig = _plot_ci_prop(label, p_hat, lower, upper, n, method, confidence,
                            fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_ci_prop(label, p_hat, lower, upper, n, method, conf, fig_width, fig_height, dpi, decimals):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    y = 0
    ax.errorbar([p_hat], [y], xerr=[[p_hat - lower], [upper - p_hat]],
                fmt="o", color="darkgreen", markersize=12, capsize=10,
                capthick=2.5, elinewidth=3.0)
    ax.scatter([p_hat], [y], s=160, color="darkgreen", zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.axvline(p_hat, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(p_hat, y + 0.10, f"p_hat = {p_hat:.{decimals}f}",
            ha="center", fontsize=10, fontweight="bold")
    ax.text(lower, y - 0.13, f"{lower:.{decimals}f}",
            ha="center", fontsize=9, color="darkgreen")
    ax.text(upper, y - 0.13, f"{upper:.{decimals}f}",
            ha="center", fontsize=9, color="darkgreen")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_xlabel("proportion")
    ax.set_title(f"{int(round(conf * 100))}% {method.title()} CI for {label}  (n={n})",
                 fontsize=12, fontweight="bold")
    return fig


# --- 13) SAMPLE SIZE FOR A MEAN -------------------------------------------

def sample_size_mean(
    margin_error: float,
    std: float,
    confidence: float = 0.95,
    decimals: int = 2,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 10.0,
    fig_height: float = 4.5,
    dpi: int = 110,
):
    """Required sample size to estimate a mean within +/- margin_error.

    Formula: n = (z_{alpha/2} * sigma / E)^2
    """
    if margin_error <= 0:
        raise ValueError(f"'margin_error' must be > 0, got {margin_error}")
    if std <= 0:
        raise ValueError(f"'std' must be > 0, got {std}")
    if not (0 < confidence < 1):
        raise ValueError(f"'confidence' must be in (0, 1), got {confidence}")

    alpha = 1.0 - confidence
    z = float(_sst.norm.ppf(1 - alpha / 2))
    n_exact = (z * std / margin_error) ** 2
    n_required = int(np.ceil(n_exact))

    summary = pd.DataFrame(
        {"value": [margin_error, std, confidence, z, n_exact, n_required]},
        index=["margin_error", "std", "confidence", "z_critical",
               "n_exact", "n_required (rounded up)"],
    )
    summary.index.name = "metric"

    if show:
        _print_header(f"Sample size for the mean  (E={margin_error}, std={std}, conf={confidence})")
        _display(_format_summary(summary, decimals))
        print(f"\nDecision: at {int(round(confidence*100))}% confidence and std = {std}, "
              f"you need n = {n_required} observations to estimate the mean within "
              f"+/- {margin_error}.\n")

    fig = None
    if plot:
        fig = _plot_ss_mean(margin_error, std, confidence, z, n_required,
                            fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_ss_mean(E, std, conf, z, n_required, fig_width, fig_height, dpi, decimals):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    e_values = np.linspace(E * 0.3, E * 3.0, 200)
    n_values = (z * std / e_values) ** 2
    ax.plot(e_values, n_values, color="steelblue", linewidth=2.0)
    ax.axvline(E, color="red", linestyle="--", linewidth=1.5)
    ax.axhline(n_required, color="red", linestyle="--", linewidth=1.5)
    ax.scatter([E], [n_required], color="red", s=140, zorder=5,
               edgecolor="white", linewidth=1.5)
    ax.annotate(f"n = {n_required}  at E = {E}",
                xy=(E, n_required), xytext=(E * 1.4, n_required * 1.5),
                fontsize=11, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.set_xlabel("Margin of error E")
    ax.set_ylabel("Required n (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"Sample size vs margin  (std = {std}, {int(round(conf*100))}% confidence)",
                 fontsize=12, fontweight="bold")
    return fig


# --- 14) SAMPLE SIZE FOR A PROPORTION  (F-M04-L07-01) ----------------------

def sample_size_proportion(
    margin_error: float,
    p: float = 0.5,
    confidence: float = 0.95,
    decimals: int = 2,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 12.0,
    fig_height: float = 4.5,
    dpi: int = 110,
):
    """Required sample size to estimate a proportion within +/- margin_error.

    Formula (F-M04-L07-01):
        n = z^2 * p * (1 - p) / E^2

    Worst-case p = 0.5 gives the largest n. Always reported alongside the
    user-specified p so survey planners see both.
    """
    if margin_error <= 0:
        raise ValueError(f"'margin_error' must be > 0, got {margin_error}")
    if not (0 < p < 1):
        raise ValueError(f"'p' must be in (0, 1), got {p}")
    if not (0 < confidence < 1):
        raise ValueError(f"'confidence' must be in (0, 1), got {confidence}")

    alpha = 1.0 - confidence
    z = float(_sst.norm.ppf(1 - alpha / 2))
    n_exact = z * z * p * (1 - p) / (margin_error ** 2)
    n_required = int(np.ceil(n_exact))
    n_worst = z * z * 0.25 / (margin_error ** 2)
    n_worst_int = int(np.ceil(n_worst))

    summary = pd.DataFrame(
        {"value": [margin_error, p, confidence, z, n_exact, n_required,
                   n_worst, n_worst_int]},
        index=["margin_error", "p_estimated", "confidence", "z_critical",
               "n_exact", "n_required (rounded up)",
               "n_worst_case_exact (p=0.5)", "n_worst_case (rounded up)"],
    )
    summary.index.name = "metric"

    if show:
        _print_header(f"Sample size for a proportion  (E={margin_error}, p={p}, conf={confidence})")
        _display(_format_summary(summary, decimals))
        print(f"\nDecision: at {int(round(confidence*100))}% confidence, you need n = "
              f"{n_required} for p = {p}, or n = {n_worst_int} in the worst case "
              f"(p = 0.5). (Formula F-M04-L07-01.)\n")

    fig = None
    if plot:
        fig = _plot_ss_prop(margin_error, p, confidence, z, n_required,
                            fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_ss_prop(E, p, conf, z, n_required, fig_width, fig_height, dpi, decimals):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    # Panel 1: n vs E
    e_values = np.linspace(E * 0.3, E * 3.0, 200)
    n_values = z * z * p * (1 - p) / (e_values ** 2)
    ax1.plot(e_values, n_values, color="steelblue", linewidth=2.0)
    ax1.axvline(E, color="red", linestyle="--", linewidth=1.5)
    ax1.scatter([E], [n_required], color="red", s=140, zorder=5,
                edgecolor="white", linewidth=1.5)
    ax1.annotate(f"n = {n_required}", xy=(E, n_required),
                 xytext=(E * 1.4, n_required * 1.5),
                 fontsize=10, color="red", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax1.set_xlabel("Margin E")
    ax1.set_ylabel("Required n (log)")
    ax1.set_yscale("log")
    ax1.set_title(f"n vs E  (p = {p}, {int(round(conf*100))}%)",
                  fontsize=11, fontweight="bold")

    # Panel 2: n vs p
    p_values = np.linspace(0.01, 0.99, 99)
    n_values2 = z * z * p_values * (1 - p_values) / (E ** 2)
    ax2.plot(p_values, n_values2, color="darkgreen", linewidth=2.0)
    ax2.axvline(0.5, color="red", linestyle="--", linewidth=1.5, label="worst case p=0.5")
    ax2.axvline(p, color="orange", linestyle=":", linewidth=2.0, label=f"your p={p}")
    ax2.scatter([p], [n_required], color="orange", s=140, zorder=5,
                edgecolor="white", linewidth=1.5)
    ax2.scatter([0.5], [z * z * 0.25 / (E ** 2)], color="red", s=140, zorder=5,
                edgecolor="white", linewidth=1.5)
    ax2.set_xlabel("Estimated proportion p")
    ax2.set_ylabel("Required n")
    ax2.set_title(f"n vs p  (E = {E})", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("Sample size for a proportion (F-M04-L07-01)",
                 fontsize=13, fontweight="bold")
    return fig


# Stage 4 aliases
cim  = confidence_interval_mean
cip  = confidence_interval_proportion
ssm  = sample_size_mean
ssp  = sample_size_proportion


# ===========================================================================
# STAGE 5 - Hypothesis tests
# ===========================================================================

_VALID_ALTERNATIVES = ("two-sided", "less", "greater")


def _ttest_plot(t_stat, df_val, alpha, alternative, p, fig_width, fig_height, dpi, title):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    x_max = max(4.0, abs(t_stat) * 1.5)
    x = np.linspace(-x_max, x_max, 500)
    y = _sst.t.pdf(x, df_val)
    ax.plot(x, y, color="steelblue", linewidth=2.0)
    ax.fill_between(x, 0, y, color="lightblue", alpha=0.25)
    if alternative == "two-sided":
        t_crit = float(_sst.t.ppf(1 - alpha / 2, df_val))
        ax.fill_between(x[x <= -t_crit], 0, y[x <= -t_crit], color="red", alpha=0.4,
                        label="reject region")
        ax.fill_between(x[x >= t_crit], 0, y[x >= t_crit], color="red", alpha=0.4)
        ax.axvline(-t_crit, color="red", linestyle="--", linewidth=1.2)
        ax.axvline(t_crit, color="red", linestyle="--", linewidth=1.2)
    elif alternative == "greater":
        t_crit = float(_sst.t.ppf(1 - alpha, df_val))
        ax.fill_between(x[x >= t_crit], 0, y[x >= t_crit], color="red", alpha=0.4,
                        label="reject region")
        ax.axvline(t_crit, color="red", linestyle="--", linewidth=1.2)
    else:
        t_crit = float(_sst.t.ppf(alpha, df_val))
        ax.fill_between(x[x <= t_crit], 0, y[x <= t_crit], color="red", alpha=0.4,
                        label="reject region")
        ax.axvline(t_crit, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(t_stat, color="black", linestyle="-", linewidth=2.5,
               label=f"observed t = {t_stat:.3f}")
    ax.set_xlabel("t"); ax.set_ylabel("density")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"t-distribution  df={df_val:.1f}, alpha={alpha}, p={p:.4f}",
                 fontsize=12, fontweight="bold")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    return fig


# --- 15) NORMALITY TEST ---------------------------------------------------

def normality_test(
    data,
    method: str = "auto",
    alpha: float = 0.05,
    decimals: int = 4,
    name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 12.0,
    fig_height: float = 4.5,
    dpi: int = 110,
):
    """Test whether data come from a Normal distribution.

    H0: data are Normally distributed.
    p < alpha => Reject H0 (data are NOT Normal).

    Methods:
        'auto'        : Shapiro-Wilk if n < 5000, else D'Agostino-Pearson.
        'shapiro'     : Shapiro-Wilk.
        'normaltest'  : D'Agostino-Pearson omnibus.
        'jarque_bera' : Jarque-Bera (based on skew/kurt).
    """
    arr = _to_array(data)
    n = len(arr)
    if n < 3:
        raise ValueError(f"Need n >= 3; got {n}")
    if method == "auto":
        method = "shapiro" if n < 5000 else "normaltest"
    if method not in ("shapiro", "normaltest", "jarque_bera"):
        raise ValueError(f"'method' must be one of shapiro/normaltest/jarque_bera/auto, got {method!r}")

    if method == "shapiro":
        res = _sst.shapiro(arr)
        stat, p = float(res.statistic), float(res.pvalue)
        test_label = "Shapiro-Wilk"
    elif method == "normaltest":
        res = _sst.normaltest(arr)
        stat, p = float(res.statistic), float(res.pvalue)
        test_label = "D'Agostino-Pearson"
    else:
        res = _sst.jarque_bera(arr)
        stat, p = float(res.statistic), float(res.pvalue)
        test_label = "Jarque-Bera"

    skew_v = float(_sst.skew(arr))
    kurt_v = float(_sst.kurtosis(arr))
    decision = "Reject H0 (NOT Normal)" if p < alpha else "Fail to reject H0 (Normal)"

    summary = pd.DataFrame({"value": [
        n, test_label, stat, p, skew_v, kurt_v, alpha, decision,
    ]}, index=[
        "n", "test", "statistic", "p_value", "skew", "kurt_excess",
        "alpha", "decision",
    ])
    summary.index.name = "metric"

    label = name or "data"
    if show:
        _print_header(f"Normality test for: {label}  ({test_label}, alpha={alpha})")
        _display(_format_summary(summary, decimals))
        if p < alpha:
            print(f"\nDecision: Reject H0 (p = {p:.4f} < {alpha}).  "
                  f"Data are NOT Normal -- consider non-parametric tests.\n")
        else:
            print(f"\nDecision: Fail to reject H0 (p = {p:.4f} >= {alpha}).  "
                  f"Data are consistent with Normal.\n")

    fig = None
    if plot:
        fig = _plot_normality(arr, p, alpha, test_label, label,
                              fig_width, fig_height, dpi)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_normality(arr, p, alpha, test_label, label, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    ax = axes[0]
    sns.histplot(arr, bins=30, color="steelblue", edgecolor="black",
                 alpha=0.7, ax=ax, stat="density")
    if arr.std() > 0:
        xx = np.linspace(arr.min(), arr.max(), 200)
        ax.plot(xx, _sst.norm.pdf(xx, arr.mean(), arr.std()),
                color="red", linewidth=2.0, label="Normal fit")
    ax.set_xlabel(label); ax.set_ylabel("density")
    ax.set_title("Histogram + Normal fit", fontweight="bold")
    ax.legend()

    ax = axes[1]
    (osm, osr), (slope_qq, intercept_qq, r_qq) = _sst.probplot(arr, dist="norm")
    ax.scatter(osm, osr, color="purple", alpha=0.5, s=25,
               edgecolor="black", linewidth=0.4)
    ax.plot(osm, slope_qq * osm + intercept_qq, color="red", linewidth=1.5)
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    ax.set_title("Q-Q plot", fontweight="bold")

    status = "NOT Normal" if p < alpha else "Normal"
    fig.suptitle(f"Normality test - {test_label}: p={p:.4f} -> {status}",
                 fontsize=13, fontweight="bold")
    return fig


# --- 16) ONE-SAMPLE t-TEST ------------------------------------------------

def t_test_one_sample(
    data,
    popmean: float,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    decimals: int = 4,
    name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 10.0,
    fig_height: float = 4.5,
    dpi: int = 110,
):
    """One-sample t-test against a hypothesized population mean."""
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(f"'alternative' must be one of {_VALID_ALTERNATIVES}, got {alternative!r}")
    arr = _to_array(data)
    n = len(arr)
    if n < 2:
        raise ValueError(f"Need n >= 2; got {n}")

    res = _sst.ttest_1samp(arr, popmean=popmean, alternative=alternative)
    t_stat = float(res.statistic)
    p = float(res.pvalue)
    df_val = n - 1

    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    se = sd / np.sqrt(n)
    cohens_d = (mean - popmean) / sd if sd > 0 else float("nan")
    t_crit_ci = float(_sst.t.ppf(0.975, df_val))
    ci_low = mean - t_crit_ci * se
    ci_high = mean + t_crit_ci * se
    decision = "Reject H0" if p < alpha else "Fail to reject H0"

    summary = pd.DataFrame({"value": [
        n, mean, popmean, mean - popmean, sd, se, t_stat, df_val, p, cohens_d,
        ci_low, ci_high, alpha, alternative, decision,
    ]}, index=[
        "n", "sample_mean", "popmean (H0)", "mean_diff", "std", "SE",
        "t_statistic", "df", "p_value", "Cohen's d", "CI_low_95", "CI_high_95",
        "alpha", "alternative", "decision",
    ])
    summary.index.name = "metric"

    label = name or "data"
    if show:
        _print_header(f"One-sample t-test for: {label}  "
                      f"(H0: mean = {popmean}, alt = {alternative})")
        _display(_format_summary(summary, decimals))
        if p < alpha:
            print(f"\nDecision: Reject H0 (p = {p:.4f} < {alpha}).  "
                  f"Sample mean {mean:.{decimals}f} differs from {popmean} "
                  f"(Cohen's d = {cohens_d:+.{decimals}f}).\n")
        else:
            print(f"\nDecision: Fail to reject H0 (p = {p:.4f} >= {alpha}).  "
                  f"No evidence sample mean differs from {popmean}.\n")

    fig = None
    if plot:
        fig = _ttest_plot(t_stat, df_val, alpha, alternative, p,
                          fig_width, fig_height, dpi,
                          title=f"One-sample t-test: {label} vs {popmean}")
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


# --- 17) TWO-SAMPLE t-TEST (F-M05-L06-01) ---------------------------------

def t_test_two_sample(
    group1,
    group2,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    equal_var: bool = False,
    decimals: int = 4,
    name1: Optional[str] = None,
    name2: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 14.0,
    fig_height: float = 4.5,
    dpi: int = 110,
):
    """Independent two-sample t-test. Welch's (equal_var=False) by default.

    Source: F-M05-L06-01.
    """
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(f"'alternative' must be one of {_VALID_ALTERNATIVES}, got {alternative!r}")
    a1 = _to_array(group1)
    a2 = _to_array(group2)
    n1, n2 = len(a1), len(a2)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Need both groups n >= 2; got n1={n1}, n2={n2}")

    res = _sst.ttest_ind(a1, a2, equal_var=equal_var, alternative=alternative)
    t_stat = float(res.statistic)
    p = float(res.pvalue)
    df_val = float(res.df) if (hasattr(res, "df") and res.df is not None) else float(n1 + n2 - 2)

    m1, m2 = float(a1.mean()), float(a2.mean())
    sd1, sd2 = float(a1.std(ddof=1)), float(a2.std(ddof=1))
    diff = m1 - m2
    pooled_sd = float(np.sqrt(((n1 - 1) * sd1 ** 2 + (n2 - 1) * sd2 ** 2) / (n1 + n2 - 2)))
    cohens_d = diff / pooled_sd if pooled_sd > 0 else float("nan")
    se_diff = float(np.sqrt(sd1 ** 2 / n1 + sd2 ** 2 / n2))
    t_crit_ci = float(_sst.t.ppf(0.975, df_val))
    ci_low = diff - t_crit_ci * se_diff
    ci_high = diff + t_crit_ci * se_diff

    label1 = name1 or "group1"
    label2 = name2 or "group2"
    test_name = "Welch's t-test" if not equal_var else "Student's t-test"
    decision = "Reject H0" if p < alpha else "Fail to reject H0"

    summary = pd.DataFrame({"value": [
        n1, n2, m1, m2, diff, sd1, sd2, se_diff, t_stat, df_val, p, cohens_d,
        ci_low, ci_high, alpha, alternative, decision,
    ]}, index=[
        f"n_{label1}", f"n_{label2}", f"mean_{label1}", f"mean_{label2}",
        "mean_diff", f"std_{label1}", f"std_{label2}", "SE_diff",
        "t_statistic", "df", "p_value", "Cohen's d",
        "CI_low_95_diff", "CI_high_95_diff",
        "alpha", "alternative", "decision",
    ])
    summary.index.name = "metric"

    if show:
        _print_header(f"Two-sample t-test ({test_name}, alt = {alternative}, alpha = {alpha})")
        _display(_format_summary(summary, decimals))
        if p < alpha:
            print(f"\nDecision: Reject H0 (p = {p:.4f} < {alpha}).  "
                  f"Means differ.  diff = {diff:+.{decimals}f}, Cohen's d = {cohens_d:+.{decimals}f}.\n")
        else:
            print(f"\nDecision: Fail to reject H0 (p = {p:.4f} >= {alpha}).  "
                  f"No significant difference.\n")

    fig = None
    if plot:
        fig = _plot_ttest_two(a1, a2, t_stat, df_val, alpha, alternative,
                              label1, label2, p, fig_width, fig_height, dpi)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_ttest_two(a1, a2, t_stat, df_val, alpha, alternative, label1, label2, p,
                    fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=dpi)
    # Distributions
    ax = axes[0]
    sns.kdeplot(a1, ax=ax, color="steelblue", linewidth=2.0, label=label1)
    sns.kdeplot(a2, ax=ax, color="darkorange", linewidth=2.0, label=label2)
    ax.axvline(a1.mean(), color="steelblue", linestyle="--", linewidth=1.2)
    ax.axvline(a2.mean(), color="darkorange", linestyle="--", linewidth=1.2)
    ax.set_title("Distributions", fontweight="bold")
    ax.legend()
    # Box plots
    ax = axes[1]
    bp = ax.boxplot([a1, a2], labels=[label1, label2], patch_artist=True,
                    medianprops={"color": "red", "linewidth": 2})
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("moccasin")
    ax.set_title("Box plots", fontweight="bold")
    # t-distribution
    ax = axes[2]
    x_max = max(4.0, abs(t_stat) * 1.5)
    x = np.linspace(-x_max, x_max, 500)
    y = _sst.t.pdf(x, df_val)
    ax.plot(x, y, color="steelblue", linewidth=2.0)
    if alternative == "two-sided":
        t_crit = float(_sst.t.ppf(1 - alpha / 2, df_val))
        ax.fill_between(x[x <= -t_crit], 0, y[x <= -t_crit], color="red", alpha=0.4)
        ax.fill_between(x[x >= t_crit], 0, y[x >= t_crit], color="red", alpha=0.4)
    elif alternative == "greater":
        t_crit = float(_sst.t.ppf(1 - alpha, df_val))
        ax.fill_between(x[x >= t_crit], 0, y[x >= t_crit], color="red", alpha=0.4)
    else:
        t_crit = float(_sst.t.ppf(alpha, df_val))
        ax.fill_between(x[x <= t_crit], 0, y[x <= t_crit], color="red", alpha=0.4)
    ax.axvline(t_stat, color="black", linestyle="-", linewidth=2.5)
    ax.set_title(f"t = {t_stat:.3f}, p = {p:.4f}", fontweight="bold")
    fig.suptitle(f"Two-sample t-test: {label1} vs {label2}",
                 fontsize=13, fontweight="bold")
    return fig


# --- 18) PAIRED t-TEST ---------------------------------------------------

def t_test_paired(
    before,
    after,
    alternative: str = "two-sided",
    alpha: float = 0.05,
    decimals: int = 4,
    name_before: Optional[str] = None,
    name_after: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 12.0,
    fig_height: float = 4.5,
    dpi: int = 110,
):
    """Paired t-test (before vs after, matched pairs)."""
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(f"'alternative' must be one of {_VALID_ALTERNATIVES}, got {alternative!r}")
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    if b.shape != a.shape:
        raise ValueError(f"before and after must have same shape, got {b.shape} vs {a.shape}")
    mask = ~(np.isnan(b) | np.isnan(a))
    b = b[mask]; a = a[mask]
    n = len(b)
    if n < 2:
        raise ValueError(f"Need n >= 2 pairs; got {n}")

    diff = a - b
    mean_diff = float(diff.mean())
    sd_diff = float(diff.std(ddof=1))
    se_diff = sd_diff / np.sqrt(n)
    res = _sst.ttest_rel(a, b, alternative=alternative)
    t_stat = float(res.statistic); p = float(res.pvalue); df_val = n - 1

    cohens_d = mean_diff / sd_diff if sd_diff > 0 else float("nan")
    t_crit_ci = float(_sst.t.ppf(0.975, df_val))
    ci_low = mean_diff - t_crit_ci * se_diff
    ci_high = mean_diff + t_crit_ci * se_diff

    nb = name_before or "before"
    na = name_after or "after"
    decision = "Reject H0" if p < alpha else "Fail to reject H0"

    summary = pd.DataFrame({"value": [
        n, float(b.mean()), float(a.mean()), mean_diff, sd_diff, se_diff,
        t_stat, df_val, p, cohens_d, ci_low, ci_high, alpha, alternative, decision,
    ]}, index=[
        "n_pairs", f"mean_{nb}", f"mean_{na}", f"mean_diff ({na} - {nb})",
        "std_diff", "SE_diff", "t_statistic", "df", "p_value", "Cohen's d",
        "CI_low_95_diff", "CI_high_95_diff",
        "alpha", "alternative", "decision",
    ])
    summary.index.name = "metric"

    if show:
        _print_header(f"Paired t-test: {na} vs {nb}  (alt = {alternative}, alpha = {alpha})")
        _display(_format_summary(summary, decimals))
        if p < alpha:
            print(f"\nDecision: Reject H0 (p = {p:.4f} < {alpha}).  "
                  f"Mean change = {mean_diff:+.{decimals}f}, "
                  f"Cohen's d = {cohens_d:+.{decimals}f}.\n")
        else:
            print(f"\nDecision: Fail to reject H0 (p = {p:.4f} >= {alpha}).  "
                  f"No significant change.\n")

    fig = None
    if plot:
        fig = _plot_ttest_paired(b, a, diff, t_stat, alpha, p, nb, na,
                                 fig_width, fig_height, dpi)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_ttest_paired(b, a, diff, t_stat, alpha, p, nb, na, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    ax = axes[0]
    for bi, ai in zip(b, a):
        color = "green" if ai > bi else "red" if ai < bi else "gray"
        ax.plot([0, 1], [bi, ai], color=color, alpha=0.4, linewidth=0.8)
    ax.scatter(np.zeros(len(b)), b, color="steelblue", s=30, alpha=0.6, zorder=3, label=nb)
    ax.scatter(np.ones(len(a)),  a, color="darkorange", s=30, alpha=0.6, zorder=3, label=na)
    ax.set_xticks([0, 1]); ax.set_xticklabels([nb, na])
    ax.set_ylabel("value")
    ax.set_title("Pairs (green = up, red = down)", fontweight="bold")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    sns.histplot(diff, bins=20, color="purple", edgecolor="black",
                 alpha=0.7, ax=ax, stat="count")
    ax.axvline(0, color="black", linestyle="-", linewidth=1.5, label="no change")
    ax.axvline(diff.mean(), color="red", linestyle="--", linewidth=2.0,
               label=f"mean diff = {diff.mean():.3f}")
    ax.set_xlabel(f"{na} - {nb}")
    ax.set_title(f"Differences  (t = {t_stat:.3f}, p = {p:.4f})", fontweight="bold")
    ax.legend()
    fig.suptitle(f"Paired t-test: {na} vs {nb}", fontsize=13, fontweight="bold")
    return fig


# --- 19) ONE-WAY ANOVA ----------------------------------------------------

def anova_oneway(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    alpha: float = 0.05,
    decimals: int = 4,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 14.0,
    fig_height: float = 4.8,
    dpi: int = 110,
):
    """One-way ANOVA: compare means of >= 2 groups."""
    if group_col not in df.columns:
        raise KeyError(f"group_col not found: {group_col!r}")
    if value_col not in df.columns:
        raise KeyError(f"value_col not found: {value_col!r}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    sub = df[[group_col, value_col]].dropna()
    groups = sub.groupby(group_col)[value_col]
    group_names = list(groups.groups.keys())
    n_groups = len(group_names)
    if n_groups < 2:
        raise ValueError(f"Need >= 2 groups; got {n_groups}")
    arrays = [groups.get_group(g).to_numpy() for g in group_names]

    f_res = _sst.f_oneway(*arrays)
    f_stat, p = float(f_res.statistic), float(f_res.pvalue)
    n_total = sum(len(arr) for arr in arrays)
    df1 = n_groups - 1
    df2 = n_total - n_groups

    grand_mean = float(sub[value_col].mean())
    ss_between = float(sum(len(arr) * (arr.mean() - grand_mean) ** 2 for arr in arrays))
    ss_total = float(((sub[value_col] - grand_mean) ** 2).sum())
    eta2 = ss_between / ss_total if ss_total > 0 else float("nan")

    desc_rows = []
    for g, arr in zip(group_names, arrays):
        desc_rows.append({
            "group": g, "n": len(arr), "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min": float(arr.min()), "max": float(arr.max()),
        })
    desc = pd.DataFrame(desc_rows).set_index("group")

    decision = "Reject H0" if p < alpha else "Fail to reject H0"
    summary = pd.DataFrame({"value": [
        n_groups, n_total, df1, df2, f_stat, p, eta2, alpha, decision,
    ]}, index=[
        "n_groups", "n_total", "df_between", "df_within",
        "F_statistic", "p_value", "eta_squared", "alpha", "decision",
    ])
    summary.index.name = "metric"

    if show:
        _print_header(f"One-way ANOVA: {df_name}[{value_col}] by {group_col}  (alpha = {alpha})")
        _display(_format_summary(summary, decimals))
        print("\nGroup descriptives:")
        _display(_format_summary(desc, decimals, int_cols=("n",)))
        if p < alpha:
            effect = "small" if eta2 < 0.06 else ("medium" if eta2 < 0.14 else "large")
            print(f"\nDecision: Reject H0 (p = {p:.4f} < {alpha}).  "
                  f"Group means differ significantly.  "
                  f"eta² = {eta2:.{decimals}f} ({effect} effect).\n")
        else:
            print(f"\nDecision: Fail to reject H0 (p = {p:.4f} >= {alpha}).  "
                  f"No significant differences across groups.\n")

    fig = None
    if plot:
        fig = _plot_anova(sub, group_col, value_col, group_names, arrays,
                          f_stat, df1, df2, alpha, p, fig_width, fig_height, dpi)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_anova(sub, group_col, value_col, group_names, arrays,
                f_stat, df1, df2, alpha, p, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    ax = axes[0]
    bp = ax.boxplot(arrays, labels=[str(g) for g in group_names],
                    patch_artist=True, medianprops={"color": "red", "linewidth": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    for i, arr in enumerate(arrays, 1):
        ax.scatter([i], [arr.mean()], color="darkorange", s=80, zorder=4,
                   marker="D", edgecolor="black", linewidth=0.8)
    ax.axhline(sub[value_col].mean(), color="black", linestyle=":", linewidth=1.0,
               label="grand mean")
    ax.set_ylabel(value_col); ax.set_xlabel(group_col)
    ax.set_title("Group distributions", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1]
    x_max = max(8.0, f_stat * 1.5)
    x = np.linspace(0.01, x_max, 500)
    y = _sst.f.pdf(x, df1, df2)
    ax.plot(x, y, color="steelblue", linewidth=2.0)
    f_crit = float(_sst.f.ppf(1 - alpha, df1, df2))
    ax.fill_between(x[x >= f_crit], 0, y[x >= f_crit], color="red", alpha=0.4,
                    label="reject region")
    ax.axvline(f_crit, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(f_stat, color="black", linestyle="-", linewidth=2.5,
               label=f"observed F = {f_stat:.3f}")
    ax.set_xlabel("F"); ax.set_ylabel("density")
    ax.set_title(f"F-distribution (df1={df1}, df2={df2})  p = {p:.4f}",
                 fontweight="bold")
    ax.legend()

    fig.suptitle(f"One-way ANOVA: {value_col} by {group_col}",
                 fontsize=13, fontweight="bold")
    return fig


# --- 20) CHI-SQUARE INDEPENDENCE ------------------------------------------

def chi_square_independence(
    df: pd.DataFrame,
    row: str,
    col: str,
    alpha: float = 0.05,
    decimals: int = 4,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 12.0,
    fig_height: float = 4.8,
    dpi: int = 110,
):
    """Chi-square test of independence between two categoricals."""
    if row not in df.columns:
        raise KeyError(f"row not found: {row!r}")
    if col not in df.columns:
        raise KeyError(f"col not found: {col!r}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    valid = df[[row, col]].dropna()
    observed = pd.crosstab(valid[row], valid[col])
    n_total = int(observed.values.sum())
    if observed.shape[0] < 2 or observed.shape[1] < 2:
        raise ValueError("Need at least 2 categories in each axis.")

    chi2, p, dof, expected = _sst.chi2_contingency(observed.values)
    chi2 = float(chi2); p = float(p)
    cramers_v = float(np.sqrt(chi2 / (n_total * min(observed.shape[0] - 1, observed.shape[1] - 1))))
    if cramers_v < 0.1: strength = "negligible"
    elif cramers_v < 0.3: strength = "weak"
    elif cramers_v < 0.5: strength = "moderate"
    else: strength = "strong"
    decision = "Reject H0" if p < alpha else "Fail to reject H0"

    summary = pd.DataFrame({"value": [
        n_total, observed.shape[0], observed.shape[1], chi2, dof, p,
        cramers_v, strength, alpha, decision,
    ]}, index=[
        "n", "rows", "cols", "chi2_statistic", "df", "p_value",
        "cramers_v", "effect_size", "alpha", "decision",
    ])
    summary.index.name = "metric"

    if show:
        _print_header(f"Chi-square test of independence: "
                      f"{df_name}[{row}] x [{col}]  (alpha = {alpha})")
        _display(_format_summary(summary, decimals))
        if p < alpha:
            print(f"\nDecision: Reject H0 (p = {p:.4f} < {alpha}).  "
                  f"{row} and {col} are associated "
                  f"(Cramer's V = {cramers_v:.{decimals}f}, {strength}).\n")
        else:
            print(f"\nDecision: Fail to reject H0 (p = {p:.4f} >= {alpha}).  "
                  f"No evidence of association.\n")

    fig = None
    if plot:
        fig = _plot_chi2_test(chi2, dof, alpha, p, row, col,
                              fig_width, fig_height, dpi)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_chi2_test(chi2, dof, alpha, p, row, col, fig_width, fig_height, dpi):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    x_max = max(float(_sst.chi2.ppf(0.999, dof)) * 1.1, chi2 * 1.2)
    x = np.linspace(0.01, x_max, 500)
    y = _sst.chi2.pdf(x, dof)
    ax.plot(x, y, color="steelblue", linewidth=2.0)
    chi2_crit = float(_sst.chi2.ppf(1 - alpha, dof))
    ax.fill_between(x[x >= chi2_crit], 0, y[x >= chi2_crit], color="red", alpha=0.4,
                    label="reject region")
    ax.axvline(chi2_crit, color="red", linestyle="--", linewidth=1.2)
    ax.axvline(chi2, color="black", linestyle="-", linewidth=2.5,
               label=f"observed chi² = {chi2:.3f}")
    ax.set_xlabel("chi²"); ax.set_ylabel("density")
    ax.set_title(f"Chi-square distribution (df = {dof})  p = {p:.4f}",
                 fontweight="bold")
    ax.legend()
    fig.suptitle(f"Chi-square test of independence: {row} vs {col}",
                 fontsize=13, fontweight="bold")
    return fig


# Stage 5 aliases
normtest = normality_test
t1       = t_test_one_sample
t2       = t_test_two_sample
tpair    = t_test_paired
aov1     = anova_oneway
chi2ind  = chi_square_independence


# ===========================================================================
# STAGE 6 - ML diagnostic tools
# ===========================================================================

# --- 21) VIF SCORES (multicollinearity diagnostic) ------------------------

def _vif_for_column(X: np.ndarray, i: int) -> float:
    """Variance Inflation Factor for column i of X.

    Fits an OLS regression of X[:, i] on all other columns (plus intercept) and
    returns 1 / (1 - R^2). Returns +inf when columns are perfectly collinear.
    """
    y = X[:, i]
    X_others = np.delete(X, i, axis=1)
    X_design = np.column_stack([np.ones(len(y)), X_others])
    beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
    y_pred = X_design @ beta
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    r2 = 1.0 - ss_res / ss_tot
    if r2 >= 1.0 - 1e-12:
        return float("inf")
    return 1.0 / (1.0 - r2)


def vif_scores(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    threshold: float = 10.0,
    decimals: int = 2,
    df_name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 11.0,
    fig_height: float = 5.0,
    dpi: int = 110,
):
    """Variance Inflation Factor for every numeric column.

    VIF_i = 1 / (1 - R^2_i)  where R^2_i is the R^2 of regressing column i
    on all other selected numeric columns.

    Rule of thumb:
        VIF < 5     -> no multicollinearity concern
        5 <= VIF <  threshold -> review
        VIF >= threshold (default 10) -> drop or combine the feature

    Used before fitting any linear / logistic regression.

    Examples
    --------
    >>> dx.vif_scores(df[['x1', 'x2', 'x3']])
    """
    if threshold <= 1:
        raise ValueError(f"'threshold' must be > 1, got {threshold}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    cols_resolved = resolve_columns(df, cols, numeric_only=True)
    if len(cols_resolved) < 2:
        raise ValueError("Need at least 2 numeric columns to compute VIF.")
    num = to_numeric_frame(df[cols_resolved].copy()).dropna()
    if len(num) < len(cols_resolved) + 1:
        raise ValueError(
            f"Need n >= n_columns+1; got n={len(num)}, cols={len(cols_resolved)}"
        )

    X = num.to_numpy(dtype=float)
    rows = []
    for i, c in enumerate(cols_resolved):
        vif = _vif_for_column(X, i)
        if pd.isna(vif):
            decision = "constant_column"
        elif vif < 5:
            decision = "OK"
        elif vif < threshold:
            decision = "review"
        elif vif == float("inf"):
            decision = "perfect_collinearity"
        else:
            decision = "drop_or_combine"
        rows.append({"feature": c, "VIF": vif, "decision": decision})

    summary = pd.DataFrame(rows).set_index("feature")

    n_drop = int((summary["decision"].isin(["drop_or_combine", "perfect_collinearity"])).sum())
    n_review = int((summary["decision"] == "review").sum())

    if show:
        _print_header(f"VIF scores for: {df_name}  (threshold = {threshold})")
        _display(_format_summary(summary, decimals))
        if n_drop > 0:
            print(f"\nDecision: {n_drop} feature(s) at or above VIF = {threshold} "
                  f"-- drop or combine before modeling.  "
                  f"{n_review} feature(s) flagged for review (VIF 5-{threshold}).\n")
        elif n_review > 0:
            print(f"\nDecision: no severe multicollinearity, but {n_review} feature(s) "
                  f"in the review zone (VIF 5-{threshold}).\n")
        else:
            print(f"\nDecision: all features below VIF = 5  --  no multicollinearity concern.\n")

    fig = None
    if plot:
        fig = _plot_vif(summary, threshold, fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_vif(summary, threshold, fig_width, fig_height, dpi, decimals):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    s = summary["VIF"].replace([np.inf], np.nan).fillna(0).sort_values(ascending=True)
    # Replace zero placeholders with a small value to show on log scale (if used)
    colors = []
    for v in s.values:
        if v < 5:
            colors.append("#2ca02c")     # green
        elif v < threshold:
            colors.append("#f7c873")     # amber
        else:
            colors.append("#c73e3e")     # red
    bars = ax.barh(s.index.astype(str), s.values, color=colors,
                   edgecolor="black", linewidth=0.4)
    ax.axvline(5,         color="gray",  linestyle=":",  linewidth=1.0, label="VIF = 5")
    ax.axvline(threshold, color="red",   linestyle="--", linewidth=1.5,
               label=f"threshold = {threshold}")
    ax.set_xlabel("VIF (Variance Inflation Factor)")
    ax.set_title("Multicollinearity diagnostic (VIF per feature)",
                 fontsize=12, fontweight="bold")
    ax.legend()
    for i, v in enumerate(s.values):
        label = f"{v:.{decimals}f}" if v < 1e6 else "inf"
        ax.text(v + max(s.values) * 0.01, i, label, va="center", fontsize=9)
    return fig


# --- 22) CLASS IMBALANCE --------------------------------------------------

def class_imbalance(
    target,
    decimals: int = 2,
    name: Optional[str] = None,
    show: bool = True,
    plot: bool = True,
    return_df: bool = False,
    return_fig: bool = False,
    fig_width: float = 12.0,
    fig_height: float = 4.5,
    dpi: int = 110,
):
    """Class balance report for a classification target.

    Computes count, percentage, and imbalance ratio (majority / minority).
    Severity classification:
        ratio <  2   -> 'balanced'
        2-5          -> 'mild'
        5-10         -> 'moderate'
        10-50        -> 'severe'
        >= 50        -> 'extreme'

    Examples
    --------
    >>> dx.class_imbalance(df['target'])
    """
    s = pd.Series(target).dropna()
    if s.empty:
        raise ValueError("target is empty after dropping NaN.")
    counts = s.value_counts()
    if len(counts) < 2:
        raise ValueError(f"Need at least 2 classes; got {len(counts)}.")
    total = int(counts.sum())
    pct = counts / total * 100
    majority = int(counts.max())
    minority = int(counts.min())
    ratio = majority / minority if minority > 0 else float("inf")

    if ratio < 2:
        severity = "balanced"
    elif ratio < 5:
        severity = "mild"
    elif ratio < 10:
        severity = "moderate"
    elif ratio < 50:
        severity = "severe"
    else:
        severity = "extreme"

    summary = pd.DataFrame({
        "count": counts.astype(int),
        "pct": pct,
        "imbalance_ratio_vs_majority": (majority / counts).astype(float),
    })
    summary.index.name = "class"

    label = name or "target"
    if show:
        _print_header(f"Class imbalance for: {label}")
        _display(_format_summary(summary, decimals,
                                 int_cols=("count",),
                                 pct_cols=("pct",)))
        recs = {
            "balanced":  "no special action needed.",
            "mild":      "consider stratified sampling.",
            "moderate":  "use class_weight in your model or up/down sample.",
            "severe":    "use SMOTE / undersampling and/or weighted loss.",
            "extreme":   "anomaly-detection framing may fit better; SMOTE alone is insufficient.",
        }
        print(f"\nDecision: {len(counts)} classes, majority:minority = "
              f"{majority}:{minority} (ratio = {ratio:.{decimals}f}) -- {severity}.  "
              f"Recommendation: {recs[severity]}\n")

    fig = None
    if plot:
        fig = _plot_imbalance(counts, pct, severity, ratio,
                              fig_width, fig_height, dpi, decimals)
    _finalize_figure(fig, show, plot, return_fig)

    if return_df and return_fig: return summary, fig
    if return_df: return summary
    if return_fig: return fig
    return None


def _plot_imbalance(counts, pct, severity, ratio, fig_width, fig_height, dpi, decimals):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    sev_colors = {
        "balanced":  "#2ca02c",
        "mild":      "#9bd09a",
        "moderate":  "#f7c873",
        "severe":    "#ec7853",
        "extreme":   "#c73e3e",
    }
    color = sev_colors.get(severity, "steelblue")

    # Bar chart
    ax = axes[0]
    s = counts.sort_values(ascending=False)
    bars = ax.bar([str(i) for i in s.index], s.values, color=color,
                  edgecolor="black", alpha=0.85)
    for bar, c in zip(bars, s.values):
        ax.text(bar.get_x() + bar.get_width() / 2, c,
                f"{int(c):,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title(f"Class counts  (ratio = {ratio:.{decimals}f}, {severity})",
                 fontweight="bold")

    # Donut / pie
    ax = axes[1]
    p = pct.sort_values(ascending=False)
    wedges, _, autotexts = ax.pie(p.values, labels=[str(i) for i in p.index],
                                   autopct="%.1f%%", startangle=90,
                                   wedgeprops=dict(width=0.4, edgecolor="white"))
    ax.set_title("Class share", fontweight="bold")

    fig.suptitle(f"Class imbalance  --  {severity.upper()}",
                 fontsize=13, fontweight="bold",
                 color=sev_colors.get(severity, "black"))
    return fig


# Stage 6 aliases
vif       = vif_scores
imbalance = class_imbalance
