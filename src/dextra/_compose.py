"""Neutral composition layer for dextra reports & dashboards (Phases 9-10).

The section BUILDERS live here, independent of any output format. They call the
tested functions of Phases 1-8 and return renderer-agnostic *blocks*; the HTML
renderer (:mod:`dextra.report`) and the Streamlit renderer
(:mod:`dextra.dashboard`) both consume them. This is the single source of truth
for *what* a report/dashboard contains; each renderer decides *how* to show it.

A builder returns ``(title, [block, ...])`` where a block is the tuple
``(subtitle, table_or_None, img_b64_or_None, decision_or_None, note_or_None)``.
Images are carried as base64 PNG strings -- a format both renderers consume
(HTML embeds them inline; Streamlit decodes them for ``st.image``). A builder
raises to signal "skip this section"; the caller records the reason.
"""

from __future__ import annotations

import base64
import inspect
import io
from contextlib import redirect_stdout
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SECTION_ORDER = ("overview", "quality", "univariate", "bivariate", "model")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_audit(out: pd.DataFrame, entry: dict) -> None:
    out.attrs.setdefault("dextra_audit", [])
    out.attrs["dextra_audit"] = list(out.attrs["dextra_audit"])
    out.attrs["dextra_audit"].append(entry)


def _fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _call(fn, *args, **kwargs):
    """Call ``fn`` passing only the keyword arguments it actually accepts."""
    sig = set(inspect.signature(fn).parameters)
    kw = {k: v for k, v in kwargs.items() if k in sig}
    return fn(*args, **kw)


def _split_ret(ret):
    """Pick the first DataFrame and first matplotlib Figure from a return."""
    items = ret if isinstance(ret, tuple) else (ret,)
    frame = next((x for x in items if isinstance(x, pd.DataFrame)), None)
    fig = next((x for x in items if isinstance(x, plt.Figure)), None)
    return frame, fig


def _extract_decision(text: str):
    for line in text.splitlines():
        if line.strip().startswith("Decision:"):
            return line.split("Decision:", 1)[1].strip()
    return None


def _run(fn, *args, **kwargs):
    """Run a dextra function, capturing its table, figure and Decision line.

    ``show`` is enabled only when it cannot trigger a blocking ``plt.show()``
    (i.e. the function supports ``return_fig`` or has no ``plot`` flag), so the
    composition never pops a window while harvesting the ``Decision:`` sentence.
    The figure is converted to a base64 PNG and closed immediately.
    """
    sig = set(inspect.signature(fn).parameters)
    call_kw = dict(kwargs)
    if "plot" in sig:
        call_kw["plot"] = True
    if "return_df" in sig:
        call_kw["return_df"] = True
    if "return_fig" in sig:
        call_kw["return_fig"] = True
    if "show" in sig:
        call_kw["show"] = ("return_fig" in sig) or ("plot" not in sig)
    buf = io.StringIO()
    with redirect_stdout(buf):
        ret = _call(fn, *args, **call_kw)
    frame, fig = _split_ret(ret)
    img = _fig_to_b64(fig) if fig is not None else None
    return frame, img, _extract_decision(buf.getvalue())


# ---------------------------------------------------------------------------
# Section builders (renderer-agnostic)
# ---------------------------------------------------------------------------

def _sec_overview(df, ctx):
    n, m = df.shape
    mem = float(df.memory_usage(deep=True).sum())
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "nulls": df.isna().sum(),
        "null_%": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=True),
    })
    n_missing_cols = int((df.isna().sum() > 0).sum())
    decision = (f"{n:,} rows x {m} cols, {mem / 1e6:.2f} MB in memory; "
                f"{n_missing_cols} column(s) contain missing values.")
    return "Overview", [(None, info, None, decision, None)]


def _sec_quality(df, ctx):
    if len(df) == 0:
        raise ValueError("no rows")
    blocks = []
    from .stats_advanced import missing_report
    t, f, d = _run(missing_report, df)
    blocks.append(("Missing values", t, f, d, None))

    dups = int(df.duplicated().sum())
    dtab = pd.DataFrame(
        {"value": [dups, round(dups / len(df) * 100, 2)]},
        index=["duplicate_rows", "duplicate_%"])
    blocks.append(("Duplicate rows", dtab, None,
                   f"{dups} duplicate row(s) ({dups / len(df) * 100:.2f}%).",
                   None))

    if df.select_dtypes("number").shape[1] >= 1:
        from .stats_advanced import outliers_report
        t, f, d = _run(outliers_report, df)
        blocks.append(("Outliers", t, f, d, None))
    return "Data quality", blocks


def _sec_univariate(df, ctx):
    blocks = []
    num = df.select_dtypes("number")
    if num.shape[1] >= 1:
        from .stats import describe_numeric
        t, _, d = _run(describe_numeric, df)
        blocks.append(("Numeric summary", t, None, d, None))
        from .plots import plot_histograms
        cols = list(num.columns)[: ctx["max_hist"]]
        ret = _call(plot_histograms, df, cols=cols, show=False, plot=True,
                    return_fig=True)
        _, fig = _split_ret(ret)
        if fig is not None:
            blocks.append(("Distributions", None, _fig_to_b64(fig), None,
                           None))
    cat = df.select_dtypes(exclude="number")
    if cat.shape[1] >= 1:
        from .stats_advanced import frequency_table
        for col in list(cat.columns)[: ctx["top_cat"]]:
            try:
                t, f, d = _run(frequency_table, df, col)
                blocks.append((f"Frequencies - {col}", t, f, d, None))
            except Exception:  # noqa: BLE001 - one bad column never aborts
                continue
    if not blocks:
        raise ValueError("no numeric or categorical columns to summarise")
    return "Univariate", blocks


def _sec_bivariate(df, ctx):
    blocks = []
    if df.select_dtypes("number").shape[1] >= 2:
        from .stats_advanced import correlation_matrix
        t, f, d = _run(correlation_matrix, df)
        blocks.append(("Correlation matrix", t, f, d, None))
    target = ctx["target"]
    if (target and target in df.columns
            and not pd.api.types.is_numeric_dtype(df[target])):
        try:
            from .stats_advanced import class_imbalance
            t, f, d = _run(class_imbalance, df[target])
            blocks.append((f"Class balance - {target}", t, f, d, None))
        except Exception:  # noqa: BLE001
            pass
    if not blocks:
        raise ValueError("need >= 2 numeric columns for a correlation matrix")
    return "Bivariate", blocks


def _sec_model(df, ctx):
    if not ctx["include_model"]:
        raise ValueError("include_model=False")
    target = ctx["target"]
    if not target or target not in df.columns:
        raise ValueError("a valid target= is required for the model section")
    y = df[target]
    is_reg = pd.api.types.is_numeric_dtype(y) and int(y.nunique()) > 10

    rng = np.random.default_rng(0)
    order = np.arange(len(df))
    rng.shuffle(order)
    cut = max(1, int(len(df) * 0.75))
    tr = df.iloc[order[:cut]]
    te = df.iloc[order[cut:]]
    if len(te) == 0:
        raise ValueError("not enough rows for a train/test split")

    if is_reg:
        from .evaluation import residual_analysis
        from .modeling import regress
        ret = _call(regress, tr, y=target, method="forest", show=False,
                    plot=False, return_params=True)
        params = ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else None
        t, f, d = _run(residual_analysis, te, params=params)
        title_block = "Baseline regression (held-out test)"
    else:
        from .evaluation import confusion_report
        from .modeling import classify
        ret = _call(classify, tr, y=target, method="forest", show=False,
                    plot=False, return_params=True)
        params = ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else None
        t, f, d = _run(confusion_report, te, params=params)
        title_block = "Baseline classification (held-out test)"
    note = ("Random-forest baseline trained on a 75% split; metrics are on the "
            "held-out 25% test split. A floor, not a tuned final model.")
    return "Baseline model & evaluation", [(title_block, t, f, d, note)]


_BUILDERS = {
    "overview": _sec_overview,
    "quality": _sec_quality,
    "univariate": _sec_univariate,
    "bivariate": _sec_bivariate,
    "model": _sec_model,
}
