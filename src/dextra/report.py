"""One-call EDA report for dextra - Phase 9 of the Roadmap.

Implements the report framework documented in REPORT_PHILOSOPHY.md at the
project root. The report COMPUTES NOTHING NEW: :func:`edareport` orchestrates the
already-tested functions of Phases 1-8, captures their tables, figures and
``Decision:`` sentences, and lays them out as a single SELF-CONTAINED HTML file
(every figure embedded as a base64 PNG, every table as inline HTML -- no sidecar
assets, no new dependency).

Design highlights (see REPORT_PHILOSOPHY.md):

* One call -> one portable ``report.html``.
* Reuse verbatim: each section calls the canonical dextra function and embeds
  exactly what it produced; the function's ``Decision:`` line becomes the
  caption. scikit-learn (optional model section) is imported lazily.
* Section isolation: a section that cannot run is SKIPPED with a recorded
  reason; the rest of the report is still produced.
* Immutability: the input DataFrame is never mutated.

Stage 9.1 - HTML framework + Overview + Data-quality ; 9.2 - Univariate +
Bivariate ; 9.3 - optional target-aware Model / Evaluation section.
"""

from __future__ import annotations

import base64
import html as _htmllib
import inspect
import io
from contextlib import redirect_stdout
from datetime import datetime, timezone
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._utils import _ensure_pandas, get_variable_name
from ._version import __version__

_SECTION_ORDER = ("overview", "quality", "univariate", "bivariate", "model")


# ---------------------------------------------------------------------------
# Small local helpers (kept dependency-light so report imports without scipy)
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_audit(out: pd.DataFrame, entry: dict) -> None:
    out.attrs.setdefault("dextra_audit", [])
    out.attrs["dextra_audit"] = list(out.attrs["dextra_audit"])
    out.attrs["dextra_audit"].append(entry)


def _esc(text) -> str:
    return _htmllib.escape(str(text))


def _fig_to_b64(fig) -> str:
    """Render a matplotlib figure to a base64 PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _df_html(frame: pd.DataFrame, decimals: int) -> str:
    def _fmt(v):
        try:
            return f"{float(v):,.{decimals}f}"
        except (TypeError, ValueError):
            return _esc(v)
    return frame.to_html(border=0, classes="dx-table", float_format=_fmt,
                         na_rep="-", escape=True)


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
    report never pops a window while harvesting the ``Decision:`` sentence.
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
# Section builders. Each returns (title, [block, ...]); a block is the tuple
# (subtitle, table_or_None, img_b64_or_None, decision_or_None, note).
# A builder raises to signal "skip this section" (the reason is recorded).
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


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

_CSS = """
:root { --ink:#1f2d3d; --muted:#6b7a8d; --line:#e3e8ee; --brand:#1f4e79;
        --band:#eef3f8; }
* { box-sizing: border-box; }
body { margin:0; background:#f6f8fa; color:var(--ink);
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,
       Arial,sans-serif; line-height:1.5; }
.container { max-width:1040px; margin:0 auto; padding:32px 24px 64px; }
header.report { border-bottom:3px solid var(--brand); margin-bottom:8px; }
header.report h1 { color:var(--brand); margin:0 0 4px; font-size:26px; }
header.report .meta { color:var(--muted); font-size:13px; margin-bottom:16px; }
section { background:#fff; border:1px solid var(--line); border-radius:10px;
          padding:20px 22px; margin:18px 0; box-shadow:0 1px 2px rgba(0,0,0,.03); }
section > h2 { color:var(--brand); margin:0 0 12px; font-size:20px;
               border-bottom:1px solid var(--line); padding-bottom:8px; }
h3 { color:var(--ink); font-size:15px; margin:18px 0 6px; }
p.decision { background:var(--band); border-left:4px solid var(--brand);
             padding:8px 12px; border-radius:0 6px 6px 0; font-size:14px;
             margin:6px 0 12px; }
p.note { color:var(--muted); font-size:13px; margin:2px 0 10px; }
img { max-width:100%; height:auto; display:block; margin:10px 0; border-radius:6px; }
.table-wrap { overflow-x:auto; margin:8px 0 12px; }
table.dx-table { border-collapse:collapse; font-size:13px; width:auto; }
table.dx-table th, table.dx-table td { border:1px solid var(--line);
            padding:6px 10px; text-align:right; }
table.dx-table th { background:var(--brand); color:#fff; font-weight:600; }
table.dx-table tbody tr:nth-child(even) { background:var(--band); }
.toc { font-size:14px; }
.toc a { color:var(--brand); text-decoration:none; margin-right:14px; }
.skip { color:var(--muted); font-style:italic; font-size:13px; }
footer { color:var(--muted); font-size:12px; text-align:center; margin-top:24px; }
"""


def _render_block(block, decimals) -> str:
    subtitle, table, img, decision, note = block
    parts = []
    if subtitle:
        parts.append(f"<h3>{_esc(subtitle)}</h3>")
    if note:
        parts.append(f'<p class="note">{_esc(note)}</p>')
    if decision:
        parts.append(f'<p class="decision"><strong>Decision:</strong> '
                     f'{_esc(decision)}</p>')
    if table is not None:
        parts.append(f'<div class="table-wrap">{_df_html(table, decimals)}</div>')
    if img is not None:
        parts.append(f'<img alt="{_esc(subtitle or "figure")}" '
                     f'src="data:image/png;base64,{img}"/>')
    return "\n".join(parts)


def _render_document(title, header_meta, sections, skipped, decimals) -> str:
    toc = " ".join(
        f'<a href="#{key}">{_esc(name)}</a>' for key, name, _ in sections)
    body = []
    for key, name, blocks in sections:
        inner = "\n".join(_render_block(b, decimals) for b in blocks)
        body.append(f'<section id="{key}"><h2>{_esc(name)}</h2>\n{inner}\n</section>')
    if skipped:
        items = " ".join(f"{_esc(k)} ({_esc(r)})" for k, r in skipped)
        body.append(f'<section id="skipped"><h2>Skipped sections</h2>'
                    f'<p class="skip">{items}</p></section>')
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"/>"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"/>"
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        f'<div class="container"><header class="report"><h1>{_esc(title)}</h1>'
        f'<div class="meta">{header_meta}</div>'
        f'<div class="toc">{toc}</div></header>'
        f"{''.join(body)}"
        f'<footer>Generated by dextra v{_esc(__version__)}</footer>'
        "</div></body></html>")


# ===========================================================================
# 9  edareport  --  one-call self-contained HTML EDA report
# ===========================================================================

def edareport(
    df: pd.DataFrame,
    *,
    out: str = "report.html",
    target: Optional[str] = None,
    title: Optional[str] = None,
    sections: Optional[Sequence[str]] = None,
    include_model: bool = False,
    max_hist: int = 24,
    top_cat: int = 10,
    theme: str = "light",
    return_params: bool = False,
    show: bool = True,
    decimals: int = 4,
    df_name: Optional[str] = None,
):
    """Compose a single self-contained HTML EDA report in one call.

    Orchestrates the tested functions of Phases 1-8 -- it computes nothing new.
    Builds, in order: an Overview, a Data-quality section (missing / duplicates
    / outliers), a Univariate section (numeric summary + histograms + top
    categorical frequencies), a Bivariate section (correlation, plus class
    balance when a categorical ``target`` is given), and -- when
    ``include_model=True`` and a ``target`` is supplied -- a Baseline model &
    evaluation section (a random-forest baseline trained on a split and judged
    on the held-out test split, via Phases 6-7). Every figure is embedded as a
    base64 PNG and every table inline, so the output is one portable file.

    Sections are isolated: any that cannot run is skipped with a recorded
    reason and the rest of the report is still written. The input DataFrame is
    never mutated.

    Parameters
    ----------
    df : pandas.DataFrame
        The data to report on. Never mutated.
    out : str
        Path of the HTML file to write.
    target : str, optional
        A column used by the class-balance and (optional) model sections.
    title : str, optional
        Report title (defaults to ``"dextra EDA report - <df_name>"``).
    sections : sequence of str, optional
        Subset of ``{"overview","quality","univariate","bivariate","model"}``
        to build (default: all applicable).
    include_model : bool
        Build the optional baseline model / evaluation section (needs ``target``
        and scikit-learn).
    max_hist : int
        Cap on the number of numeric columns plotted as histograms.
    top_cat : int
        Number of categorical columns to tabulate as frequency tables.
    theme : str
        Visual theme (``"light"``).
    return_params : bool
        Return the JSON-safe build manifest instead of the output path.
    show : bool
        Print the one-line ``Decision:`` summary.
    decimals : int
        Numeric formatting precision in the embedded tables.
    df_name : str, optional
        Name used in the title / audit (inferred when omitted).

    Returns
    -------
    str or dict
        The output path, or -- when ``return_params=True`` -- the manifest.

    Examples
    --------
    >>> dx.edareport(df)                                   # writes report.html
    >>> dx.edareport(df, target='churn', include_model=True)
    >>> manifest = dx.edareport(df, return_params=True)
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)
    if title is None:
        title = f"dextra EDA report - {df_name}"

    if sections is None:
        wanted = list(_SECTION_ORDER)
    else:
        wanted = [s for s in _SECTION_ORDER if s in set(sections)]
    if "model" not in wanted and include_model and (
            sections is None or "model" in set(sections)):
        wanted.append("model")
    if not include_model and "model" in wanted:
        wanted.remove("model")

    ctx = {"target": target, "include_model": include_model,
           "max_hist": int(max_hist), "top_cat": int(top_cat),
           "decimals": int(decimals)}

    built, skipped, manifest_sections = [], [], {}
    for key in wanted:
        builder = _BUILDERS.get(key)
        if builder is None:
            continue
        try:
            name, blocks = builder(df, ctx)
            if not blocks:
                raise ValueError("no content")
            built.append((key, name, blocks))
            manifest_sections[key] = {
                "status": "built",
                "decisions": [b[3] for b in blocks if b[3]],
            }
        except Exception as exc:  # noqa: BLE001 - section isolation by design
            reason = str(exc) or exc.__class__.__name__
            skipped.append((key, reason))
            manifest_sections[key] = {"status": "skipped", "reason": reason}

    generated_at = _now_iso()
    n, m = df.shape
    header_meta = (f"{n:,} rows &times; {m} columns &middot; generated "
                   f"{generated_at} &middot; {len(built)} section(s)")
    document = _render_document(title, header_meta, built, skipped, decimals)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(document)

    decision = (f"Report written to '{out}': {len(built)} section(s) built"
                + (f", {len(skipped)} skipped" if skipped else "") + ".")

    out_copy = df.copy()
    out_copy.attrs = dict(df.attrs)
    _append_audit(out_copy, {
        "stage": "report", "function": "edareport",
        "timestamp": generated_at,
        "params": {"out": out, "target": target,
                   "include_model": bool(include_model),
                   "sections_built": [k for k, _, _ in built]},
        "decision": decision,
    })

    if show:
        print(f"Decision: {decision}")

    if return_params:
        return {
            "function": "edareport",
            "out": out,
            "title": title,
            "sections": manifest_sections,
            "metadata": {"n_rows": int(n), "n_cols": int(m),
                         "target": target, "include_model": bool(include_model),
                         "theme": theme},
            "version": __version__,
            "generated_at": generated_at,
        }
    return out


# Short alias, consistent with the underscore-free Phase 8 / 9 naming.
edarep = edareport
