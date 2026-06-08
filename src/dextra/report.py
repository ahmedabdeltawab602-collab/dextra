"""One-call EDA report for dextra - Phase 9 of the Roadmap.

The HTML RENDERER over the neutral composition layer (:mod:`dextra._compose`).
:func:`edareport` asks ``_compose`` for the report's sections (each a table +
base64 figure + ``Decision:`` line, produced by reusing the tested functions of
Phases 1-8 -- it computes nothing new) and lays them out as a single
SELF-CONTAINED HTML file: every figure embedded as a base64 PNG, every table as
inline HTML -- no sidecar assets, no new dependency.

Design highlights (see REPORT_PHILOSOPHY.md):

* One call -> one portable ``report.html``.
* Reuse verbatim: the sections are ``_compose`` builders; nothing is
  re-implemented here. scikit-learn (optional model section) is imported lazily.
* Section isolation: a section that cannot run is SKIPPED with a recorded
  reason; the rest of the report is still produced.
* Immutability: the input DataFrame is never mutated.
"""

from __future__ import annotations

import html as _htmllib
from typing import Optional, Sequence

import pandas as pd

from ._compose import _BUILDERS, _SECTION_ORDER
from ._utils import _ensure_pandas, append_audit, get_variable_name, now_iso
from ._version import __version__


def _esc(text) -> str:
    return _htmllib.escape(str(text))


def _df_html(frame: pd.DataFrame, decimals: int) -> str:
    def _fmt(v):
        try:
            return f"{float(v):,.{decimals}f}"
        except (TypeError, ValueError):
            return _esc(v)
    return frame.to_html(border=0, classes="dx-table", float_format=_fmt,
                         na_rep="-", escape=True)


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

    generated_at = now_iso()
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
    append_audit(out_copy, {
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
