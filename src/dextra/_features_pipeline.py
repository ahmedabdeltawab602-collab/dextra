"""dextra features - the featpipe pipeline wrapper."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from ._features_common import (
    _display,
    _finalize_figure,
    _fmt_table,
    _print_header,
    _ret_pack,
)
from ._features_derive import aggfeat, cross, dtfeats
from ._features_discretize import bin, encode
from ._features_numeric import scale, transform
from ._utils import _ensure_pandas, append_audit, get_variable_name, now_iso
from ._version import __version__

_FEATPIPE_DISPATCH = {
    "transform": transform,
    "scale": scale,
    "bin": bin,
    "encode": encode,
    "dtfeats": dtfeats,
    "cross": cross,
    "aggfeat": aggfeat,
}


_FEATPIPE_CONTROL_FLAGS = (
    "return_params", "return_df", "return_fig", "show", "plot", "params",
    "df_name", "decimals", "fig_width", "fig_height", "dpi",
)


_FEATPIPE_INPLACE_FNS = ("transform", "scale", "bin")


def _featpipe_compare_key(fn_name: str) -> str:
    """Return the keyword that selects the method for a given function."""
    return "agg" if fn_name == "aggfeat" else "method"


def _featpipe_validate_steps(steps) -> list:
    """Validate a steps list; return a clean list of ``(fn_name, kwargs)``.

    Rejects malformed steps, unknown function names, and any step that asks
    for ``compare`` mode (compare is exploratory and single-function only;
    featpipe is the commit tool). Control flags owned by featpipe are stripped
    with a warning so they cannot collide with featpipe's own call.
    """
    if not isinstance(steps, (list, tuple)) or len(steps) == 0:
        raise ValueError(
            "featpipe fit mode requires 'steps' to be a non-empty list of "
            "dicts, e.g. [{'fn': 'scale', 'cols': ['price'], "
            "'method': 'robust'}].")
    clean = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(
                f"featpipe step {idx}: each step must be a dict, got "
                f"{type(step).__name__}.")
        if "fn" not in step:
            raise ValueError(
                f"featpipe step {idx}: missing required key 'fn' (one of "
                f"{tuple(_FEATPIPE_DISPATCH)}).")
        fn_name = step["fn"]
        if fn_name not in _FEATPIPE_DISPATCH:
            raise ValueError(
                f"featpipe step {idx}: unknown fn {fn_name!r}; valid "
                f"functions are {tuple(_FEATPIPE_DISPATCH)}.")
        kwargs = {k: v for k, v in step.items() if k != "fn"}
        # Stage 4.4 decision: compare mode is forbidden inside a pipeline.
        ckey = _featpipe_compare_key(fn_name)
        if str(kwargs.get(ckey, "")).lower() == "compare":
            raise ValueError(
                f"featpipe step {idx} (fn={fn_name}): {ckey}='compare' is not "
                f"allowed inside a pipeline. featpipe is a commit tool -- "
                f"explore options with {fn_name}({ckey}='compare') on its own "
                f"first, then chain the chosen {ckey} here.")
        collided = [k for k in kwargs if k in _FEATPIPE_CONTROL_FLAGS]
        if collided:
            warnings.warn(
                f"featpipe step {idx} (fn={fn_name}): control flag(s) "
                f"{collided} are managed by featpipe and were ignored.",
                stacklevel=3)
            kwargs = {k: v for k, v in kwargs.items()
                      if k not in _FEATPIPE_CONTROL_FLAGS}
        clean.append((fn_name, kwargs))
    return clean


def _plot_featpipe(summary_rows, input_cols, df_name, mode,
                   fig_width, fig_height, dpi):
    """Two-panel visual: columns added per step + DataFrame width growth."""
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    labels = [f"{i}:{r['fn']}" for i, r in enumerate(summary_rows)]
    added = [r["cols_added"] for r in summary_rows]
    after = [r["cols_after"] for r in summary_rows]

    ax0 = axes[0]
    ax0.bar(labels, added, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax0.set_ylabel("new columns")
    ax0.set_title("Columns added per step", fontweight="bold")
    ax0.tick_params(axis="x", rotation=45)
    for i, v in enumerate(added):
        ax0.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    ax1 = axes[1]
    xs = ["input"] + labels
    ys = [input_cols] + after
    pos = list(range(len(xs)))
    ax1.plot(pos, ys, marker="o", color="#dd8452", linewidth=2)
    ax1.fill_between(pos, ys, alpha=0.15, color="#dd8452")
    ax1.set_xticks(pos)
    ax1.set_xticklabels(xs, rotation=45, ha="right")
    ax1.set_ylabel("total columns")
    ax1.set_title("DataFrame width through the pipeline", fontweight="bold")
    for x, y in zip(pos, ys):
        ax1.text(x, y, str(y), ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"featpipe  (mode={mode})  --  {df_name}",
                 fontsize=14, fontweight="bold")
    return fig


def featpipe(
    df: pd.DataFrame,
    steps: Optional[Sequence[dict]] = None,
    params: Optional[dict] = None,
    *,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 4.6,
    dpi: int = 110,
):
    """Chain the seven dextra feature-engineering functions into one pipeline.

    featpipe is the Stage 4.4 convenience wrapper. It runs ``transform``,
    ``scale``, ``bin``, ``encode``, ``dtfeats``, ``cross`` and ``aggfeat`` in
    sequence, threading the transformed DataFrame from one step to the next,
    and collects every step's ``params`` dict into a single combined,
    versioned, JSON-serialisable artifact -- a lightweight feature store.

    Two modes mirror the per-function contract in FEATURES_PHILOSOPHY.md.
    In FIT mode (``steps`` supplied) each step is fitted on ``df`` and its
    params recorded. In APPLY mode (``params`` or ``load_path`` supplied) the
    saved per-step params are replayed verbatim, in order, with no re-fitting
    -- the safeguard against leakage across a train/test boundary.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    steps : sequence of dict, optional
        Fit-mode recipe. Each dict has a ``'fn'`` key naming one of
        ``transform / scale / bin / encode / dtfeats / cross / aggfeat``; every
        other key is forwarded as a keyword argument to that function, e.g.
        ``{'fn': 'scale', 'cols': ['price'], 'method': 'robust'}``. A step may
        reference a column produced by an earlier step. ``method='compare'``
        (or ``agg='compare'`` for aggfeat) is rejected -- featpipe commits a
        chosen recipe; explore options with the single function first.
    params : dict, optional
        Apply-mode artifact: a combined dict returned by an earlier fit.
        Triggers apply mode; ``steps`` must not also be given.
    save_path : str, optional
        Fit mode only. After fitting, the combined params dict is written to
        this path as indented JSON.
    load_path : str, optional
        Apply-mode shortcut. The combined params dict is read from this JSON
        file, then applied. Mutually exclusive with ``params`` and ``steps``.
    return_params : bool, default False
        If True the combined params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The fully transformed DataFrame, and -- when requested -- the combined
        params dict and/or the matplotlib figure.

    Notes
    -----
    The combined params dict has the shape::

        {"function": "featpipe", "version": ..., "fit_at": ...,
         "steps": [<params of step 0>, <params of step 1>, ...],
         "metadata": {"n_steps": ..., "step_summary": [...],
                      "input_shape": [...], "output_shape": [...]}}

    Each element of ``steps`` is exactly the JSON-serialisable params dict the
    corresponding function already returns, so the whole artifact survives
    ``json.dump`` / ``json.load`` and reproduces the transform on another
    machine or day.

    Examples
    --------
    >>> recipe = [
    ...     {'fn': 'transform', 'cols': ['income'], 'method': 'log1p'},
    ...     {'fn': 'scale', 'cols': ['income_log1p', 'age'], 'method': 'robust'},
    ...     {'fn': 'encode', 'cols': ['city'], 'method': 'onehot'},
    ... ]
    >>> df_tr, p = dx.featpipe(df_train, steps=recipe, return_params=True,
    ...                        save_path='pipeline.json')
    >>> df_te = dx.featpipe(df_test, params=p)            # apply, no re-fit
    >>> df_te2 = dx.featpipe(df_test, load_path='pipeline.json')  # same result
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- resolve mode ---------------------------------------------------
    if load_path is not None:
        if params is not None or steps is not None:
            raise ValueError(
                "featpipe: 'load_path' is an apply-mode shortcut; do not also "
                "pass 'params' or 'steps'.")
        import json
        with open(load_path, "r", encoding="utf-8") as fh:
            params = json.load(fh)

    if params is not None:
        if steps is not None:
            raise ValueError(
                "featpipe: pass EITHER 'steps' (fit mode) OR 'params' "
                "(apply mode), not both.")
        if save_path is not None:
            raise ValueError(
                "featpipe: 'save_path' saves a freshly fitted pipeline and is "
                "valid in fit mode only (when 'steps' is given).")
        return _featpipe_apply(df, params, show, plot, return_df,
                               return_params, return_fig, decimals, df_name,
                               fig_width, fig_height, dpi)

    if steps is None:
        raise ValueError(
            "featpipe: provide 'steps' to fit a pipeline, or 'params' / "
            "'load_path' to apply a saved one.")

    return _featpipe_fit(df, steps, save_path, show, plot, return_df,
                         return_params, return_fig, decimals, df_name,
                         fig_width, fig_height, dpi)


def _featpipe_fit(df, steps, save_path, show, plot, return_df,
                  return_params, return_fig, decimals, df_name,
                  fig_width, fig_height, dpi):
    clean = _featpipe_validate_steps(steps)

    out = df.copy()
    out.attrs = dict(df.attrs)
    input_cols = out.shape[1]
    step_params_list, summary_rows, step_summary = [], [], []
    prev_cols = list(out.columns)

    for idx, (fn_name, kwargs) in enumerate(clean):
        fn = _FEATPIPE_DISPATCH[fn_name]
        before_n = out.shape[1]
        try:
            new_out, sp = fn(out, return_params=True, return_df=True,
                             show=False, plot=False, **kwargs)
        except Exception as exc:
            raise type(exc)(
                f"featpipe step {idx} (fn={fn_name}, fit): {exc}") from exc
        out = new_out
        after_n = out.shape[1]
        added = [c for c in out.columns if c not in prev_cols]
        prev_cols = list(out.columns)
        step_params_list.append(sp)
        summary_rows.append({
            "fn": fn_name, "method": str(sp.get("method", "-")),
            "cols_before": before_n, "cols_after": after_n,
            "cols_added": len(added)})
        step_summary.append({"step": idx, "fn": fn_name,
                             "method": sp.get("method")})

    combined = {
        "function": "featpipe",
        "version": __version__,
        "fit_at": now_iso(),
        "steps": step_params_list,
        "metadata": {
            "n_steps": len(clean),
            "step_summary": step_summary,
            "input_shape": list(df.shape),
            "output_shape": list(out.shape),
        },
    }

    saved_note = ""
    if save_path is not None:
        import json
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2)
        saved_note = f" Saved to '{save_path}'."

    n_new = sum(r["cols_added"] for r in summary_rows)
    chain = " -> ".join(r["fn"] for r in summary_rows)
    decision = (f"Fitted a {len(clean)}-step featpipe pipeline ({chain}); "
                f"{n_new} new column(s) produced; combined params is a "
                f"versioned, JSON-serialisable artifact.{saved_note} Apply to "
                f"held-out data with featpipe(df_test, params=...).")

    append_audit(out, {
        "stage": "feature_pipeline",
        "function": "featpipe",
        "timestamp": combined["fit_at"],
        "mode": "fit",
        "params": {"n_steps": len(clean), "chain": chain,
                   "save_path": save_path},
        "decision": decision,
    })

    summary = pd.DataFrame(summary_rows)
    summary.index.name = "step"
    if show:
        _print_header(f"Feature pipeline for: {df_name}  "
                      f"({len(clean)} step(s), mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_featpipe(summary_rows, input_cols, df_name, "fit",
                             fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, combined, fig, return_df, return_params, return_fig)


def _featpipe_apply(df, params, show, plot, return_df, return_params,
                    return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "featpipe":
        got = (params.get("function") if isinstance(params, dict)
               else type(params).__name__)
        raise ValueError(
            f"featpipe apply: params is not a featpipe pipeline "
            f"(function={got!r}).")
    step_params = params.get("steps")
    if not isinstance(step_params, list) or len(step_params) == 0:
        raise ValueError(
            "featpipe apply: params['steps'] must be a non-empty list of "
            "per-function params dicts.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    input_cols = out.shape[1]
    summary_rows = []
    prev_cols = list(out.columns)

    for idx, sp in enumerate(step_params):
        if not isinstance(sp, dict) or "function" not in sp:
            raise ValueError(
                f"featpipe apply: step {idx} is not a valid params dict.")
        fn_name = sp["function"]
        if fn_name not in _FEATPIPE_DISPATCH:
            raise ValueError(
                f"featpipe apply: step {idx} references unknown function "
                f"{fn_name!r}.")
        fn = _FEATPIPE_DISPATCH[fn_name]
        before_n = out.shape[1]
        call_kwargs = {"params": sp, "show": False, "plot": False,
                       "return_df": True, "return_params": False}
        if fn_name in _FEATPIPE_INPLACE_FNS:
            ip = sp.get("metadata", {}).get("inplace")
            if ip is not None:
                call_kwargs["inplace"] = bool(ip)
        try:
            out = fn(out, **call_kwargs)
        except Exception as exc:
            raise type(exc)(
                f"featpipe step {idx} (fn={fn_name}, apply): {exc}") from exc
        after_n = out.shape[1]
        added = [c for c in out.columns if c not in prev_cols]
        prev_cols = list(out.columns)
        summary_rows.append({
            "fn": fn_name, "method": str(sp.get("method", "-")),
            "cols_before": before_n, "cols_after": after_n,
            "cols_added": len(added)})

    n_new = sum(r["cols_added"] for r in summary_rows)
    fit_at = params.get("fit_at", "?")
    chain = " -> ".join(r["fn"] for r in summary_rows)
    decision = (f"Applied a saved {len(summary_rows)}-step featpipe pipeline "
                f"({chain}; fitted {fit_at}); {n_new} new column(s) produced; "
                f"no re-fit -- leakage-safe.")

    append_audit(out, {
        "stage": "feature_pipeline",
        "function": "featpipe",
        "timestamp": now_iso(),
        "mode": "apply",
        "params": {"n_steps": len(summary_rows), "chain": chain,
                   "fit_at": fit_at},
        "decision": decision,
    })

    summary = pd.DataFrame(summary_rows)
    summary.index.name = "step"
    if show:
        _print_header(f"Feature pipeline for: {df_name}  "
                      f"({len(summary_rows)} step(s), mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_featpipe(summary_rows, input_cols, df_name, "apply",
                             fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)
