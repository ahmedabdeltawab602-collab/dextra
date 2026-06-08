"""Evaluation helpers for dextra - Phase 7 of the Roadmap.

Implements the deep, multi-metric evaluation framework documented in
EVALUATION_PHILOSOPHY.md at the project root. Evaluation judges an
ALREADY-TRAINED model; it never trains a new one (the lone, deliberate
exception is :func:`learning_curves`, which re-fits on progressively larger
*subsets* because producing that curve is its job).

Every function in this module:

* Accepts a pandas DataFrame and NEVER mutates it. The diagnostics are returned
  as a fresh metrics DataFrame; the audit trail is written to a copy's attrs.
* Supports two input modes:
    - LABEL mode    : ``df`` + ``y_true`` + ``y_pred`` (+ ``scores`` for
      ``roc_pr``). A pure, estimator-free evaluation of predictions from
      anywhere.
    - ARTIFACT mode : ``df`` + ``params`` (a Phase-6 modeling artifact). The
      target, features, prediction column and fitted estimator are read from
      the artifact and the truth is derived from ``df``.
* Prints a dense metrics table and a one-line ``Decision:`` sentence naming the
  headline metric and the split / mode it was measured on.
* Renders a multi-panel diagnostic figure.
* Appends an entry to a copy's ``df.attrs['dextra_audit']``.
* Exposes a JSON-safe evaluation report via ``return_params=True`` (a
  descriptor of the computed metrics + metadata -- evaluation builds no
  estimator, so there is none to expose).

scikit-learn is imported lazily, only when a metric actually needs it, so the
rest of dextra keeps working without it (install the ``ml`` extra to enable).
``scipy.stats`` (a core dependency) is imported lazily inside
:func:`residual_analysis` for the Q-Q plot and the Jarque-Bera test.

Stage 7.1 - confusion_report ; 7.2 - residual_analysis ;
Stage 7.3 - roc_pr          ; 7.4 - learning_curves.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import _ensure_pandas, append_audit, get_variable_name, now_iso
from ._version import __version__
from .modeling import (
    _display,
    _finalize_figure,
    _fmt_metric,
    _fmt_table,
    _json_safe_num,
    _print_header,
    _require_sklearn,
    _ret_pack,
)

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Shared helpers (input resolution for the two modes)
# ---------------------------------------------------------------------------

def _resolve_eval_series(df, ref, argname, func_name):
    """Resolve a column name or array-like to (name, Series aligned to df)."""
    if ref is None:
        raise ValueError(
            f"{func_name}: {argname}= is required in label mode "
            f"(or pass params= for artifact mode).")
    if isinstance(ref, str):
        if ref not in df.columns:
            raise KeyError(f"{func_name}: {argname}={ref!r} not found in df.")
        return ref, df[ref]
    s = ref if isinstance(ref, pd.Series) else pd.Series(list(ref))
    if len(s) != len(df):
        raise ValueError(
            f"{func_name}: {argname} has length {len(s)}, df has {len(df)} rows.")
    s = s.copy()
    s.index = df.index
    name = str(s.name) if s.name is not None else argname
    return name, s


def _artifact_features_X(df, params, func_name):
    """Return the numeric feature matrix for an artifact, or None if unusable."""
    features = list(params.get("features") or [])
    estimator = params.get("estimator")
    if estimator is None or not features:
        return None, features, estimator
    missing = [c for c in features if c not in df.columns]
    if missing:
        return None, features, estimator
    X = df[features].apply(pd.to_numeric, errors="coerce")
    return X, features, estimator


def _artifact_truth(df, params, func_name):
    """Derive (y_true, y_pred, target) from a Phase-6 supervised artifact."""
    if not isinstance(params, dict):
        raise ValueError(
            f"{func_name}: params must be a dextra modeling artifact dict, "
            f"got {type(params).__name__}.")
    target = params.get("target")
    if target is None:
        raise ValueError(
            f"{func_name}: this artifact has no target (clustering is "
            f"unsupervised). Pass y_true / y_pred explicitly.")
    if target not in df.columns:
        raise KeyError(
            f"{func_name}: artifact target {target!r} is not a column of df; "
            f"pass y_true= explicitly.")
    y_true = df[target]
    X, features, estimator = _artifact_features_X(df, params, func_name)
    pred_col = params.get("pred_col")
    if pred_col and pred_col in df.columns:
        y_pred = df[pred_col]
    elif X is not None and not X.isna().any().any():
        y_pred = pd.Series(estimator.predict(X.to_numpy(dtype=float)),
                           index=df.index, name=f"{target}_pred")
    else:
        raise ValueError(
            f"{func_name}: cannot derive predictions -- supply y_pred=, or a df "
            f"carrying '{pred_col}' or the artifact's feature columns "
            f"{features}.")
    return y_true, y_pred, target


def _estimator_scores(estimator, Xa):
    """Return (scores, classes) from predict_proba or decision_function."""
    classes = [c for c in list(getattr(estimator, "classes_", []))]
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(Xa), dtype=float), classes
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(Xa), dtype=float), classes
    raise ValueError(
        "the fitted estimator exposes neither predict_proba nor "
        "decision_function; cannot build ROC / PR curves.")


def _drop_pair_na(a, b):
    """Align and drop rows where either array is NaN/non-finite (numeric)."""
    av = pd.to_numeric(pd.Series(np.asarray(a).ravel()), errors="coerce")
    bv = pd.to_numeric(pd.Series(np.asarray(b).ravel()), errors="coerce")
    mask = av.notna() & bv.notna()
    return av[mask].to_numpy(dtype=float), bv[mask].to_numpy(dtype=float)


# ===========================================================================
# 7.1  confusion_report  --  classification confusion diagnostics
# ===========================================================================

def confusion_report(
    df: pd.DataFrame,
    y_true=None,
    y_pred=None,
    *,
    params: Optional[dict] = None,
    labels: Optional[Sequence] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 15.0,
    fig_height: float = 4.6,
    dpi: int = 110,
):
    """Per-class confusion diagnostics for a classifier in one line.

    Two input modes. In LABEL mode pass ``y_true`` and ``y_pred`` (column names
    or array-likes). In ARTIFACT mode pass ``params`` (a Phase-6 ``classify``
    artifact) and the data to judge on; the truth is derived from it. Returns a
    dense per-class precision / recall / F1 / support table (plus accuracy and
    macro / weighted averages), a three-panel figure, and a one-line decision.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    y_true, y_pred : str or array-like
        True and predicted labels (label mode). A column name or a
        Series/array aligned to ``df``.
    params : dict, optional
        A Phase-6 ``classify`` artifact (artifact mode). Supersedes
        ``y_true`` / ``y_pred``.
    labels : sequence, optional
        Explicit class order. Defaults to the sorted union of the labels seen.
    return_params, show, plot, return_df, return_fig, decimals, df_name
        Standard dextra flags. ``return_params=True`` returns a JSON-safe
        evaluation report descriptor.

    Returns
    -------
    pandas.DataFrame
        The per-class metrics table, and -- when requested -- the report
        descriptor and/or the matplotlib figure.

    Examples
    --------
    >>> dx.confusion_report(df, y_true='churn', y_pred='churn_pred')
    >>> _, p = dx.classify(tr, y='churn', method='forest', return_params=True)
    >>> dx.confusion_report(te, params=p)              # artifact mode
    """
    _require_sklearn("confusion_report")
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    mode = "artifact" if params is not None else "label"
    if mode == "artifact":
        yt, yp, target = _artifact_truth(df, params, "confusion_report")
    else:
        _, yt = _resolve_eval_series(df, y_true, "y_true", "confusion_report")
        _, yp = _resolve_eval_series(df, y_pred, "y_pred", "confusion_report")
        target = yt.name if yt.name is not None else "target"

    pair = pd.DataFrame({"t": yt.to_numpy(), "p": yp.to_numpy()}).dropna()
    if len(pair) == 0:
        raise ValueError("confusion_report: no rows with both labels present.")
    t = pair["t"].to_numpy()
    p = pair["p"].to_numpy()

    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )
    if labels is not None:
        classes = list(labels)
    else:
        classes = sorted(set(t.tolist()) | set(p.tolist()), key=lambda v: str(v))
    prec, rec, f1, sup = precision_recall_fscore_support(
        t, p, labels=classes, zero_division=0)
    acc = float(accuracy_score(t, p))
    mp, mr, mf, _ = precision_recall_fscore_support(
        t, p, labels=classes, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(
        t, p, labels=classes, average="weighted", zero_division=0)
    cm = confusion_matrix(t, p, labels=classes)

    idx = [str(c) for c in classes] + ["accuracy", "macro avg", "weighted avg"]
    table = pd.DataFrame(
        {
            "precision": list(prec) + [np.nan, mp, wp],
            "recall": list(rec) + [np.nan, mr, wr],
            "f1": list(f1) + [acc, mf, wf],
            "support": list(sup) + [int(sup.sum()), int(sup.sum()),
                                    int(sup.sum())],
        },
        index=idx,
    )

    worst_i = int(np.argmin(rec)) if len(rec) else 0
    worst_cls = str(classes[worst_i]) if len(classes) else "?"
    report = {
        "function": "confusion_report",
        "task": "classification",
        "target": str(target),
        "version": __version__,
        "evaluated_at": now_iso(),
        "metrics": {
            "per_class": {
                str(c): {
                    "precision": _json_safe_num(prec[i]),
                    "recall": _json_safe_num(rec[i]),
                    "f1": _json_safe_num(f1[i]),
                    "support": int(sup[i]),
                }
                for i, c in enumerate(classes)
            },
            "overall": {
                "accuracy": _json_safe_num(acc),
                "macro_f1": _json_safe_num(mf),
                "weighted_f1": _json_safe_num(wf),
            },
            "confusion_matrix": cm.tolist(),
            "labels": [str(c) for c in classes],
        },
        "metadata": {"n": int(len(pair)), "n_classes": len(classes),
                     "input_mode": mode},
    }

    decision = (
        f"{len(pair)} rows, {len(classes)} classes: accuracy="
        f"{_fmt_metric(acc, decimals)}, weighted-F1={_fmt_metric(wf, decimals)}, "
        f"macro-F1={_fmt_metric(mf, decimals)}. Weakest class '{worst_cls}' "
        f"recall={_fmt_metric(rec[worst_i] if len(rec) else None, decimals)} "
        f"(mode={mode}).")

    out = df.copy()
    out.attrs = dict(df.attrs)
    append_audit(out, {
        "stage": "evaluation",
        "function": "confusion_report",
        "timestamp": report["evaluated_at"],
        "mode": mode,
        "params": {"target": str(target), "classes": [str(c) for c in classes]},
        "decision": decision,
    })

    if show:
        _print_header(f"Confusion report for: {df_name}  "
                      f"(target={target}, mode={mode})")
        _display(_fmt_table(table, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_confusion(cm, classes, list(f1), target,
                              fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(table, report, fig, return_df, return_params, return_fig)


def _plot_confusion(cm, classes, f1, target, fig_width, fig_height, dpi):
    labs = [str(c) for c in classes]
    cmf = cm.astype(float)
    row_sums = cmf.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cmf / row_sums
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Confusion diagnostics -- {target}", fontsize=13,
                 fontweight="bold")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labs, yticklabels=labs, ax=axes[0])
    axes[0].set_xlabel("predicted")
    axes[0].set_ylabel("actual")
    axes[0].set_title("Confusion matrix (counts)")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", cbar=False,
                vmin=0, vmax=1, xticklabels=labs, yticklabels=labs, ax=axes[1])
    axes[1].set_xlabel("predicted")
    axes[1].set_ylabel("actual")
    axes[1].set_title("Row-normalised (recall per class)")
    axes[2].barh(labs, f1, color="#4c72b0")
    axes[2].set_xlim(0, 1)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("F1")
    axes[2].set_title("Per-class F1")
    for i, v in enumerate(f1):
        axes[2].text(min(v + 0.02, 0.95), i, f"{v:.2f}", va="center",
                     fontsize=9)
    return fig


# ===========================================================================
# 7.2  residual_analysis  --  regression residual diagnostics
# ===========================================================================

def residual_analysis(
    df: pd.DataFrame,
    y_true=None,
    y_pred=None,
    *,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 14.0,
    fig_height: float = 9.0,
    dpi: int = 110,
):
    """Residual diagnostics for a regression model in one line.

    Two input modes (LABEL: ``y_true`` + ``y_pred``; ARTIFACT: a Phase-6
    ``regress`` artifact via ``params``). Returns a dense diagnostics table
    (residual mean / std / skew / kurtosis, R2 / RMSE / MAE, Durbin-Watson, a
    heteroscedasticity hint, and a Jarque-Bera normality p-value), a four-panel
    figure (residual-vs-fitted, residual distribution, Normal Q-Q,
    scale-location), and a one-line decision on the residual assumptions.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    y_true, y_pred : str or array-like
        True and predicted numeric values (label mode).
    params : dict, optional
        A Phase-6 ``regress`` artifact (artifact mode).
    return_params, show, plot, return_df, return_fig, decimals, df_name
        Standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The diagnostics table, and -- when requested -- the report descriptor
        and/or the matplotlib figure.

    Examples
    --------
    >>> dx.residual_analysis(df, y_true='price', y_pred='price_pred')
    >>> _, p = dx.regress(tr, y='price', method='forest', return_params=True)
    >>> dx.residual_analysis(te, params=p)             # artifact mode
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    mode = "artifact" if params is not None else "label"
    if mode == "artifact":
        yt, yp, target = _artifact_truth(df, params, "residual_analysis")
    else:
        _, yt = _resolve_eval_series(df, y_true, "y_true", "residual_analysis")
        _, yp = _resolve_eval_series(df, y_pred, "y_pred", "residual_analysis")
        target = yt.name if yt.name is not None else "target"

    actual, fitted = _drop_pair_na(yt.to_numpy(), yp.to_numpy())
    if len(actual) < 3:
        raise ValueError(
            "residual_analysis: need >= 3 complete (non-NaN) rows; "
            f"got {len(actual)}.")
    resid = actual - fitted

    from scipy import stats as _sps
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    mae = float(np.mean(np.abs(resid)))
    r_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    skew = float(_sps.skew(resid)) if r_std > 0 else 0.0
    kurt = float(_sps.kurtosis(resid)) if r_std > 0 else 0.0
    denom = float(np.sum(resid ** 2))
    dw = float(np.sum(np.diff(resid) ** 2) / denom) if denom > 0 else None
    if r_std > 0 and np.std(fitted) > 0:
        hetero = float(np.corrcoef(np.abs(resid), fitted)[0, 1])
    else:
        hetero = 0.0
    try:
        jb_p = float(_sps.jarque_bera(resid)[1]) if len(resid) >= 8 else None
    except Exception:
        jb_p = None

    table = pd.DataFrame(
        {"value": [
            int(len(actual)), float(resid.mean()), r_std, skew, kurt,
            r2, rmse, mae, dw, hetero, jb_p,
        ]},
        index=["n", "resid_mean", "resid_std", "resid_skew", "resid_kurtosis",
               "r2", "rmse", "mae", "durbin_watson", "hetero_corr",
               "jarque_bera_p"],
    )

    normal = (jb_p is not None and jb_p >= 0.05)
    homosced = abs(hetero) < 0.3
    norm_txt = ("residuals look normal" if normal
                else "residuals deviate from normal" if jb_p is not None
                else "normality untested (n<8)")
    var_txt = ("constant variance" if homosced
               else "possible heteroscedasticity")
    report = {
        "function": "residual_analysis",
        "task": "regression",
        "target": str(target),
        "version": __version__,
        "evaluated_at": now_iso(),
        "metrics": {
            "resid_mean": _json_safe_num(resid.mean()),
            "resid_std": _json_safe_num(r_std),
            "resid_skew": _json_safe_num(skew),
            "resid_kurtosis": _json_safe_num(kurt),
            "r2": _json_safe_num(r2),
            "rmse": _json_safe_num(rmse),
            "mae": _json_safe_num(mae),
            "durbin_watson": _json_safe_num(dw),
            "hetero_corr": _json_safe_num(hetero),
            "jarque_bera_p": _json_safe_num(jb_p),
        },
        "metadata": {"n": int(len(actual)), "input_mode": mode,
                     "normal": bool(normal), "homoscedastic": bool(homosced)},
    }

    decision = (
        f"{len(actual)} residuals: R^2={_fmt_metric(r2, decimals)}, "
        f"RMSE={_fmt_metric(rmse, decimals)}, MAE={_fmt_metric(mae, decimals)}; "
        f"Durbin-Watson={_fmt_metric(dw, decimals)}. {norm_txt} "
        f"(Jarque-Bera p={_fmt_metric(jb_p, decimals)}), {var_txt} "
        f"(|corr|={_fmt_metric(abs(hetero), decimals)}) (mode={mode}).")

    out = df.copy()
    out.attrs = dict(df.attrs)
    append_audit(out, {
        "stage": "evaluation",
        "function": "residual_analysis",
        "timestamp": report["evaluated_at"],
        "mode": mode,
        "params": {"target": str(target)},
        "decision": decision,
    })

    if show:
        _print_header(f"Residual analysis for: {df_name}  "
                      f"(target={target}, mode={mode})")
        _display(_fmt_table(table, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_residuals(actual, fitted, resid, target,
                              fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(table, report, fig, return_df, return_params, return_fig)


def _plot_residuals(actual, fitted, resid, target, fig_width, fig_height, dpi):
    from scipy import stats as _sps
    fig, axes = plt.subplots(2, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Residual diagnostics -- {target}", fontsize=13,
                 fontweight="bold")

    ax = axes[0, 0]
    ax.scatter(fitted, resid, s=18, alpha=0.6, color="#4c72b0",
               edgecolor="none")
    ax.axhline(0.0, color="#c44e52", lw=1.2, ls="--")
    ax.set_xlabel("fitted (predicted)")
    ax.set_ylabel("residual")
    ax.set_title("Residuals vs fitted")

    ax = axes[0, 1]
    sns.histplot(resid, kde=True, ax=ax, color="#55a868", bins="auto")
    ax.axvline(0.0, color="#c44e52", lw=1.2, ls="--")
    ax.set_xlabel("residual")
    ax.set_title("Residual distribution")

    ax = axes[1, 0]
    r_std = np.std(resid, ddof=1) if len(resid) > 1 else 0.0
    if r_std > 0:
        _sps.probplot(resid, dist="norm", plot=ax)
        ax.get_lines()[0].set_markerfacecolor("#4c72b0")
        ax.get_lines()[0].set_markeredgecolor("none")
        ax.get_lines()[0].set_markersize(5)
        ax.get_lines()[1].set_color("#c44e52")
    else:
        ax.text(0.5, 0.5, "degenerate (zero-variance) residuals",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Normal Q-Q")

    ax = axes[1, 1]
    std_resid = resid / r_std if r_std > 0 else np.zeros_like(resid)
    ax.scatter(fitted, np.sqrt(np.abs(std_resid)), s=18, alpha=0.6,
               color="#8172b3", edgecolor="none")
    ax.set_xlabel("fitted (predicted)")
    ax.set_ylabel("sqrt(|standardised residual|)")
    ax.set_title("Scale-location")
    return fig


# ===========================================================================
# 7.3  roc_pr  --  ROC and Precision-Recall curves
# ===========================================================================

def roc_pr(
    df: pd.DataFrame,
    y_true=None,
    scores=None,
    *,
    params: Optional[dict] = None,
    pos_label=None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 13.0,
    fig_height: float = 5.6,
    dpi: int = 110,
):
    """ROC and Precision-Recall curves for a classifier in one line.

    Two input modes. In LABEL mode pass ``y_true`` and ``scores`` (predicted
    probabilities / decision scores: a 1-D positive-class vector for binary, or
    an ``(n, n_classes)`` matrix for multiclass one-vs-rest). In ARTIFACT mode
    pass ``params`` (a Phase-6 ``classify`` artifact carrying a fitted
    estimator) and the data; scores come from ``predict_proba`` /
    ``decision_function``. Returns a per-class ROC-AUC / average-precision
    table (plus macro), a two-panel ROC + PR figure, and a one-line decision.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    y_true : str or array-like
        True labels (label mode).
    scores : array-like
        Predicted scores (label mode): shape ``(n,)`` (binary positive class)
        or ``(n, n_classes)`` (multiclass).
    params : dict, optional
        A Phase-6 ``classify`` artifact (artifact mode).
    pos_label : optional
        The positive class for a binary 1-D ``scores`` vector. Defaults to the
        larger of the two sorted labels.
    return_params, show, plot, return_df, return_fig, decimals, df_name
        Standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The per-class AUC / AP table, and -- when requested -- the report
        descriptor and/or the matplotlib figure.

    Examples
    --------
    >>> dx.roc_pr(df, y_true='churn', scores='churn_proba')
    >>> _, p = dx.classify(tr, y='churn', method='forest', return_params=True)
    >>> dx.roc_pr(te, params=p)                        # artifact mode
    """
    _require_sklearn("roc_pr")
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    mode = "artifact" if params is not None else "label"
    if mode == "artifact":
        if not isinstance(params, dict):
            raise ValueError("roc_pr: params must be a dextra modeling artifact.")
        target = params.get("target")
        if target is None or target not in df.columns:
            raise KeyError(
                "roc_pr: artifact target missing from df; pass y_true / scores.")
        yt = df[target]
        X, _features, estimator = _artifact_features_X(df, params, "roc_pr")
        if X is None or X.isna().any().any() or estimator is None:
            raise ValueError(
                "roc_pr: artifact mode needs a fitted estimator and its "
                "feature columns (non-NaN) in df. Pass scores= instead.")
        score_arr, est_classes = _estimator_scores(
            estimator, X.to_numpy(dtype=float))
        classes = est_classes if est_classes else sorted(
            set(yt.dropna().tolist()), key=lambda v: str(v))
    else:
        _, yt = _resolve_eval_series(df, y_true, "y_true", "roc_pr")
        target = yt.name if yt.name is not None else "target"
        if scores is None:
            raise ValueError("roc_pr label mode requires scores=.")
        if isinstance(scores, str):
            score_arr = df[[scores]].apply(
                pd.to_numeric, errors="coerce").to_numpy(dtype=float).ravel()
        else:
            score_arr = np.asarray(scores, dtype=float)
        classes = sorted(set(yt.dropna().tolist()), key=lambda v: str(v))

    yt_arr = yt.to_numpy()
    score_arr = np.asarray(score_arr, dtype=float)
    binary = score_arr.ndim == 1 or (score_arr.ndim == 2 and
                                     score_arr.shape[1] <= 2)

    from sklearn.metrics import (
        auc as _auc,
    )
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_curve,
    )

    curves = {}
    rows_auc = []
    rows_ap = []
    index = []
    if binary:
        if len(classes) < 2:
            raise ValueError("roc_pr: need 2 classes for a binary ROC/PR curve.")
        pos = pos_label if pos_label is not None else classes[-1]
        if score_arr.ndim == 2:
            col = classes.index(pos) if pos in classes else score_arr.shape[1] - 1
            s = score_arr[:, col]
        else:
            s = score_arr
        ybin = (yt_arr == pos).astype(int)
        mask = ~pd.isna(pd.Series(s)).to_numpy()
        ybin, s = ybin[mask], s[mask]
        fpr, tpr, _ = roc_curve(ybin, s)
        prec, rec, _ = precision_recall_curve(ybin, s)
        au = float(_auc(fpr, tpr))
        ap = float(average_precision_score(ybin, s))
        curves[str(pos)] = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec}
        index = [f"positive ({pos})"]
        rows_auc = [au]
        rows_ap = [ap]
        macro_auc, macro_ap = au, ap
    else:
        from sklearn.preprocessing import label_binarize
        Yb = label_binarize(yt_arr, classes=classes)
        for i, c in enumerate(classes):
            s = score_arr[:, i]
            fpr, tpr, _ = roc_curve(Yb[:, i], s)
            prec, rec, _ = precision_recall_curve(Yb[:, i], s)
            curves[str(c)] = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec}
            rows_auc.append(float(_auc(fpr, tpr)))
            rows_ap.append(float(average_precision_score(Yb[:, i], s)))
            index.append(str(c))
        macro_auc = float(np.nanmean(rows_auc))
        macro_ap = float(np.nanmean(rows_ap))
        index.append("macro")
        rows_auc.append(macro_auc)
        rows_ap.append(macro_ap)

    table = pd.DataFrame({"roc_auc": rows_auc, "avg_precision": rows_ap},
                         index=index)

    report = {
        "function": "roc_pr",
        "task": "classification",
        "target": str(target),
        "version": __version__,
        "evaluated_at": now_iso(),
        "metrics": {
            "per_class": {
                index[i]: {"roc_auc": _json_safe_num(rows_auc[i]),
                           "avg_precision": _json_safe_num(rows_ap[i])}
                for i in range(len(index))
            },
            "macro_roc_auc": _json_safe_num(macro_auc),
            "macro_avg_precision": _json_safe_num(macro_ap),
        },
        "metadata": {"n": int(len(yt_arr)), "binary": bool(binary),
                     "n_classes": len(classes), "input_mode": mode},
    }

    decision = (
        f"{'Binary' if binary else 'Multiclass'} ranking: macro ROC-AUC="
        f"{_fmt_metric(macro_auc, decimals)}, macro avg-precision="
        f"{_fmt_metric(macro_ap, decimals)} over {len(classes)} class(es) "
        f"(mode={mode}). ROC-AUC 0.5 = random; pick a threshold from the PR "
        f"curve to fit your precision/recall trade-off.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    append_audit(out, {
        "stage": "evaluation",
        "function": "roc_pr",
        "timestamp": report["evaluated_at"],
        "mode": mode,
        "params": {"target": str(target), "binary": bool(binary)},
        "decision": decision,
    })

    if show:
        _print_header(f"ROC / PR report for: {df_name}  "
                      f"(target={target}, mode={mode})")
        _display(_fmt_table(table, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_roc_pr(curves, target, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(table, report, fig, return_df, return_params, return_fig)


def _plot_roc_pr(curves, target, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Ranking diagnostics -- {target}", fontsize=13,
                 fontweight="bold")
    palette = sns.color_palette("tab10", n_colors=max(len(curves), 1))
    for (name, c), col in zip(curves.items(), palette):
        axes[0].plot(c["fpr"], c["tpr"], lw=1.6, color=col, label=name)
        axes[1].plot(c["rec"], c["prec"], lw=1.6, color=col, label=name)
    axes[0].plot([0, 1], [0, 1], ls="--", lw=1.0, color="grey")
    axes[0].set_xlabel("false-positive rate")
    axes[0].set_ylabel("true-positive rate")
    axes[0].set_title("ROC curve")
    axes[0].set_xlim(-0.02, 1.02)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-Recall curve")
    axes[1].set_xlim(-0.02, 1.02)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8, loc="lower left")
    return fig


# ===========================================================================
# 7.4  learning_curves  --  bias / variance via training-size sweep
# ===========================================================================

def learning_curves(
    df: pd.DataFrame,
    y=None,
    cols: Optional[Sequence[str]] = None,
    *,
    params: Optional[dict] = None,
    estimator=None,
    task: Optional[str] = None,
    scoring: Optional[str] = None,
    cv: int = 5,
    train_sizes=None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 13.0,
    fig_height: float = 5.4,
    dpi: int = 110,
):
    """Learning curves (train vs cross-validated score) in one line.

    Re-fits an estimator on progressively larger subsets and plots the training
    score against the cross-validated score -- the canonical bias / variance
    diagnostic. In ARTIFACT mode pass ``params`` (any supervised Phase-6
    artifact); the estimator, features and target are read from it. Otherwise
    pass ``estimator=`` together with ``y`` and ``cols``. Returns a per-train-
    size score table, a two-panel figure (curve + train-CV gap), and a
    one-line decision naming the likely regime.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    y : str or array-like
        Target (when not using an artifact). Column name or aligned array.
    cols : sequence of str, optional
        Feature columns (when not using an artifact). Defaults to every numeric
        non-boolean column except the target.
    params : dict, optional
        A Phase-6 supervised artifact (``regress`` / ``classify``).
    estimator : sklearn estimator, optional
        An estimator to evaluate (when not using an artifact).
    task : {'regression', 'classification'}, optional
        Defaults to the artifact's task, else inferred from ``y``.
    scoring : str, optional
        sklearn scoring name. Defaults to ``'r2'`` (regression) /
        ``'accuracy'`` (classification).
    cv : int, default 5
        Cross-validation folds.
    train_sizes : array-like, optional
        Fractions of the training set. Defaults to ``linspace(0.1, 1.0, 5)``.
    return_params, show, plot, return_df, return_fig, decimals, df_name
        Standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The per-train-size score table, and -- when requested -- the report
        descriptor and/or the matplotlib figure.

    Examples
    --------
    >>> _, p = dx.classify(tr, y='churn', method='forest', return_params=True)
    >>> dx.learning_curves(tr, params=p)               # artifact mode
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> dx.learning_curves(tr, y='price', estimator=RandomForestRegressor())
    """
    _require_sklearn("learning_curves")
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    mode = "artifact" if params is not None else "estimator"
    if mode == "artifact":
        if not isinstance(params, dict):
            raise ValueError("learning_curves: params must be an artifact dict.")
        est = params.get("estimator")
        if est is None:
            raise ValueError(
                "learning_curves: artifact carries no fitted estimator.")
        target = params.get("target")
        if target is None:
            raise ValueError(
                "learning_curves: unsupervised artifact has no target.")
        if target not in df.columns:
            raise KeyError(
                f"learning_curves: target {target!r} not in df.")
        features = list(params.get("features") or [])
        miss = [c for c in features if c not in df.columns]
        if miss:
            raise KeyError(f"learning_curves: missing feature columns {miss}.")
        y_series = df[target]
        task = task or params.get("task")
    else:
        if estimator is None:
            raise ValueError(
                "learning_curves: pass params= (artifact) or estimator= + y=.")
        est = estimator
        _, y_series = _resolve_eval_series(df, y, "y", "learning_curves")
        target = y_series.name if y_series.name is not None else "target"
        exclude = {target} if isinstance(y, str) else set()
        if cols is None:
            features = [c for c in df.columns
                        if c not in exclude
                        and pd.api.types.is_numeric_dtype(df[c])
                        and not pd.api.types.is_bool_dtype(df[c])]
        else:
            features = [c for c in cols if c not in exclude]
        if not features:
            raise ValueError("learning_curves: no numeric feature columns; "
                             "pass cols= explicitly.")
        if task is None:
            task = _infer_task(y_series)

    X = df[features].apply(pd.to_numeric, errors="coerce")
    classification = (task == "classification")
    if classification:
        ymask = X.notna().all(axis=1) & y_series.notna()
        Xa = X.loc[ymask].to_numpy(dtype=float)
        ya = y_series.loc[ymask].to_numpy()
    else:
        yv = pd.to_numeric(y_series, errors="coerce")
        ymask = X.notna().all(axis=1) & yv.notna()
        Xa = X.loc[ymask].to_numpy(dtype=float)
        ya = yv.loc[ymask].to_numpy(dtype=float)
    if len(Xa) < (cv * 2):
        raise ValueError(
            f"learning_curves: need >= {cv * 2} complete rows for {cv}-fold CV; "
            f"got {len(Xa)}.")

    if scoring is None:
        scoring = "accuracy" if classification else "r2"
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    import warnings as _warnings

    from sklearn.base import clone
    from sklearn.model_selection import learning_curve
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        sizes, train_scores, val_scores = learning_curve(
            clone(est), Xa, ya, train_sizes=train_sizes, cv=cv,
            scoring=scoring, shuffle=True, random_state=0)

    tr_mean = train_scores.mean(axis=1)
    tr_std = train_scores.std(axis=1)
    cv_mean = val_scores.mean(axis=1)
    cv_std = val_scores.std(axis=1)

    table = pd.DataFrame(
        {"train_mean": tr_mean, "train_std": tr_std,
         "cv_mean": cv_mean, "cv_std": cv_std},
        index=[int(s) for s in sizes],
    )
    table.index.name = "train_size"

    final_gap = float(tr_mean[-1] - cv_mean[-1])
    final_cv = float(cv_mean[-1])
    if final_cv < 0.6 and final_gap < 0.1:
        regime = ("high bias (underfitting): both scores are low and close -- "
                  "add features or a more flexible model")
    elif final_gap >= 0.15:
        regime = ("high variance (overfitting): the train-CV gap is wide -- "
                  "add data or regularise")
    else:
        regime = "balanced: train and CV scores converge at a healthy level"

    report = {
        "function": "learning_curves",
        "task": task,
        "target": str(target),
        "version": __version__,
        "evaluated_at": now_iso(),
        "metrics": {
            "scoring": scoring,
            "train_sizes": [int(s) for s in sizes],
            "train_mean": [_json_safe_num(v) for v in tr_mean],
            "cv_mean": [_json_safe_num(v) for v in cv_mean],
            "final_train": _json_safe_num(tr_mean[-1]),
            "final_cv": _json_safe_num(final_cv),
            "final_gap": _json_safe_num(final_gap),
        },
        "metadata": {"n": int(len(Xa)), "n_features": len(features),
                     "cv_folds": int(cv), "input_mode": mode},
    }

    decision = (
        f"learning curve ({scoring}, {cv}-fold): final train="
        f"{_fmt_metric(tr_mean[-1], decimals)}, CV="
        f"{_fmt_metric(final_cv, decimals)}, gap="
        f"{_fmt_metric(final_gap, decimals)}. Diagnosis: {regime} "
        f"(mode={mode}).")

    out = df.copy()
    out.attrs = dict(df.attrs)
    append_audit(out, {
        "stage": "evaluation",
        "function": "learning_curves",
        "timestamp": report["evaluated_at"],
        "mode": mode,
        "params": {"target": str(target), "scoring": scoring, "cv": int(cv)},
        "decision": decision,
    })

    if show:
        _print_header(f"Learning curves for: {df_name}  "
                      f"(target={target}, scoring={scoring}, mode={mode})")
        _display(_fmt_table(table, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_learning(sizes, tr_mean, tr_std, cv_mean, cv_std,
                             scoring, target, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(table, report, fig, return_df, return_params, return_fig)


def _infer_task(y_series) -> str:
    """Best-effort regression/classification inference from a target Series."""
    if (pd.api.types.is_numeric_dtype(y_series)
            and not pd.api.types.is_bool_dtype(y_series)):
        nun = int(y_series.nunique(dropna=True))
        n = max(len(y_series), 1)
        if nun > 20 or (nun > 10 and nun / n > 0.2):
            return "regression"
    return "classification"


def _plot_learning(sizes, tr_mean, tr_std, cv_mean, cv_std, scoring, target,
                   fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Learning curves -- {target}", fontsize=13,
                 fontweight="bold")
    ax = axes[0]
    ax.plot(sizes, tr_mean, "o-", color="#4c72b0", label="train")
    ax.fill_between(sizes, tr_mean - tr_std, tr_mean + tr_std, alpha=0.15,
                    color="#4c72b0")
    ax.plot(sizes, cv_mean, "o-", color="#c44e52", label="cross-val")
    ax.fill_between(sizes, cv_mean - cv_std, cv_mean + cv_std, alpha=0.15,
                    color="#c44e52")
    ax.set_xlabel("training examples")
    ax.set_ylabel(scoring)
    ax.set_title("Train vs cross-validated score")
    ax.legend(fontsize=9, loc="best")

    ax = axes[1]
    gap = tr_mean - cv_mean
    ax.plot(sizes, gap, "o-", color="#8172b3")
    ax.axhline(0.0, color="grey", lw=1.0, ls="--")
    ax.set_xlabel("training examples")
    ax.set_ylabel("train - CV gap")
    ax.set_title("Generalisation gap")
    return fig


# ---------------------------------------------------------------------------
# Short aliases (consistent with Phases 2-6)
# ---------------------------------------------------------------------------

confrep = confusion_report
rocpr = roc_pr
residan = residual_analysis
learncv = learning_curves
