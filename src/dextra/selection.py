"""Feature-selection helpers for dextra - Phase 5 of the Roadmap.

Implements the fit / apply framework documented in SELECTION_PHILOSOPHY.md at
the project root. Feature selection keeps the subset of EXISTING columns that
helps a model and drops the rest -- it never creates a new column (that is
Phase 4). Every function in this module:

* Accepts a pandas DataFrame and returns a NEW DataFrame (immutable; the
  original is never mutated) with fewer columns.
* Supports two modes:
    - FIT mode  : scores the candidate features and learns which to keep.
    - APPLY mode: re-uses a saved ``params`` dict verbatim -- it only subsets
      the DataFrame to the learned column set, never re-scoring. This is the
      technical safeguard against selection leakage.
* Exposes a JSON-serialisable ``params`` dict via ``return_params=True``.
* Prints a kept / dropped summary table with the score behind each decision.
* Renders a multi-panel visual of the ranking.
* Prints a one-line ``Decision:`` sentence.
* Appends an entry to ``df.attrs['dextra_audit']``.
* Is idempotent under apply mode: subsetting to the kept columns twice gives
  the same DataFrame.

Stage 5.1 - the Filter family:
  - redundancy(df, method=...)  variance / correlation / vif / compare
  - relevance(df, y, method=...) anova / chi2 / mutualinfo / compare

Stage 5.2 - the Embedded + Wrapper families (lazily import scikit-learn):
  - importance(df, y, method=...) tree / l1 / linear / compare
  - rfe(df, y, estimator=...)      tree / linear / compare

Stage 5.3 - Pipeline wrapper:
  - selectpipe(df, steps=..., y=...)  chain the four selectors; fit -> combined
                                      params -> apply / save / load
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import _ensure_pandas, get_variable_name
from ._version import __version__

try:
    from IPython.display import display as _ipy_display
except ImportError:  # pragma: no cover
    _ipy_display = None


sns.set_style("whitegrid")

AUDIT_KEY = "dextra_audit"

_VALID_REDUNDANCY_METHODS = ("variance", "correlation", "vif", "compare")
_REDUNDANCY_CANDIDATES = ("variance", "correlation", "vif")
_VALID_RELEVANCE_METHODS = ("anova", "chi2", "mutualinfo", "compare")
_RELEVANCE_CANDIDATES = ("anova", "chi2", "mutualinfo")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _display(frame: pd.DataFrame) -> None:
    if _ipy_display is not None:
        _ipy_display(frame)
    else:
        print(frame.to_string())


def _print_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finalize_figure(fig, return_fig: bool) -> None:
    if fig is None:
        return
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if not return_fig:
        plt.show()


def _ret_pack(out, params, fig, return_df, return_params, return_fig):
    """Pack outputs in the fixed order: dataframe, params, figure."""
    results = []
    if return_df:
        results.append(out)
    if return_params:
        results.append(params)
    if return_fig:
        results.append(fig)
    if not results:
        return out
    if len(results) == 1:
        return results[0]
    return tuple(results)


def _append_audit(out: pd.DataFrame, entry: dict) -> None:
    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
    out.attrs[AUDIT_KEY].append(entry)


def _fmt_table(frame: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Format every cell of a summary table for readable display."""
    def _fmt(v):
        if pd.isna(v):
            return "-"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        try:
            return f"{float(v):,.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)
    return frame.map(_fmt)


def _json_safe_num(v):
    """Map a numeric value to a strict-JSON-safe value (no inf / nan)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    if np.isinf(f):
        return 1e12 if f > 0 else -1e12
    return f


def _auto_numeric_cols(df: pd.DataFrame) -> list:
    """Return numeric, non-boolean column names."""
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _resolve_features(df: pd.DataFrame, cols, func_name: str,
                      exclude: Optional[Sequence[str]] = None) -> list:
    """Validate an explicit cols selector or auto-pick numeric columns.

    ``exclude`` columns (the target, protected columns) are removed from the
    candidate pool even when the caller passed them or when auto-picking.
    """
    exclude = set(exclude or [])
    if cols is None:
        chosen = [c for c in _auto_numeric_cols(df) if c not in exclude]
        if not chosen:
            raise ValueError(
                f"{func_name}: no numeric candidate columns found. "
                f"Pass cols= explicitly.")
        return chosen
    chosen = list(dict.fromkeys(cols))
    bad = [c for c in chosen if c not in df.columns]
    if bad:
        raise KeyError(
            f"{func_name}: cols references columns not in df: {bad}")
    return [c for c in chosen if c not in exclude]


def _vif_for_column(X: np.ndarray, i: int) -> float:
    """Variance Inflation Factor for column i of X (OLS of col i on the rest)."""
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


def _redundancy_default_threshold(method: str) -> float:
    return {"variance": 0.0, "correlation": 0.95, "vif": 10.0}[method]


# ===========================================================================
# 1. redundancy  --  filter family, target-free (drop uninformative columns)
# ===========================================================================

def _redundancy_compute(df: pd.DataFrame, candidates: list, method: str,
                        threshold: float):
    """Score the candidates and split them into kept / dropped.

    Returns (kept, dropped, scores, extra) where ``scores`` maps every
    candidate to its criterion value and ``extra`` carries method-specific
    detail (drop reasons, the correlation matrix) for the visual.
    """
    non_num = [c for c in candidates
               if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        raise TypeError(
            f"redundancy method '{method}' needs numeric columns; "
            f"non-numeric candidate(s) passed: {non_num}")
    num = df[candidates].apply(pd.to_numeric, errors="coerce")
    scores: dict = {}

    if method == "variance":
        for c in candidates:
            scores[c] = float(num[c].var(ddof=0))
        dropped = [c for c in candidates if scores[c] <= threshold]
        kept = [c for c in candidates if c not in dropped]
        if not kept and candidates:
            best = max(candidates, key=lambda c: scores[c])
            kept = [best]
            dropped = [c for c in candidates if c != best]
            warnings.warn(
                "redundancy: every candidate is (near-)constant; kept the "
                "highest-variance column to avoid an empty feature set.",
                stacklevel=3)
        return kept, dropped, scores, {}

    if method == "correlation":
        if len(candidates) < 2:
            raise ValueError(
                "redundancy method 'correlation' needs >= 2 candidate columns.")
        clean = num.dropna()
        if len(clean) < 2:
            raise ValueError(
                "redundancy method 'correlation' needs >= 2 complete rows.")
        cmat = clean.corr().abs()
        kept, dropped, drop_reason = [], [], {}
        for c in candidates:
            partner = None
            for k in kept:
                r = cmat.loc[c, k]
                if pd.notna(r) and r >= threshold:
                    partner = [k, _json_safe_num(r)]
                    break
            if partner is None:
                kept.append(c)
            else:
                dropped.append(c)
                drop_reason[c] = partner
        for c in candidates:
            row = cmat.loc[c].drop(labels=[c], errors="ignore")
            scores[c] = float(row.max()) if row.notna().any() else 0.0
        return kept, dropped, scores, {"drop_reason": drop_reason,
                                       "cmat": cmat}

    # method == "vif"
    if len(candidates) < 2:
        raise ValueError(
            "redundancy method 'vif' needs >= 2 candidate columns.")
    clean = num.dropna()
    if len(clean) < len(candidates) + 1:
        raise ValueError(
            f"redundancy method 'vif' needs n >= n_candidates+1 complete "
            f"rows; got n={len(clean)}, candidates={len(candidates)}.")
    current = list(candidates)
    dropped, drop_vif = [], {}
    while len(current) > 1:
        X = clean[current].to_numpy(dtype=float)
        vifs = {current[i]: _vif_for_column(X, i) for i in range(len(current))}
        worst = max(current, key=lambda c: (vifs[c]
                    if np.isfinite(vifs[c]) else np.inf))
        wv = vifs[worst]
        if (not np.isfinite(wv)) or wv > threshold:
            dropped.append(worst)
            drop_vif[worst] = _json_safe_num(wv)
            current.remove(worst)
        else:
            break
    kept = current
    if len(kept) >= 2:
        Xk = clean[kept].to_numpy(dtype=float)
        for i, c in enumerate(kept):
            scores[c] = _json_safe_num(_vif_for_column(Xk, i))
    else:
        for c in kept:
            scores[c] = 1.0
    for c in dropped:
        scores[c] = drop_vif[c]
    return kept, dropped, scores, {"drop_vif": drop_vif}


def redundancy(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "variance",
    *,
    threshold: Optional[float] = None,
    protect: Optional[Sequence[str]] = None,
    params: Optional[dict] = None,
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
    """Drop features that carry little or duplicated information (no target).

    A Filter-family selector. It judges columns purely on their intrinsic
    statistics -- never looking at a target -- and removes the redundant ones.

    Two modes. In FIT mode (``params=None``) the candidates are scored on
    ``df``. In APPLY mode (``params`` supplied) the saved decision is replayed
    verbatim: the DataFrame is subset to the kept columns with no re-scoring
    -- the safeguard against selection leakage.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    cols : sequence of str, optional
        The candidate-feature pool. If ``None`` all numeric (non-boolean)
        columns are used. Columns outside this pool always pass through.
    method : {'variance', 'correlation', 'vif', 'compare'}
        'variance'    -> drop features whose variance <= ``threshold``.
        'correlation' -> within each pair of features with |corr| >=
                         ``threshold`` drop the later one (by column order).
        'vif'         -> iteratively drop the highest-VIF feature until every
                         remaining VIF <= ``threshold``.
        'compare'     -> rank by all three criteria; write / drop nothing.
    threshold : float, optional
        The cut value; its meaning depends on ``method`` (min variance / max
        |corr| / max VIF). Defaults: variance 0.0, correlation 0.95, vif 10.0.
    protect : sequence of str, optional
        Columns that must never be dropped even if flagged.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the learned params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with redundant candidate columns removed, and -- when
        requested -- the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.redundancy(df_train, method='correlation',
    ...                          threshold=0.9, return_params=True)
    >>> df_te = dx.redundancy(df_test, params=p)          # apply, no re-score
    >>> dx.redundancy(df, method='compare')               # explore
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _redundancy_apply(df, params, show, plot, return_df,
                                 return_params, return_fig, decimals, df_name,
                                 fig_width, fig_height, dpi)

    # ---- FIT MODE -------------------------------------------------------
    if method not in _VALID_REDUNDANCY_METHODS:
        raise ValueError(
            f"'method' must be one of {_VALID_REDUNDANCY_METHODS}, "
            f"got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "method='compare' selects nothing; call redundancy with a "
            "concrete method (variance/correlation/vif) to get params.")

    protect = list(protect or [])
    candidates = _resolve_features(df, cols, "redundancy", exclude=protect)
    if not candidates:
        raise ValueError("redundancy: no candidate columns left after "
                          "removing protected columns.")

    if method == "compare":
        return _redundancy_compare(df, candidates, show, plot, return_df,
                                   return_params, return_fig, decimals,
                                   df_name, fig_width, fig_height, dpi)

    if threshold is None:
        threshold = _redundancy_default_threshold(method)

    kept, dropped, scores, extra = _redundancy_compute(
        df, candidates, method, float(threshold))

    out = df.drop(columns=dropped)
    out.attrs = dict(df.attrs)

    params_out = {
        "function": "redundancy",
        "method": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "target": None,
        "candidates": list(candidates),
        "kept": list(kept),
        "dropped": list(dropped),
        "scores": {c: _json_safe_num(scores[c]) for c in candidates},
        "metadata": {
            "threshold": float(threshold),
            "protect": protect,
            "n_candidates": len(candidates),
            "n_dropped": len(dropped),
        },
    }

    crit = {"variance": "variance <= ", "correlation": "|corr| >= ",
            "vif": "VIF > "}[method]
    decision = (f"Filtered {len(candidates)} candidate feature(s) by "
                f"'{method}' redundancy ({crit}{threshold:g}); dropped "
                f"{len(dropped)}, kept {len(kept)}. Apply to held-out data "
                f"with redundancy(df_test, params=...).")

    _append_audit(out, {
        "stage": "feature_selection",
        "function": "redundancy",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "threshold": float(threshold),
                   "n_dropped": len(dropped)},
        "decision": decision,
    })

    summary = _redundancy_summary(candidates, kept, scores, method)
    if show:
        _print_header(f"Redundancy filter for: {df_name}  "
                      f"(method={method}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_redundancy(df, candidates, kept, dropped, scores,
                               method, extra, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _redundancy_summary(candidates, kept, scores, method) -> pd.DataFrame:
    label = {"variance": "variance", "correlation": "max_abs_corr",
             "vif": "vif"}[method]
    kept_set = set(kept)
    rows = [{label: scores.get(c), "decision": "keep" if c in kept_set
             else "drop"} for c in candidates]
    out = pd.DataFrame(rows, index=list(candidates))
    out.index.name = "feature"
    return out


def _redundancy_apply(df, params, show, plot, return_df, return_params,
                      return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "redundancy":
        got = (params.get("function") if isinstance(params, dict)
               else type(params).__name__)
        raise ValueError(
            f"params dict is not for 'redundancy' (function={got!r}).")
    method = params["method"]
    kept = params["kept"]
    dropped = params["dropped"]
    # Column-mismatch rejection: the kept columns must all be present.
    missing = [c for c in kept if c not in df.columns]
    if missing:
        raise KeyError(
            f"redundancy apply failed: params expects kept column(s) "
            f"{missing} which are not present in this DataFrame. The data "
            f"does not match the fitted selector.")

    out = df.drop(columns=[c for c in dropped if c in df.columns])
    out.attrs = dict(df.attrs)

    decision = (f"Applied saved '{method}' redundancy filter (fitted "
                f"{params.get('fit_at', '?')}); subset to {len(kept)} kept "
                f"feature(s), dropped {len(dropped)} -- no re-scoring, "
                f"leakage-safe.")
    _append_audit(out, {
        "stage": "feature_selection",
        "function": "redundancy",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"method": method, "n_dropped": len(dropped),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    scores = params.get("scores", {})
    candidates = params.get("candidates", kept + dropped)
    summary = _redundancy_summary(candidates, kept, scores, method)
    if show:
        _print_header(f"Redundancy filter for: {df_name}  "
                      f"(method={method}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_redundancy(df, [c for c in candidates if c in df.columns],
                               kept, dropped, scores, method, {},
                               fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _redundancy_compare(df, candidates, show, plot, return_df, return_params,
                        return_fig, decimals, df_name,
                        fig_width, fig_height, dpi):
    rows = []
    for m in _REDUNDANCY_CANDIDATES:
        thr = _redundancy_default_threshold(m)
        try:
            kept, dropped, _, _ = _redundancy_compute(df, candidates, m, thr)
            rows.append({"method": m, "threshold": thr,
                         "n_kept": len(kept), "n_dropped": len(dropped),
                         "dropped": ", ".join(dropped) if dropped else "-"})
        except (ValueError, TypeError) as exc:
            rows.append({"method": m, "threshold": thr, "n_kept": None,
                         "n_dropped": None, "dropped": f"n/a ({exc})"})
    summary = pd.DataFrame(rows).set_index("method")

    decision = (f"Compared {len(_REDUNDANCY_CANDIDATES)} redundancy criteria "
                f"on {len(candidates)} candidate(s). Table shows what each "
                f"would drop at its default threshold. No columns dropped -- "
                f"pick a method then call redundancy(method=...).")
    if show:
        _print_header(f"Redundancy COMPARE for: {df_name}  (nothing dropped)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_redundancy_compare(df, candidates,
                                       fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# 2. relevance  --  filter family, target-based (rank features vs the target)
# ===========================================================================

def _resolve_target(df, y, func_name):
    """Resolve y to (name, Series); raise on a missing column / length error."""
    if y is None:
        raise ValueError(f"{func_name} fit mode requires y= (the target).")
    if isinstance(y, str):
        if y not in df.columns:
            raise KeyError(f"{func_name}: target y={y!r} not found in df.")
        return y, df[y]
    s = y if isinstance(y, pd.Series) else pd.Series(y)
    if len(s) != len(df):
        raise ValueError(
            f"{func_name}: y has length {len(s)}, df has {len(df)} rows.")
    s = s.copy()
    s.index = df.index
    return (s.name if s.name is not None else None), s


def _infer_target_kind(s: pd.Series) -> str:
    """Classification vs regression: numeric high-cardinality -> regression."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return "regression" if int(s.nunique(dropna=True)) > 20 \
            else "classification"
    return "classification"


def _anova_f(groups) -> float:
    """One-way ANOVA F-statistic computed with NumPy only (no SciPy)."""
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 0.0
    n = sum(len(g) for g in groups)
    k = len(groups)
    if n <= k:
        return 0.0
    allv = np.concatenate(groups)
    grand = allv.mean()
    ssb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ssw = sum(float(((g - g.mean()) ** 2).sum()) for g in groups)
    if ssw <= 0:
        return float("inf") if ssb > 0 else 0.0
    f_val = (ssb / (k - 1)) / (ssw / (n - k))
    return float(f_val) if np.isfinite(f_val) else 0.0


def _chi2_scores(X: np.ndarray, y) -> np.ndarray:
    """Per-feature chi-squared statistic vs the class labels (sklearn formula)."""
    yarr = np.asarray(y)
    classes = pd.unique(yarr)
    n = len(yarr)
    Y = np.zeros((n, len(classes)))
    for ci, c in enumerate(classes):
        Y[:, ci] = (yarr == c).astype(float)
    observed = Y.T @ X
    feature_sum = X.sum(axis=0)
    class_prob = Y.mean(axis=0)
    expected = np.outer(class_prob, feature_sum)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = (observed - expected) ** 2 / expected
    terms[~np.isfinite(terms)] = 0.0
    return terms.sum(axis=0)


def _mutualinfo_scores(X: np.ndarray, y, kind: str) -> np.ndarray:
    """Mutual information per feature; lazily imports scikit-learn."""
    try:
        if kind == "classification":
            from sklearn.feature_selection import mutual_info_classif as _mi
        else:
            from sklearn.feature_selection import mutual_info_regression as _mi
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "relevance method 'mutualinfo' requires scikit-learn, which is "
            "not installed. Install it with `pip install scikit-learn`, or "
            "use method='anova' / 'chi2' which need only NumPy.") from exc
    yarr = np.asarray(y)
    if kind == "classification":
        codes = pd.Categorical(yarr).codes
        return np.asarray(_mi(X, codes, random_state=0), dtype=float)
    return np.asarray(_mi(X, yarr.astype(float), random_state=0), dtype=float)


def _anova_scores(X: np.ndarray, y, kind: str) -> np.ndarray:
    n, f = X.shape
    out = np.zeros(f)
    if kind == "classification":
        yarr = np.asarray(y)
        classes = pd.unique(yarr)
        for j in range(f):
            out[j] = _anova_f([X[yarr == c, j] for c in classes])
    else:
        yv = np.asarray(y, dtype=float)
        for j in range(f):
            col = X[:, j]
            if np.std(col) == 0 or np.std(yv) == 0:
                out[j] = 0.0
                continue
            r = float(np.corrcoef(col, yv)[0, 1])
            r2 = r * r
            out[j] = (float("inf") if r2 >= 1.0 - 1e-12
                      else r2 * (n - 2) / (1.0 - r2))
    return out


def _clean_feature_matrix(df, candidates, y_series, func_name, method):
    """Return (X ndarray, y aligned) over rows complete in features + target."""
    non_num = [c for c in candidates
               if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        raise TypeError(
            f"{func_name} method '{method}' needs numeric feature columns; "
            f"non-numeric candidate(s): {non_num}")
    feats = df[candidates].apply(pd.to_numeric, errors="coerce")
    mask = feats.notna().all(axis=1) & y_series.notna()
    if int(mask.sum()) < 3:
        raise ValueError(
            f"{func_name}: not enough complete rows (features + target) to "
            f"score the candidates.")
    return feats.loc[mask].to_numpy(dtype=float), y_series.loc[mask]


def _relevance_scores(df, candidates, y_series, target_kind, method) -> dict:
    X, yv = _clean_feature_matrix(df, candidates, y_series,
                                  "relevance", method)
    if method == "anova":
        arr = _anova_scores(X, yv, target_kind)
    elif method == "chi2":
        if target_kind != "classification":
            raise ValueError(
                "relevance method 'chi2' needs a classification target.")
        if (X < 0).any():
            raise ValueError(
                "relevance method 'chi2' needs non-negative features.")
        arr = _chi2_scores(X, yv)
    elif method == "mutualinfo":
        arr = _mutualinfo_scores(X, yv, target_kind)
    else:  # pragma: no cover
        raise ValueError(f"unknown relevance method {method!r}")
    return {candidates[i]: _json_safe_num(arr[i])
            for i in range(len(candidates))}


def _relevance_cut(candidates, scores, keep, threshold):
    """Rank candidates by score (desc) and split into kept / dropped."""
    def _key(c):
        s = scores.get(c)
        return s if s is not None else float("-inf")
    ranked = sorted(candidates, key=_key, reverse=True)
    if keep is not None:
        kept = ranked[:keep]
    elif threshold is not None:
        kept = [c for c in ranked if _key(c) >= threshold]
        if not kept and ranked:
            kept = ranked[:1]
            warnings.warn(
                "relevance: no feature met the score threshold; kept the "
                "single best to avoid an empty feature set.", stacklevel=3)
    else:
        kept = ranked
    kept_set = set(kept)
    dropped = [c for c in candidates if c not in kept_set]
    return kept, dropped, ranked


def relevance(
    df: pd.DataFrame,
    y=None,
    cols: Optional[Sequence[str]] = None,
    method: str = "anova",
    *,
    keep: Optional[int] = 10,
    threshold: Optional[float] = None,
    protect: Optional[Sequence[str]] = None,
    params: Optional[dict] = None,
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
    """Rank features by their univariate association with the target, keep top.

    A Filter-family selector. Each candidate feature is scored against the
    target ``y`` by a univariate criterion; the strongest are kept.

    Two modes. In FIT mode (``params=None``) the candidates are scored on
    ``df`` / ``y``. In APPLY mode (``params`` supplied) the saved decision is
    replayed verbatim -- the DataFrame is subset to the kept columns with no
    re-scoring -- the safeguard against target leakage. ``y`` is not needed
    in apply mode.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    y : str or array-like
        The target. A column name in ``df`` or an array-like aligned to its
        rows. Required in fit mode; ignored in apply mode. Auto-protected.
    cols : sequence of str, optional
        The candidate-feature pool (numeric). If ``None`` all numeric
        (non-boolean) columns except the target are used.
    method : {'anova', 'chi2', 'mutualinfo', 'compare'}
        'anova'      -> ANOVA F-test (classification) or regression F-test.
        'chi2'       -> chi-squared (non-negative features, classification).
        'mutualinfo' -> mutual information (captures non-linear dependence;
                        lazily imports scikit-learn).
        'compare'    -> rank by all three criteria; keep / drop nothing.
    keep : int, optional, default 10
        Keep the top-``keep`` features by score. Takes precedence over
        ``threshold``. Pass ``keep=None`` to select by ``threshold`` instead.
    threshold : float, optional
        Used only when ``keep is None``: keep features with score >= this.
    protect : sequence of str, optional
        Columns that must never be dropped.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the learned params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with low-relevance candidate columns removed, and -- when
        requested -- the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.relevance(df_train, y='churn', method='anova', keep=8,
    ...                         return_params=True)
    >>> df_te = dx.relevance(df_test, params=p)           # apply, no re-score
    >>> dx.relevance(df, y='churn', method='compare')     # explore
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _relevance_apply(df, params, show, plot, return_df,
                                return_params, return_fig, decimals, df_name,
                                fig_width, fig_height, dpi)

    # ---- FIT MODE -------------------------------------------------------
    if method not in _VALID_RELEVANCE_METHODS:
        raise ValueError(
            f"'method' must be one of {_VALID_RELEVANCE_METHODS}, "
            f"got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "method='compare' selects nothing; call relevance with a "
            "concrete method (anova/chi2/mutualinfo) to get params.")
    if keep is not None:
        if not isinstance(keep, (int, np.integer)) or keep < 1:
            raise ValueError(f"'keep' must be an int >= 1 or None, got {keep!r}")
        keep = int(keep)

    y_name, y_series = _resolve_target(df, y, "relevance")
    protect = list(protect or [])
    exclude = protect + ([y_name] if y_name is not None else [])
    candidates = _resolve_features(df, cols, "relevance", exclude=exclude)
    if not candidates:
        raise ValueError("relevance: no candidate columns left after "
                          "removing the target and protected columns.")
    target_kind = _infer_target_kind(y_series)

    if method == "compare":
        return _relevance_compare(df, candidates, y_name, y_series,
                                  target_kind, show, plot, return_df,
                                  return_params, return_fig, decimals,
                                  df_name, fig_width, fig_height, dpi)

    scores = _relevance_scores(df, candidates, y_series, target_kind, method)
    kept, dropped, ranked = _relevance_cut(candidates, scores, keep, threshold)

    out = df.drop(columns=dropped)
    out.attrs = dict(df.attrs)

    params_out = {
        "function": "relevance",
        "method": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "target": y_name,
        "target_kind": target_kind,
        "candidates": list(candidates),
        "kept": list(kept),
        "dropped": list(dropped),
        "scores": {c: scores[c] for c in candidates},
        "metadata": {
            "keep": keep,
            "threshold": (None if threshold is None else float(threshold)),
            "protect": protect,
            "n_candidates": len(candidates),
            "n_dropped": len(dropped),
        },
    }

    cut = (f"top {keep}" if keep is not None
           else f"score >= {threshold:g}")
    decision = (f"Ranked {len(candidates)} candidate feature(s) by '{method}' "
                f"relevance to '{y_name}' ({target_kind}); kept {cut} "
                f"({len(kept)}), dropped {len(dropped)}. Apply to held-out "
                f"data with relevance(df_test, params=...).")

    _append_audit(out, {
        "stage": "feature_selection",
        "function": "relevance",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "target": y_name, "keep": keep,
                   "threshold": params_out["metadata"]["threshold"],
                   "n_dropped": len(dropped)},
        "decision": decision,
    })

    summary = _relevance_summary(candidates, kept, scores, ranked, method)
    if show:
        _print_header(f"Relevance filter for: {df_name}  "
                      f"(method={method}, target={y_name}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_relevance(candidates, kept, scores, ranked, method,
                              y_name, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _relevance_summary(candidates, kept, scores, ranked, method) -> pd.DataFrame:
    kept_set = set(kept)
    rank_of = {c: i + 1 for i, c in enumerate(ranked)}
    rows = [{"score": scores.get(c), "rank": rank_of.get(c),
             "decision": "keep" if c in kept_set else "drop"}
            for c in candidates]
    out = pd.DataFrame(rows, index=list(candidates))
    out.index.name = "feature"
    return out.sort_values("rank")


def _relevance_apply(df, params, show, plot, return_df, return_params,
                     return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "relevance":
        got = (params.get("function") if isinstance(params, dict)
               else type(params).__name__)
        raise ValueError(
            f"params dict is not for 'relevance' (function={got!r}).")
    method = params["method"]
    kept = params["kept"]
    dropped = params["dropped"]
    missing = [c for c in kept if c not in df.columns]
    if missing:
        raise KeyError(
            f"relevance apply failed: params expects kept column(s) "
            f"{missing} which are not present in this DataFrame. The data "
            f"does not match the fitted selector.")

    out = df.drop(columns=[c for c in dropped if c in df.columns])
    out.attrs = dict(df.attrs)

    decision = (f"Applied saved '{method}' relevance filter (fitted "
                f"{params.get('fit_at', '?')}); subset to {len(kept)} kept "
                f"feature(s), dropped {len(dropped)} -- no re-scoring, "
                f"leakage-safe.")
    _append_audit(out, {
        "stage": "feature_selection",
        "function": "relevance",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"method": method, "target": params.get("target"),
                   "n_dropped": len(dropped), "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    scores = params.get("scores", {})
    candidates = params.get("candidates", kept + dropped)
    ranked = sorted(candidates,
                    key=lambda c: (scores.get(c) if scores.get(c) is not None
                                   else float("-inf")), reverse=True)
    summary = _relevance_summary(candidates, kept, scores, ranked, method)
    if show:
        _print_header(f"Relevance filter for: {df_name}  "
                      f"(method={method}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_relevance(candidates, kept, scores, ranked, method,
                              params.get("target"),
                              fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _relevance_compare(df, candidates, y_name, y_series, target_kind,
                       show, plot, return_df, return_params, return_fig,
                       decimals, df_name, fig_width, fig_height, dpi):
    table = {}
    for m in _RELEVANCE_CANDIDATES:
        try:
            table[m] = _relevance_scores(df, candidates, y_series,
                                         target_kind, m)
        except (ValueError, TypeError, ImportError) as exc:
            table[m] = {c: None for c in candidates}
            table[m]["__error__"] = str(exc)
    rows = []
    for c in candidates:
        rows.append({m: table[m].get(c) for m in _RELEVANCE_CANDIDATES})
    summary = pd.DataFrame(rows, index=list(candidates))
    summary.index.name = "feature"

    decision = (f"Compared {len(_RELEVANCE_CANDIDATES)} relevance criteria on "
                f"{len(candidates)} candidate(s) vs '{y_name}' ({target_kind})."
                f" Higher = more relevant. No columns dropped -- pick a method "
                f"then call relevance(method=...).")
    if show:
        _print_header(f"Relevance COMPARE for: {df_name}  (nothing dropped)")
        _display(_fmt_table(summary, decimals))
        for m in _RELEVANCE_CANDIDATES:
            err = table[m].get("__error__")
            if err:
                print(f"  note: method '{m}' unavailable -- {err}")
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_relevance_compare(summary, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# 3. Plotting helpers
# ===========================================================================

def _plot_redundancy(df, candidates, kept, dropped, scores, method,
                     extra, fig_width, fig_height, dpi):
    if not candidates:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    kept_set = set(kept)

    def _sv(c):
        v = scores.get(c)
        if v is None:
            return 0.0
        return float(min(v, 100.0)) if method == "vif" else float(v)

    order = sorted(candidates, key=_sv, reverse=True)
    vals = [_sv(c) for c in order]
    colors = ["#55a868" if c in kept_set else "#c44e52" for c in order]
    ax0 = axes[0]
    ax0.bar(range(len(order)), vals, color=colors,
            edgecolor="black", alpha=0.85)
    ax0.set_xticks(range(len(order)))
    ax0.set_xticklabels(order, rotation=45, ha="right")
    ax0.set_ylabel({"variance": "variance", "correlation": "max |corr|",
                    "vif": "VIF (capped at 100)"}[method])
    ax0.set_title("Redundancy score per feature  (green=keep, red=drop)",
                  fontweight="bold")

    ax1 = axes[1]
    if method in ("correlation", "vif") and len(candidates) >= 2:
        num = df[candidates].apply(pd.to_numeric, errors="coerce")
        cmat = num.corr().abs()
        sns.heatmap(cmat, ax=ax1, cmap="rocket_r", vmin=0, vmax=1,
                    cbar=True, xticklabels=True, yticklabels=True)
        ax1.set_title("Absolute correlation matrix", fontweight="bold")
    else:
        ax1.bar(["kept", "dropped"], [len(kept), len(dropped)],
                color=["#55a868", "#c44e52"], edgecolor="black", alpha=0.85)
        ax1.set_ylabel("feature count")
        ax1.set_title("Kept vs dropped", fontweight="bold")
        for i, v in enumerate([len(kept), len(dropped)]):
            ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"redundancy  (method={method})",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_redundancy_compare(df, candidates, fig_width, fig_height, dpi):
    fig, ax = plt.subplots(figsize=(fig_width * 0.6, fig_height), dpi=dpi)
    methods, drops = [], []
    for m in _REDUNDANCY_CANDIDATES:
        thr = _redundancy_default_threshold(m)
        try:
            _, dropped, _, _ = _redundancy_compute(df, candidates, m, thr)
            methods.append(m)
            drops.append(len(dropped))
        except (ValueError, TypeError):
            methods.append(m)
            drops.append(0)
    ax.bar(methods, drops, color="#4c72b0", edgecolor="black", alpha=0.85)
    ax.set_ylabel("features dropped")
    ax.set_title("redundancy COMPARE -- drops per criterion (default threshold)",
                 fontweight="bold")
    for i, v in enumerate(drops):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=10)
    fig.suptitle("redundancy COMPARE", fontsize=14, fontweight="bold")
    return fig


def _plot_relevance(candidates, kept, scores, ranked, method, y_name,
                    fig_width, fig_height, dpi):
    if not candidates:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    kept_set = set(kept)

    def _sv(c):
        v = scores.get(c)
        return 0.0 if v is None else float(v)

    order = list(ranked)
    raw = [_sv(c) for c in order]
    finite = [v for v in raw if v < 1e11]
    cap = (max(finite) * 1.5) if finite else 1.0
    disp = [min(v, cap) for v in raw]
    colors = ["#55a868" if c in kept_set else "#c44e52" for c in order]

    ax0 = axes[0]
    ypos = list(range(len(order)))
    ax0.barh(ypos, disp, color=colors, edgecolor="black", alpha=0.85)
    ax0.set_yticks(ypos)
    ax0.set_yticklabels(order)
    ax0.invert_yaxis()
    ax0.set_xlabel(f"{method} score")
    ax0.set_title(f"Relevance to '{y_name}'  (green=keep, red=drop)",
                  fontweight="bold")

    ax1 = axes[1]
    n_keep = len(kept_set)
    n_drop = len(candidates) - n_keep
    ax1.bar(["kept", "dropped"], [n_keep, n_drop],
            color=["#55a868", "#c44e52"], edgecolor="black", alpha=0.85)
    ax1.set_ylabel("feature count")
    ax1.set_title("Kept vs dropped", fontweight="bold")
    for i, v in enumerate([n_keep, n_drop]):
        ax1.text(i, v, str(v), ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"relevance  (method={method})",
                 fontsize=14, fontweight="bold")
    return fig


def _plot_relevance_compare(summary, fig_width, fig_height, dpi):
    norm = summary.astype(float).copy()
    for col in norm.columns:
        s = norm[col]
        rng = float(s.max() - s.min()) if s.notna().any() else 0.0
        norm[col] = (s - s.min()) / rng if rng > 0 else s * 0.0
    height = min(fig_height + 0.32 * len(norm), 12.0)
    fig, ax = plt.subplots(figsize=(fig_width * 0.6, height), dpi=dpi)
    sns.heatmap(norm, ax=ax, cmap="viridis", annot=True, fmt=".2f",
                cbar_kws={"label": "normalised score"})
    ax.set_title("relevance COMPARE -- score per criterion (colour normalised)",
                 fontweight="bold")
    fig.suptitle("relevance COMPARE", fontsize=14, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Short aliases (consistent with Phase 2 / Phase 3 naming)
# ---------------------------------------------------------------------------

redun = redundancy
relev = relevance


# ===========================================================================
# 4. importance  --  embedded family (read a trained model's importances)
# ===========================================================================

_VALID_IMPORTANCE_METHODS = ("tree", "l1", "linear", "compare")
_IMPORTANCE_CANDIDATES = ("tree", "l1", "linear")
_VALID_RFE_ESTIMATORS = ("tree", "linear", "compare")
_VALID_TASKS = ("classification", "regression")


def _require_sklearn(what: str) -> None:
    """Raise a helpful ImportError if scikit-learn is not installed."""
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"{what} requires scikit-learn, which is not installed. Install "
            f"it with `pip install scikit-learn`. The Filter-family selectors "
            f"redundancy() and relevance() (anova/chi2) need only NumPy."
        ) from exc


def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-score every column; zero-variance columns are left at 0."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def _model_importance(X: np.ndarray, y, method: str,
                      target_kind: str) -> np.ndarray:
    """Fit one model and return a non-negative importance per feature."""
    Xv = _standardize(X) if method in ("l1", "linear") else X
    if method == "tree":
        if target_kind == "classification":
            from sklearn.ensemble import RandomForestClassifier as _M
        else:
            from sklearn.ensemble import RandomForestRegressor as _M
        model = _M(n_estimators=200, random_state=0)
        model.fit(Xv, y)
        return np.asarray(model.feature_importances_, dtype=float)
    if method == "l1":
        if target_kind == "classification":
            from sklearn.linear_model import LogisticRegression
            with warnings.catch_warnings():
                # scikit-learn 1.8 deprecated the explicit 'penalty' argument;
                # the L1 fit is still numerically correct -- silence only that
                # one notice, leaving genuine warnings (e.g. convergence)
                # visible.
                warnings.filterwarnings("ignore", message=".*penalty.*")
                model = LogisticRegression(penalty="l1", solver="liblinear",
                                           C=1.0, random_state=0,
                                           max_iter=2000)
                model.fit(Xv, y)
        else:
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=0.01, random_state=0, max_iter=5000)
            model.fit(Xv, y)
    else:  # linear
        if target_kind == "classification":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(random_state=0, max_iter=2000)
        else:
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        model.fit(Xv, y)
    coef = np.abs(np.asarray(model.coef_, dtype=float))
    if coef.ndim > 1:
        coef = coef.mean(axis=0)
    return coef


def _resolve_task(y_series, task, func_name):
    if task is None:
        return _infer_target_kind(y_series)
    if task not in _VALID_TASKS:
        raise ValueError(
            f"{func_name}: 'task' must be one of {_VALID_TASKS} or None, "
            f"got {task!r}")
    return task


def _importance_scores(df, candidates, y_series, target_kind, method) -> dict:
    X, yv = _clean_feature_matrix(df, candidates, y_series,
                                  "importance", method)
    arr = _model_importance(X, yv, method, target_kind)
    return {candidates[i]: _json_safe_num(arr[i])
            for i in range(len(candidates))}


def importance(
    df: pd.DataFrame,
    y=None,
    cols: Optional[Sequence[str]] = None,
    method: str = "tree",
    *,
    keep: Optional[int] = 10,
    threshold: Optional[float] = None,
    task: Optional[str] = None,
    protect: Optional[Sequence[str]] = None,
    params: Optional[dict] = None,
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
    """Rank features by a trained model's importances, keep the strongest.

    An Embedded-family selector: it trains one model and reads the selection
    that model implies. Requires scikit-learn (imported lazily).

    Two modes. In FIT mode (``params=None``) a model is trained on ``df`` /
    ``y`` and its importances ranked. In APPLY mode (``params`` supplied) the
    saved decision is replayed verbatim -- the DataFrame is subset to the kept
    columns with no re-training -- the safeguard against leakage. ``y`` is not
    needed in apply mode.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    y : str or array-like
        The target. A column name or an array-like aligned to ``df``'s rows.
        Required in fit mode; ignored in apply mode. Auto-protected.
    cols : sequence of str, optional
        The candidate-feature pool (numeric). If ``None`` all numeric
        (non-boolean) columns except the target are used.
    method : {'tree', 'l1', 'linear', 'compare'}
        'tree'    -> random-forest ``feature_importances_``.
        'l1'      -> L1-penalised model; coefficient magnitude on z-scored X.
        'linear'  -> plain linear/logistic coefficient magnitude on z-scored X.
        'compare' -> rank by all three; keep / drop nothing.
    keep : int, optional, default 10
        Keep the top-``keep`` features by importance. Takes precedence over
        ``threshold``. Pass ``keep=None`` to select by ``threshold``.
    threshold : float, optional
        Used only when ``keep is None``: keep features with importance >= this.
    task : {'classification', 'regression'}, optional
        Force the task type; by default it is inferred from ``y``.
    protect : sequence of str, optional
        Columns that must never be dropped.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the learned params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The DataFrame with low-importance candidate columns removed, and -- when
        requested -- the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.importance(df_train, y='churn', method='tree', keep=8,
    ...                          return_params=True)
    >>> df_te = dx.importance(df_test, params=p)          # apply, no re-train
    >>> dx.importance(df, y='churn', method='compare')    # explore
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _selection_apply(df, params, "importance", show, plot,
                                return_df, return_params, return_fig,
                                decimals, df_name, fig_width, fig_height, dpi)

    # ---- FIT MODE -------------------------------------------------------
    if method not in _VALID_IMPORTANCE_METHODS:
        raise ValueError(
            f"'method' must be one of {_VALID_IMPORTANCE_METHODS}, "
            f"got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "method='compare' selects nothing; call importance with a "
            "concrete method (tree/l1/linear) to get params.")
    if keep is not None:
        if not isinstance(keep, (int, np.integer)) or keep < 1:
            raise ValueError(f"'keep' must be an int >= 1 or None, got {keep!r}")
        keep = int(keep)
    _require_sklearn("importance")

    y_name, y_series = _resolve_target(df, y, "importance")
    protect = list(protect or [])
    exclude = protect + ([y_name] if y_name is not None else [])
    candidates = _resolve_features(df, cols, "importance", exclude=exclude)
    if not candidates:
        raise ValueError("importance: no candidate columns left after "
                          "removing the target and protected columns.")
    target_kind = _resolve_task(y_series, task, "importance")

    if method == "compare":
        return _importance_compare(df, candidates, y_name, y_series,
                                   target_kind, show, plot, return_df,
                                   return_params, return_fig, decimals,
                                   df_name, fig_width, fig_height, dpi)

    scores = _importance_scores(df, candidates, y_series, target_kind, method)
    kept, dropped, ranked = _relevance_cut(candidates, scores, keep, threshold)

    out = df.drop(columns=dropped)
    out.attrs = dict(df.attrs)

    params_out = {
        "function": "importance",
        "method": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "target": y_name,
        "target_kind": target_kind,
        "candidates": list(candidates),
        "kept": list(kept),
        "dropped": list(dropped),
        "scores": {c: scores[c] for c in candidates},
        "metadata": {
            "keep": keep,
            "threshold": (None if threshold is None else float(threshold)),
            "protect": protect,
            "n_candidates": len(candidates),
            "n_dropped": len(dropped),
        },
    }

    cut = f"top {keep}" if keep is not None else f"importance >= {threshold:g}"
    decision = (f"Ranked {len(candidates)} candidate feature(s) by '{method}' "
                f"model importance for '{y_name}' ({target_kind}); kept {cut} "
                f"({len(kept)}), dropped {len(dropped)}. Apply to held-out "
                f"data with importance(df_test, params=...).")

    _append_audit(out, {
        "stage": "feature_selection",
        "function": "importance",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "target": y_name, "keep": keep,
                   "n_dropped": len(dropped)},
        "decision": decision,
    })

    summary = _relevance_summary(candidates, kept, scores, ranked, method)
    if show:
        _print_header(f"Importance selection for: {df_name}  "
                      f"(method={method}, target={y_name}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_relevance(candidates, kept, scores, ranked, method,
                              y_name, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _importance_compare(df, candidates, y_name, y_series, target_kind,
                        show, plot, return_df, return_params, return_fig,
                        decimals, df_name, fig_width, fig_height, dpi):
    rows = {}
    for m in _IMPORTANCE_CANDIDATES:
        try:
            rows[m] = _importance_scores(df, candidates, y_series,
                                         target_kind, m)
        except (ValueError, TypeError) as exc:
            rows[m] = {c: None for c in candidates}
            rows[m]["__error__"] = str(exc)
    summary = pd.DataFrame(
        [{m: rows[m].get(c) for m in _IMPORTANCE_CANDIDATES}
         for c in candidates], index=list(candidates))
    summary.index.name = "feature"

    decision = (f"Compared {len(_IMPORTANCE_CANDIDATES)} model-importance "
                f"criteria on {len(candidates)} candidate(s) for '{y_name}' "
                f"({target_kind}). Higher = more important. No columns "
                f"dropped -- pick a method then call importance(method=...).")
    if show:
        _print_header(f"Importance COMPARE for: {df_name}  (nothing dropped)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_relevance_compare(summary, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ===========================================================================
# 5. rfe  --  wrapper family (recursive feature elimination)
# ===========================================================================

def _selection_apply(df, params, expected_fn, show, plot, return_df,
                     return_params, return_fig, decimals, df_name,
                     fig_width, fig_height, dpi):
    """Generic apply path for the model-based selectors (importance, rfe)."""
    if not isinstance(params, dict) or params.get("function") != expected_fn:
        got = (params.get("function") if isinstance(params, dict)
               else type(params).__name__)
        raise ValueError(
            f"params dict is not for {expected_fn!r} (function={got!r}).")
    method = params.get("method", "?")
    kept = params["kept"]
    dropped = params["dropped"]
    missing = [c for c in kept if c not in df.columns]
    if missing:
        raise KeyError(
            f"{expected_fn} apply failed: params expects kept column(s) "
            f"{missing} which are not present in this DataFrame. The data "
            f"does not match the fitted selector.")

    out = df.drop(columns=[c for c in dropped if c in df.columns])
    out.attrs = dict(df.attrs)

    decision = (f"Applied saved '{method}' {expected_fn} selection (fitted "
                f"{params.get('fit_at', '?')}); subset to {len(kept)} kept "
                f"feature(s), dropped {len(dropped)} -- no re-fit, "
                f"leakage-safe.")
    _append_audit(out, {
        "stage": "feature_selection",
        "function": expected_fn,
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"method": method, "n_dropped": len(dropped),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })

    scores = params.get("scores", {})
    candidates = params.get("candidates", kept + dropped)
    ranked = sorted(candidates,
                    key=lambda c: (scores.get(c) if scores.get(c) is not None
                                   else float("-inf")), reverse=True)
    summary = _relevance_summary(candidates, kept, scores, ranked, method)
    if show:
        _print_header(f"{expected_fn.capitalize()} selection for: {df_name}  "
                      f"(method={method}, mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_relevance(candidates, kept, scores, ranked, method,
                              params.get("target"),
                              fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _rfe_estimator(estimator: str, target_kind: str):
    """Build the scikit-learn estimator that RFE will recurse on."""
    if estimator == "tree":
        if target_kind == "classification":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=200, random_state=0)
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=200, random_state=0)
    if target_kind == "classification":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(random_state=0, max_iter=2000)
    from sklearn.linear_model import LinearRegression
    return LinearRegression()


def _rfe_compute(df, candidates, y_series, target_kind, estimator, keep, step):
    """Run RFE; return (kept, dropped, ranking dict raw 1..k)."""
    X, yv = _clean_feature_matrix(df, candidates, y_series, "rfe", estimator)
    Xv = _standardize(X) if estimator == "linear" else X
    n = len(candidates)
    if keep >= n:
        return list(candidates), [], {c: 1 for c in candidates}
    from sklearn.feature_selection import RFE
    sel = RFE(estimator=_rfe_estimator(estimator, target_kind),
              n_features_to_select=keep, step=step)
    sel.fit(Xv, yv)
    support = sel.support_
    rk = sel.ranking_
    kept = [candidates[i] for i in range(n) if support[i]]
    dropped = [candidates[i] for i in range(n) if not support[i]]
    ranking = {candidates[i]: int(rk[i]) for i in range(n)}
    return kept, dropped, ranking


def rfe(
    df: pd.DataFrame,
    y=None,
    cols: Optional[Sequence[str]] = None,
    *,
    keep: int = 10,
    estimator: str = "tree",
    step: int = 1,
    task: Optional[str] = None,
    protect: Optional[Sequence[str]] = None,
    params: Optional[dict] = None,
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
    """Recursive Feature Elimination -- the wrapper-family selector.

    Fits a model, drops the weakest feature(s), refits, and repeats until only
    ``keep`` features remain. Requires scikit-learn (imported lazily).

    Two modes. In FIT mode (``params=None``) the elimination runs on ``df`` /
    ``y``. In APPLY mode (``params`` supplied) the saved decision is replayed
    verbatim -- the DataFrame is subset to the kept columns with no re-fitting.
    ``y`` is not needed in apply mode.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    y : str or array-like
        The target. Required in fit mode; ignored in apply mode. Auto-protected.
    cols : sequence of str, optional
        The candidate-feature pool (numeric). If ``None`` all numeric
        (non-boolean) columns except the target are used.
    keep : int, default 10
        Number of features RFE should keep.
    estimator : {'tree', 'linear', 'compare'}
        The model RFE recurses on: a random forest, a linear/logistic model,
        or 'compare' to run both and report without dropping anything.
    step : int, default 1
        How many features to eliminate per iteration.
    task : {'classification', 'regression'}, optional
        Force the task type; by default it is inferred from ``y``.
    protect : sequence of str, optional
        Columns that must never be dropped.
    params : dict, optional
        A params dict from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the learned params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The DataFrame reduced to the kept features, and -- when requested --
        the params dict and/or the matplotlib figure.

    Examples
    --------
    >>> df_tr, p = dx.rfe(df_train, y='churn', keep=8, estimator='tree',
    ...                   return_params=True)
    >>> df_te = dx.rfe(df_test, params=p)                 # apply, no re-fit
    >>> dx.rfe(df, y='churn', estimator='compare')        # explore
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _selection_apply(df, params, "rfe", show, plot, return_df,
                                return_params, return_fig, decimals, df_name,
                                fig_width, fig_height, dpi)

    # ---- FIT MODE -------------------------------------------------------
    if estimator not in _VALID_RFE_ESTIMATORS:
        raise ValueError(
            f"'estimator' must be one of {_VALID_RFE_ESTIMATORS}, "
            f"got {estimator!r}")
    if estimator == "compare" and return_params:
        raise ValueError(
            "estimator='compare' selects nothing; call rfe with a concrete "
            "estimator (tree/linear) to get params.")
    if not isinstance(keep, (int, np.integer)) or keep < 1:
        raise ValueError(f"'keep' must be an int >= 1, got {keep!r}")
    if not isinstance(step, (int, np.integer)) or step < 1:
        raise ValueError(f"'step' must be an int >= 1, got {step!r}")
    keep, step = int(keep), int(step)
    _require_sklearn("rfe")

    y_name, y_series = _resolve_target(df, y, "rfe")
    protect = list(protect or [])
    exclude = protect + ([y_name] if y_name is not None else [])
    candidates = _resolve_features(df, cols, "rfe", exclude=exclude)
    if not candidates:
        raise ValueError("rfe: no candidate columns left after removing the "
                          "target and protected columns.")
    target_kind = _resolve_task(y_series, task, "rfe")

    if estimator == "compare":
        return _rfe_compare(df, candidates, y_name, y_series, target_kind,
                            keep, step, show, plot, return_df, return_params,
                            return_fig, decimals, df_name,
                            fig_width, fig_height, dpi)

    kept, dropped, ranking = _rfe_compute(df, candidates, y_series,
                                          target_kind, estimator, keep, step)
    max_rank = max(ranking.values())
    scores = {c: float(max_rank + 1 - ranking[c]) for c in candidates}
    ranked = sorted(candidates, key=lambda c: ranking[c])

    out = df.drop(columns=dropped)
    out.attrs = dict(df.attrs)

    params_out = {
        "function": "rfe",
        "method": estimator,
        "version": __version__,
        "fit_at": _now_iso(),
        "target": y_name,
        "target_kind": target_kind,
        "candidates": list(candidates),
        "kept": list(kept),
        "dropped": list(dropped),
        "scores": {c: scores[c] for c in candidates},
        "metadata": {
            "keep": keep,
            "step": step,
            "estimator": estimator,
            "protect": protect,
            "rfe_ranking": {c: ranking[c] for c in candidates},
            "n_candidates": len(candidates),
            "n_dropped": len(dropped),
        },
    }

    decision = (f"Recursively eliminated features for '{y_name}' "
                f"({target_kind}) with a '{estimator}' estimator (step="
                f"{step}); kept {len(kept)} of {len(candidates)} candidate(s),"
                f" dropped {len(dropped)}. Apply to held-out data with "
                f"rfe(df_test, params=...).")

    _append_audit(out, {
        "stage": "feature_selection",
        "function": "rfe",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"estimator": estimator, "target": y_name, "keep": keep,
                   "step": step, "n_dropped": len(dropped)},
        "decision": decision,
    })

    summary = _relevance_summary(candidates, kept, scores, ranked, estimator)
    if show:
        _print_header(f"RFE selection for: {df_name}  "
                      f"(estimator={estimator}, target={y_name}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_relevance(candidates, kept, scores, ranked, estimator,
                              y_name, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _rfe_compare(df, candidates, y_name, y_series, target_kind, keep, step,
                 show, plot, return_df, return_params, return_fig, decimals,
                 df_name, fig_width, fig_height, dpi):
    table = {}
    for est in ("tree", "linear"):
        try:
            kept, _, _ = _rfe_compute(df, candidates, y_series, target_kind,
                                      est, keep, step)
            table[est] = {c: (1 if c in set(kept) else 0) for c in candidates}
        except (ValueError, TypeError) as exc:
            table[est] = {c: None for c in candidates}
            table[est]["__error__"] = str(exc)
    summary = pd.DataFrame(
        [{est: table[est].get(c) for est in ("tree", "linear")}
         for c in candidates], index=list(candidates))
    summary.index.name = "feature"

    decision = (f"Compared RFE with 2 estimator(s) on {len(candidates)} "
                f"candidate(s) for '{y_name}' ({target_kind}), keeping {keep} "
                f"each. 1 = kept, 0 = dropped. No columns dropped -- pick an "
                f"estimator then call rfe(estimator=...).")
    if show:
        _print_header(f"RFE COMPARE for: {df_name}  (nothing dropped)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_relevance_compare(summary, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(df, None, fig, return_df, return_params, return_fig)


# ---------------------------------------------------------------------------
# Stage 5.2 short aliases
# ---------------------------------------------------------------------------

imps = importance


# ===========================================================================
# 6. selectpipe  --  chain the four feature-selection functions (Stage 5.3)
# ===========================================================================

_SELECTPIPE_DISPATCH = {
    "redundancy": redundancy,
    "relevance": relevance,
    "importance": importance,
    "rfe": rfe,
}

# Functions that need a target; selectpipe injects the pipeline-level y here.
_SELECTPIPE_TARGET_FNS = ("relevance", "importance", "rfe")

# dextra I/O / return-control flags that selectpipe owns; stripped + warned.
_SELECTPIPE_CONTROL_FLAGS = (
    "return_params", "return_df", "return_fig", "show", "plot", "params",
    "df_name", "decimals", "fig_width", "fig_height", "dpi",
)


def _selectpipe_compare_key(fn_name: str) -> str:
    """Return the kwarg that selects the method for a given selector."""
    return "estimator" if fn_name == "rfe" else "method"


def _selectpipe_validate_steps(steps) -> list:
    """Validate the steps list; return a clean list of (fn_name, kwargs)."""
    if not isinstance(steps, (list, tuple)) or len(steps) == 0:
        raise ValueError(
            "selectpipe fit mode requires 'steps' to be a non-empty list of "
            "dicts, e.g. [{'fn': 'redundancy', 'method': 'correlation'}].")
    clean = []
    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(
                f"selectpipe step {idx}: each step must be a dict, got "
                f"{type(step).__name__}.")
        if "fn" not in step:
            raise ValueError(
                f"selectpipe step {idx}: missing required key 'fn' (one of "
                f"{tuple(_SELECTPIPE_DISPATCH)}).")
        fn_name = step["fn"]
        if fn_name not in _SELECTPIPE_DISPATCH:
            raise ValueError(
                f"selectpipe step {idx}: unknown fn {fn_name!r}; valid "
                f"functions are {tuple(_SELECTPIPE_DISPATCH)}.")
        kwargs = {k: v for k, v in step.items() if k != "fn"}
        # compare mode is forbidden inside a pipeline (Stage 5.3 decision).
        ckey = _selectpipe_compare_key(fn_name)
        if str(kwargs.get(ckey, "")).lower() == "compare":
            raise ValueError(
                f"selectpipe step {idx} (fn={fn_name}): {ckey}='compare' is "
                f"not allowed inside a pipeline. selectpipe is a commit tool "
                f"-- explore with {fn_name}({ckey}='compare') on its own "
                f"first, then chain the chosen {ckey} here.")
        collided = [k for k in kwargs if k in _SELECTPIPE_CONTROL_FLAGS]
        if collided:
            warnings.warn(
                f"selectpipe step {idx} (fn={fn_name}): control flag(s) "
                f"{collided} are managed by selectpipe and were ignored.",
                stacklevel=3)
            kwargs = {k: v for k, v in kwargs.items()
                      if k not in _SELECTPIPE_CONTROL_FLAGS}
        clean.append((fn_name, kwargs))
    return clean


def _plot_selectpipe(summary_rows, input_cols, df_name, mode,
                     fig_width, fig_height, dpi):
    """Two-panel visual: columns dropped per step + DataFrame width shrink."""
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    labels = [f"{i}:{r['fn']}" for i, r in enumerate(summary_rows)]
    dropped = [r["cols_dropped"] for r in summary_rows]
    after = [r["cols_after"] for r in summary_rows]

    ax0 = axes[0]
    ax0.bar(range(len(labels)), dropped, color="#c44e52",
            edgecolor="black", alpha=0.85)
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, rotation=45, ha="right")
    ax0.set_ylabel("columns dropped")
    ax0.set_title("Columns dropped per step", fontweight="bold")
    for i, v in enumerate(dropped):
        ax0.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    ax1 = axes[1]
    xs = ["input"] + labels
    ys = [input_cols] + after
    pos = list(range(len(xs)))
    ax1.plot(pos, ys, marker="o", color="#4c72b0", linewidth=2)
    ax1.fill_between(pos, ys, alpha=0.15, color="#4c72b0")
    ax1.set_xticks(pos)
    ax1.set_xticklabels(xs, rotation=45, ha="right")
    ax1.set_ylabel("total columns")
    ax1.set_title("DataFrame width through the pipeline", fontweight="bold")
    for x, yv in zip(pos, ys):
        ax1.text(x, yv, str(yv), ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"selectpipe  (mode={mode})  --  {df_name}",
                 fontsize=14, fontweight="bold")
    return fig


def selectpipe(
    df: pd.DataFrame,
    steps: Optional[Sequence[dict]] = None,
    params: Optional[dict] = None,
    *,
    y=None,
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
    """Chain the four dextra feature-selection functions into one pipeline.

    selectpipe is the Stage 5.3 convenience wrapper. It runs ``redundancy``,
    ``relevance``, ``importance`` and ``rfe`` in sequence, threading the
    progressively narrower DataFrame from one step to the next, and collects
    every step's ``params`` dict into a single combined, versioned,
    JSON-serialisable artifact -- a lightweight selection record.

    Two modes. In FIT mode (``steps`` supplied) each step is fitted in order;
    a step's candidate pool is whatever survived the previous steps. In APPLY
    mode (``params`` or ``load_path`` supplied) the saved per-step decisions
    are replayed verbatim -- each step only subsets to its kept columns, no
    re-scoring -- the safeguard against selection leakage.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    steps : sequence of dict, optional
        Fit-mode recipe. Each dict has a ``'fn'`` key naming one of
        ``redundancy / relevance / importance / rfe``; every other key is
        forwarded as a keyword argument to that function, e.g.
        ``{'fn': 'relevance', 'method': 'anova', 'keep': 15}``.
        ``method='compare'`` (or ``estimator='compare'`` for rfe) is rejected.
    params : dict, optional
        Apply-mode artifact: a combined dict returned by an earlier fit.
        Triggers apply mode; ``steps`` must not also be given.
    y : str or array-like, optional
        The shared target. Auto-injected into every ``relevance`` /
        ``importance`` / ``rfe`` step that does not specify its own ``y``. When
        ``y`` is a column name it is also shielded from ``redundancy`` steps so
        a target-free filter can never drop the target.
    save_path : str, optional
        Fit mode only. Writes the combined params dict to this path as JSON.
    load_path : str, optional
        Apply-mode shortcut. Reads the combined params from this JSON file,
        then applies it. Mutually exclusive with ``params`` and ``steps``.
    return_params : bool, default False
        If True the combined params dict is returned alongside the DataFrame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The DataFrame reduced to the surviving features, and -- when requested
        -- the combined params dict and/or the matplotlib figure.

    Examples
    --------
    >>> recipe = [
    ...     {'fn': 'redundancy', 'method': 'correlation', 'threshold': 0.9},
    ...     {'fn': 'relevance', 'method': 'anova', 'keep': 15},
    ...     {'fn': 'importance', 'method': 'tree', 'keep': 8},
    ... ]
    >>> df_tr, p = dx.selectpipe(df_train, steps=recipe, y='churn',
    ...                          return_params=True, save_path='select.json')
    >>> df_te = dx.selectpipe(df_test, params=p)          # apply, no re-score
    >>> df_te2 = dx.selectpipe(df_test, load_path='select.json')  # same result
    """
    df = _ensure_pandas(df)
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- resolve mode ---------------------------------------------------
    if load_path is not None:
        if params is not None or steps is not None:
            raise ValueError(
                "selectpipe: 'load_path' is an apply-mode shortcut; do not "
                "also pass 'params' or 'steps'.")
        import json
        with open(load_path, "r", encoding="utf-8") as fh:
            params = json.load(fh)

    if params is not None:
        if steps is not None:
            raise ValueError(
                "selectpipe: pass EITHER 'steps' (fit mode) OR 'params' "
                "(apply mode), not both.")
        if save_path is not None:
            raise ValueError(
                "selectpipe: 'save_path' saves a freshly fitted pipeline and "
                "is valid in fit mode only (when 'steps' is given).")
        return _selectpipe_apply(df, params, show, plot, return_df,
                                 return_params, return_fig, decimals, df_name,
                                 fig_width, fig_height, dpi)

    if steps is None:
        raise ValueError(
            "selectpipe: provide 'steps' to fit a pipeline, or 'params' / "
            "'load_path' to apply a saved one.")

    return _selectpipe_fit(df, steps, y, save_path, show, plot, return_df,
                           return_params, return_fig, decimals, df_name,
                           fig_width, fig_height, dpi)


def _selectpipe_fit(df, steps, y, save_path, show, plot, return_df,
                    return_params, return_fig, decimals, df_name,
                    fig_width, fig_height, dpi):
    clean = _selectpipe_validate_steps(steps)

    out = df.copy()
    out.attrs = dict(df.attrs)
    input_cols = out.shape[1]
    step_params_list, summary_rows, step_summary = [], [], []
    prev_cols = list(out.columns)

    for idx, (fn_name, kwargs) in enumerate(clean):
        fn = _SELECTPIPE_DISPATCH[fn_name]
        call_kwargs = dict(kwargs)
        # Inject the shared target where needed; shield it from redundancy.
        if fn_name in _SELECTPIPE_TARGET_FNS:
            if "y" not in call_kwargs and y is not None:
                call_kwargs["y"] = y
        elif fn_name == "redundancy" and isinstance(y, str):
            existing = list(call_kwargs.get("protect", []) or [])
            if y not in existing:
                call_kwargs["protect"] = existing + [y]
        before_n = out.shape[1]
        try:
            new_out, sp = fn(out, return_params=True, return_df=True,
                             show=False, plot=False, **call_kwargs)
        except Exception as exc:
            raise type(exc)(
                f"selectpipe step {idx} (fn={fn_name}, fit): {exc}") from exc
        out = new_out
        after_n = out.shape[1]
        removed = [c for c in prev_cols if c not in out.columns]
        prev_cols = list(out.columns)
        step_params_list.append(sp)
        summary_rows.append({
            "fn": fn_name, "method": str(sp.get("method", "-")),
            "cols_before": before_n, "cols_after": after_n,
            "cols_dropped": len(removed)})
        step_summary.append({"step": idx, "fn": fn_name,
                             "method": sp.get("method")})

    combined = {
        "function": "selectpipe",
        "version": __version__,
        "fit_at": _now_iso(),
        "steps": step_params_list,
        "metadata": {
            "n_steps": len(clean),
            "step_summary": step_summary,
            "input_shape": list(df.shape),
            "output_shape": list(out.shape),
            "kept_final": list(out.columns),
        },
    }

    saved_note = ""
    if save_path is not None:
        import json
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(combined, fh, indent=2)
        saved_note = f" Saved to '{save_path}'."

    n_dropped = input_cols - out.shape[1]
    chain = " -> ".join(r["fn"] for r in summary_rows)
    decision = (f"Fitted a {len(clean)}-step selectpipe pipeline ({chain}); "
                f"dropped {n_dropped} column(s), {out.shape[1]} remain; "
                f"combined params is a versioned, JSON-serialisable "
                f"artifact.{saved_note} Apply to held-out data with "
                f"selectpipe(df_test, params=...).")

    _append_audit(out, {
        "stage": "feature_selection",
        "function": "selectpipe",
        "timestamp": combined["fit_at"],
        "mode": "fit",
        "params": {"n_steps": len(clean), "chain": chain,
                   "save_path": save_path},
        "decision": decision,
    })

    summary = pd.DataFrame(summary_rows)
    summary.index.name = "step"
    if show:
        _print_header(f"Feature-selection pipeline for: {df_name}  "
                      f"({len(clean)} step(s), mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_selectpipe(summary_rows, input_cols, df_name, "fit",
                               fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, combined, fig, return_df, return_params, return_fig)


def _selectpipe_apply(df, params, show, plot, return_df, return_params,
                      return_fig, decimals, df_name, fig_width, fig_height,
                      dpi):
    if not isinstance(params, dict) or params.get("function") != "selectpipe":
        got = (params.get("function") if isinstance(params, dict)
               else type(params).__name__)
        raise ValueError(
            f"selectpipe apply: params is not a selectpipe pipeline "
            f"(function={got!r}).")
    step_params = params.get("steps")
    if not isinstance(step_params, list) or len(step_params) == 0:
        raise ValueError(
            "selectpipe apply: params['steps'] must be a non-empty list of "
            "per-function params dicts.")

    out = df.copy()
    out.attrs = dict(df.attrs)
    input_cols = out.shape[1]
    summary_rows = []

    for idx, sp in enumerate(step_params):
        if not isinstance(sp, dict) or "function" not in sp:
            raise ValueError(
                f"selectpipe apply: step {idx} is not a valid params dict.")
        fn_name = sp["function"]
        if fn_name not in _SELECTPIPE_DISPATCH:
            raise ValueError(
                f"selectpipe apply: step {idx} references unknown function "
                f"{fn_name!r}.")
        step_dropped = sp.get("dropped")
        if not isinstance(step_dropped, list):
            raise ValueError(
                f"selectpipe apply: step {idx} ({fn_name}) params has no "
                f"valid 'dropped' list.")
        before_n = out.shape[1]
        # A selection step only ever removes columns. Replaying the saved
        # 'dropped' list -- intersected with the columns actually present --
        # keeps the pipeline tolerant: a column an earlier step kept may have
        # been removed by a later step, and re-applying to already-selected
        # data is a clean no-op (idempotency).
        out = out.drop(columns=[c for c in step_dropped if c in out.columns])
        after_n = out.shape[1]
        summary_rows.append({
            "fn": fn_name, "method": str(sp.get("method", "-")),
            "cols_before": before_n, "cols_after": after_n,
            "cols_dropped": before_n - after_n})

    out.attrs = dict(df.attrs)
    n_dropped = input_cols - out.shape[1]
    fit_at = params.get("fit_at", "?")
    chain = " -> ".join(r["fn"] for r in summary_rows)
    decision = (f"Applied a saved {len(summary_rows)}-step selectpipe "
                f"pipeline ({chain}; fitted {fit_at}); dropped {n_dropped} "
                f"column(s), {out.shape[1]} remain; no re-scoring -- "
                f"leakage-safe.")

    _append_audit(out, {
        "stage": "feature_selection",
        "function": "selectpipe",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"n_steps": len(summary_rows), "chain": chain,
                   "fit_at": fit_at},
        "decision": decision,
    })

    summary = pd.DataFrame(summary_rows)
    summary.index.name = "step"
    if show:
        _print_header(f"Feature-selection pipeline for: {df_name}  "
                      f"({len(summary_rows)} step(s), mode=apply)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_selectpipe(summary_rows, input_cols, df_name, "apply",
                               fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


# Stage 5.3 short alias
selpipe = selectpipe
