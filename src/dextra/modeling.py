"""Modeling helpers for dextra - Phase 6 of the Roadmap.

Implements the fit / apply / compare framework documented in
MODELING_PHILOSOPHY.md at the project root. Modeling learns a function from the
selected features to a target and produces an instant, well-diagnosed BASELINE
with a single call. It is baseline-first, not a tuning / AutoML framework.

Every function in this module:

* Accepts a pandas DataFrame and returns a NEW DataFrame (immutable; the
  original is never mutated) with the model's predictions appended.
* Supports three modes:
    - FIT mode    : trains one baseline algorithm on the data passed.
    - APPLY mode  : re-uses a saved ``params`` artifact verbatim -- it only
      predicts, never re-fitting. This is the safeguard against modeling leakage.
    - COMPARE mode: cross-validates every candidate algorithm, ranks them, and
      writes nothing (no artifact, no prediction column).
* Exposes a HYBRID ``params`` artifact via ``return_params=True``: a
  JSON-serialisable descriptor PLUS the fitted scikit-learn estimator under
  ``params['estimator']`` (persist it with joblib, drop it into an sklearn
  Pipeline / GridSearchCV). Strip the estimator key for a pure-JSON descriptor.
* Prints a metrics table (train vs cross-validated, side by side).
* Renders a multi-panel diagnostic figure.
* Prints a one-line ``Decision:`` sentence naming the metric and the split.
* Appends an entry to ``df.attrs['dextra_audit']``.

scikit-learn is imported lazily, only when a model is actually fitted, so the
rest of dextra keeps working without it (install the ``ml`` extra to enable).

Stage 6.1 - regression:
  - regress(df, y, method=...)  linear / ridge / lasso / tree / forest / compare
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ._utils import get_variable_name
from ._version import __version__

try:
    from IPython.display import display as _ipy_display
except ImportError:  # pragma: no cover
    _ipy_display = None


sns.set_style("whitegrid")

AUDIT_KEY = "dextra_audit"

# The expanded-baseline candidate set for regression compare mode.
_REGRESS_METHODS = ("linear", "ridge", "lasso", "tree", "forest")
_VALID_REGRESS_METHODS = _REGRESS_METHODS + ("compare",)
# Linear-family methods benefit from feature standardisation.
_REGRESS_LINEAR = ("linear", "ridge", "lasso")


# ---------------------------------------------------------------------------
# Shared helpers (mirrors selection.py / features.py conventions)
# ---------------------------------------------------------------------------

def _display(frame: pd.DataFrame) -> None:
    if _ipy_display is not None:
        _ipy_display(frame)
    else:
        with pd.option_context("display.max_colwidth", 60):
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


def _require_sklearn(what: str) -> None:
    """Raise a helpful ImportError if scikit-learn is not installed."""
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"{what} requires scikit-learn, which is not installed. Install it "
            f"with `pip install scikit-learn` (or `pip install dextra[ml]`)."
        ) from exc


def _auto_numeric_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


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
    return (s.name if s.name is not None else "target"), s


def _resolve_features(df, cols, func_name, exclude):
    """Validate an explicit cols selector or auto-pick numeric feature columns."""
    exclude = set(exclude or [])
    if cols is None:
        chosen = [c for c in _auto_numeric_cols(df) if c not in exclude]
        if not chosen:
            raise ValueError(
                f"{func_name}: no numeric feature columns found. "
                f"Pass cols= explicitly.")
        return chosen
    chosen = list(dict.fromkeys(cols))
    bad = [c for c in chosen if c not in df.columns]
    if bad:
        raise KeyError(f"{func_name}: cols references columns not in df: {bad}")
    chosen = [c for c in chosen if c not in exclude]
    if not chosen:
        raise ValueError(f"{func_name}: no feature columns left after "
                         f"excluding the target.")
    return chosen


def _clean_xy(df, features, y_series, func_name):
    """Coerce features + target to numeric, drop rows with any NaN, validate."""
    X = df[features].apply(pd.to_numeric, errors="coerce")
    yv = pd.to_numeric(y_series, errors="coerce")
    mask = X.notna().all(axis=1) & yv.notna()
    X = X.loc[mask]
    yv = yv.loc[mask]
    if len(X) < 2:
        raise ValueError(
            f"{func_name}: need >= 2 complete (non-NaN) rows after cleaning; "
            f"got {len(X)}.")
    return X, yv


# ---------------------------------------------------------------------------
# Estimator construction & metrics
# ---------------------------------------------------------------------------

def _regress_estimator(method: str, standardize: bool):
    """Build an sklearn Pipeline (optional scaler + the chosen regressor).

    Returning a Pipeline keeps the artifact fully sklearn-compatible: it can be
    persisted with joblib and dropped into Pipeline / GridSearchCV unchanged.
    """
    from sklearn.pipeline import Pipeline
    if method == "linear":
        from sklearn.linear_model import LinearRegression
        est = LinearRegression()
    elif method == "ridge":
        from sklearn.linear_model import Ridge
        est = Ridge(alpha=1.0)
    elif method == "lasso":
        from sklearn.linear_model import Lasso
        est = Lasso(alpha=0.01, max_iter=5000)
    elif method == "tree":
        from sklearn.tree import DecisionTreeRegressor
        est = DecisionTreeRegressor(random_state=0)
    elif method == "forest":
        from sklearn.ensemble import RandomForestRegressor
        est = RandomForestRegressor(n_estimators=200, random_state=0)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"regress: unknown method {method!r}")
    steps = []
    if standardize:
        from sklearn.preprocessing import StandardScaler
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", est))
    return Pipeline(steps)


def _hyperparams(method: str) -> dict:
    return {
        "linear": {},
        "ridge": {"alpha": 1.0},
        "lasso": {"alpha": 0.01, "max_iter": 5000},
        "tree": {"random_state": 0},
        "forest": {"n_estimators": 200, "random_state": 0},
    }[method]


def _effective_cv(cv: int, n: int) -> int:
    return int(max(2, min(cv, n)))


def _cv_metrics(estimator, X, yv, cv_folds: int) -> dict:
    """K-fold cross-validated regression metrics (the honest estimate)."""
    from sklearn.model_selection import KFold, cross_validate
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
    scoring = ["r2", "neg_root_mean_squared_error", "neg_mean_absolute_error"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cross_validate(estimator, X, yv, cv=kf, scoring=scoring)
    return {
        "r2": _json_safe_num(np.mean(res["test_r2"])),
        "rmse": _json_safe_num(-np.mean(res["test_neg_root_mean_squared_error"])),
        "mae": _json_safe_num(-np.mean(res["test_neg_mean_absolute_error"])),
    }


def _train_metrics(yv, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(yv, y_pred)))
    return {
        "r2": _json_safe_num(r2_score(yv, y_pred)),
        "rmse": _json_safe_num(rmse),
        "mae": _json_safe_num(mean_absolute_error(yv, y_pred)),
    }


def _descriptor(params: dict) -> dict:
    """The JSON-serialisable view of a params artifact (drops the estimator)."""
    return {k: v for k, v in params.items() if k != "estimator"}


def _fmt_metric(v, decimals: int) -> str:
    """Format a metric for the Decision sentence; None/NaN -> 'n/a'.

    Degenerate inputs (a constant target, or a constant target inside a CV
    fold) make R^2 undefined; _json_safe_num maps that to None. This keeps the
    one-line summary safe instead of raising on ``f"{None:.4f}"``.
    """
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if np.isnan(f):
        return "n/a"
    return f"{f:,.{decimals}f}"


# ===========================================================================
# regress  --  instant regression baseline (fit / apply / compare)
# ===========================================================================

def regress(
    df: pd.DataFrame,
    y=None,
    cols: Optional[Sequence[str]] = None,
    method: str = "forest",
    *,
    cv: int = 5,
    standardize: Optional[bool] = None,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 13.0,
    fig_height: float = 5.0,
    dpi: int = 110,
):
    """Fit an instant, cross-validated regression baseline in one line.

    Three modes. In FIT mode (``params=None`` and a concrete ``method``) one
    baseline regressor is trained on ``df`` and its in-sample predictions are
    appended as ``"<target>_pred"``. In APPLY mode (``params`` supplied) the
    saved fitted estimator predicts on new data with no re-fit -- the safeguard
    against modeling leakage. In COMPARE mode (``method='compare'``) every
    candidate regressor is cross-validated and ranked, but nothing is written.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    y : str or array-like
        The numeric target. A column name (fit mode) or a Series/array aligned
        to ``df``. Not needed in apply mode (it comes from ``params``).
    cols : sequence of str, optional
        Feature columns. If ``None`` all numeric (non-boolean) columns except
        the target are used. Ignored in apply mode.
    method : {'linear', 'ridge', 'lasso', 'tree', 'forest', 'compare'}
        The baseline algorithm, or 'compare' to rank them all.
    cv : int, default 5
        Number of cross-validation folds (clamped to ``[2, n_rows]``).
    standardize : bool, optional
        Whether to z-score features. Default: ``True`` for the linear family
        (linear/ridge/lasso), ``False`` for tree/forest.
    params : dict, optional
        A hybrid artifact from an earlier fit. Triggers apply mode.
    return_params : bool, default False
        If True the hybrid ``params`` artifact is returned alongside the frame.
    show, plot, return_df, return_fig, decimals, df_name : standard dextra flags.

    Returns
    -------
    pandas.DataFrame
        The input frame plus a prediction column, and -- when requested -- the
        hybrid ``params`` artifact and/or the matplotlib figure.

    Examples
    --------
    >>> out, p = dx.regress(df_train, y='price', method='forest', return_params=True)
    >>> preds = dx.regress(df_test, params=p)            # apply, no re-fit
    >>> dx.regress(df_train, y='price', method='compare')  # rank baselines
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'df' must be a pandas DataFrame, got {type(df).__name__}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _regress_apply(df, params, show, plot, return_df,
                              return_params, return_fig, decimals, df_name,
                              fig_width, fig_height, dpi)

    # ---- FIT / COMPARE guards (before importing sklearn) ----------------
    if method not in _VALID_REGRESS_METHODS:
        raise ValueError(
            f"regress: 'method' must be one of {_VALID_REGRESS_METHODS}, "
            f"got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "regress: method='compare' fits no single model and writes no "
            "artifact; call with a concrete method to get params.")
    if cv < 2:
        raise ValueError(f"regress: 'cv' must be >= 2, got {cv}")

    name, y_series = _resolve_target(df, y, "regress")
    if not pd.api.types.is_numeric_dtype(y_series) or \
            pd.api.types.is_bool_dtype(y_series):
        raise ValueError(
            "regress: the target is not numeric. Regression needs a numeric "
            "target; use classify() for a categorical target.")
    target_name = name if isinstance(y, str) else (name or "target")
    features = _resolve_features(df, cols, "regress", exclude=[target_name])

    _require_sklearn("regress")
    X, yv = _clean_xy(df, features, y_series, "regress")
    cv_folds = _effective_cv(cv, len(X))

    if method == "compare":
        return _regress_compare(df, X, yv, features, target_name, cv_folds,
                                standardize, show, plot, return_df,
                                return_fig, decimals, df_name,
                                fig_width, fig_height, dpi)

    use_scaler = (method in _REGRESS_LINEAR) if standardize is None \
        else bool(standardize)
    estimator = _regress_estimator(method, use_scaler)
    Xa = X.to_numpy(dtype=float)
    ya = yv.to_numpy(dtype=float)
    cv_m = _cv_metrics(estimator, Xa, ya, cv_folds)
    with warnings.catch_warnings():
        try:  # silence only iterative-solver convergence chatter (e.g. Lasso)
            from sklearn.exceptions import ConvergenceWarning
            warnings.simplefilter("ignore", ConvergenceWarning)
        except Exception:
            pass
        estimator.fit(Xa, ya)
        y_pred_clean = estimator.predict(Xa)
    train_m = _train_metrics(yv.to_numpy(dtype=float), y_pred_clean)

    out = df.copy()
    out.attrs = dict(df.attrs)
    pred_col = f"{target_name}_pred"
    full_pred = pd.Series(
        estimator.predict(
            df[features].apply(pd.to_numeric, errors="coerce")
            .fillna(X.mean()).to_numpy(dtype=float)),
        index=df.index, name=pred_col)
    out[pred_col] = full_pred

    params_out = {
        "function": "regress",
        "task": "regression",
        "algorithm": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "features": list(features),
        "target": target_name,
        "hyperparams": _hyperparams(method),
        "metrics": {"train": train_m, "cv": cv_m},
        "pred_col": pred_col,
        "metadata": {"n_train": int(len(X)), "n_features": len(features),
                     "standardize": bool(use_scaler), "cv_folds": cv_folds},
        "estimator": estimator,
    }

    decision = (
        f"'{method}' regression baseline -- CV R^2={_fmt_metric(cv_m['r2'], decimals)}, "
        f"RMSE={_fmt_metric(cv_m['rmse'], decimals)}, MAE={_fmt_metric(cv_m['mae'], decimals)} "
        f"({cv_folds}-fold). Train R^2={_fmt_metric(train_m['r2'], decimals)}. "
        f"In-sample predictions added as '{pred_col}'. "
        f"Persist with joblib.dump(params['estimator'], ...).")

    _append_audit(out, {
        "stage": "modeling",
        "function": "regress",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "features": list(features),
                   "target": target_name, "cv": cv_folds},
        "decision": decision,
    })

    summary = _metrics_table(params_out['metrics'])
    if show:
        _print_header(f"Regression baseline for: {df_name}  "
                      f"(method={method}, target={target_name}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_regress(yv.to_numpy(dtype=float), y_pred_clean, method,
                            target_name, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _metrics_table(metrics: dict) -> pd.DataFrame:
    """Render a {split_name: {metric: value}} mapping as a metric-by-split table.

    Family-agnostic: regress/classify use splits {'train', 'cv'}; cluster uses
    {'fit'}. The union of metric names becomes the rows, splits the columns, so
    every Phase 6 family reuses this renderer unchanged.
    """
    splits = list(metrics.keys())
    rows = []
    for d in metrics.values():
        for k in d.keys():
            if k not in rows:
                rows.append(k)
    data = {split: [metrics[split].get(m) for m in rows] for split in splits}
    return pd.DataFrame(data, index=[r.upper() for r in rows])


def _regress_apply(df, params, show, plot, return_df, return_params,
                   return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "regress":
        got = (params.get("function") if isinstance(params, dict)
               else type(params).__name__)
        raise ValueError(f"params dict is not for 'regress' (function={got!r}).")
    estimator = params.get("estimator")
    if estimator is None or not hasattr(estimator, "predict"):
        raise ValueError(
            "regress apply: params has no fitted estimator. Re-fit and pass the "
            "full params (the artifact carries params['estimator']).")
    features = params["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(
            f"regress apply failed: params expects feature column(s) {missing} "
            f"which are not present in this DataFrame.")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError(
            "regress apply: feature columns contain NaN/non-numeric values. "
            "Clean / impute (Phase 3) before predicting.")
    pred_col = params.get("pred_col", f"{params.get('target', 'target')}_pred")
    out = df.copy()
    out.attrs = dict(df.attrs)
    out[pred_col] = pd.Series(estimator.predict(X.to_numpy(dtype=float)),
                              index=df.index, name=pred_col)

    decision = (f"Applied saved '{params.get('algorithm', '?')}' regression "
                f"model (fitted {params.get('fit_at', '?')}); predicted "
                f"{len(out)} row(s) into '{pred_col}' -- no re-fit, "
                f"leakage-safe.")
    _append_audit(out, {
        "stage": "modeling",
        "function": "regress",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"algorithm": params.get("algorithm"),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })
    if show:
        _print_header(f"Regression prediction for: {df_name}  "
                      f"(algorithm={params.get('algorithm')}, mode=apply)")
        _display(_fmt_table(_metrics_table(params["metrics"]), decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _regress_compare(df, X, yv, features, target_name, cv_folds, standardize,
                     show, plot, return_df, return_fig, decimals, df_name,
                     fig_width, fig_height, dpi):
    Xa = X.to_numpy(dtype=float)
    ya = yv.to_numpy(dtype=float)
    rows = {}
    for m in _REGRESS_METHODS:
        use_scaler = (m in _REGRESS_LINEAR) if standardize is None \
            else bool(standardize)
        est = _regress_estimator(m, use_scaler)
        rows[m] = _cv_metrics(est, Xa, ya, cv_folds)
    summary = pd.DataFrame(
        {"CV_R2": {m: rows[m]["r2"] for m in _REGRESS_METHODS},
         "CV_RMSE": {m: rows[m]["rmse"] for m in _REGRESS_METHODS},
         "CV_MAE": {m: rows[m]["mae"] for m in _REGRESS_METHODS}})
    summary = summary.sort_values("CV_R2", ascending=False)
    best = summary.index[0]
    best_r2 = summary.loc[best, "CV_R2"]

    out = df.copy()
    out.attrs = dict(df.attrs)
    decision = (f"Compared {len(_REGRESS_METHODS)} regressors by {cv_folds}-fold "
                f"CV; best by R^2 is '{best}' (R^2={_fmt_metric(best_r2, decimals)}). "
                f"Nothing written -- choose a method to fit.")
    _append_audit(out, {
        "stage": "modeling",
        "function": "regress",
        "timestamp": _now_iso(),
        "mode": "compare",
        "params": {"candidates": list(_REGRESS_METHODS), "cv": cv_folds,
                   "target": target_name},
        "decision": decision,
    })
    if show:
        _print_header(f"Regression model comparison for: {df_name}  "
                      f"(target={target_name}, {cv_folds}-fold CV)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_regress_compare(summary, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    # compare writes nothing and returns no params
    return _ret_pack(out, None, fig, return_df, False, return_fig)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_regress(y_true, y_pred, method, target_name, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Regression diagnostics -- {method} ({target_name})",
                 fontsize=13, fontweight="bold")
    # (1) actual vs predicted
    ax = axes[0]
    ax.scatter(y_true, y_pred, s=18, alpha=0.6, color="#2E75B6",
               edgecolor="white", linewidth=0.4)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], "--", color="#d62728", linewidth=1.6,
            label="perfect (y = x)")
    ax.set_xlabel("actual")
    ax.set_ylabel("predicted")
    ax.set_title("Actual vs predicted")
    ax.legend(loc="upper left", fontsize=9)
    # (2) residuals vs predicted
    ax2 = axes[1]
    resid = y_true - y_pred
    ax2.scatter(y_pred, resid, s=18, alpha=0.6, color="#2ca02c",
                edgecolor="white", linewidth=0.4)
    ax2.axhline(0, color="#d62728", linestyle="--", linewidth=1.6)
    ax2.set_xlabel("predicted")
    ax2.set_ylabel("residual (actual - predicted)")
    ax2.set_title("Residuals vs predicted")
    return fig


def _plot_regress_compare(summary, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle("Regression baselines -- cross-validated comparison",
                 fontsize=13, fontweight="bold")
    order = list(summary.index)
    r2 = [summary.loc[m, "CV_R2"] if summary.loc[m, "CV_R2"] is not None else 0.0
          for m in order]
    rmse = [summary.loc[m, "CV_RMSE"] if summary.loc[m, "CV_RMSE"] is not None
            else 0.0 for m in order]
    axes[0].barh(order, r2, color="#2E75B6")
    axes[0].set_xlabel("CV R^2 (higher is better)")
    axes[0].invert_yaxis()
    axes[0].set_title("R^2 by algorithm")
    axes[1].barh(order, rmse, color="#ff7f0e")
    axes[1].set_xlabel("CV RMSE (lower is better)")
    axes[1].invert_yaxis()
    axes[1].set_title("RMSE by algorithm")
    return fig


# Short alias (consistent with the rest of dextra).
reg = regress


# ===========================================================================
# Stage 6.2 -- classify  (instant classification baseline: fit/apply/compare)
# ===========================================================================

_CLASSIFY_METHODS = ("logistic", "tree", "forest", "knn")
_VALID_CLASSIFY_METHODS = _CLASSIFY_METHODS + ("compare",)
# Distance/linear methods benefit from feature standardisation.
_CLASSIFY_SCALED = ("logistic", "knn")


def _json_safe_label(c):
    """A class label rendered JSON-safe for the params descriptor."""
    if isinstance(c, (bool, np.bool_)):
        return bool(c)
    if isinstance(c, (int, np.integer)):
        return int(c)
    if isinstance(c, (float, np.floating)):
        return float(c)
    return str(c)


def _clean_xy_clf(df, features, y_series, func_name):
    """Like _clean_xy but the target is NOT coerced to numeric (it is a label)."""
    X = df[features].apply(pd.to_numeric, errors="coerce")
    yv = y_series.copy()
    mask = X.notna().all(axis=1) & yv.notna()
    X = X.loc[mask]
    yv = yv.loc[mask]
    if len(X) < 4:
        raise ValueError(
            f"{func_name}: need >= 4 complete (non-NaN) rows after cleaning; "
            f"got {len(X)}.")
    return X, yv


def _classify_n_classes(y_series, func_name) -> int:
    """Validate a classification target and return the number of classes."""
    classes = pd.unique(y_series.dropna())
    n = len(classes)
    if n < 2:
        raise ValueError(
            f"{func_name}: the target has fewer than 2 classes; nothing to "
            f"classify.")
    is_num = (pd.api.types.is_numeric_dtype(y_series)
              and not pd.api.types.is_bool_dtype(y_series))
    if is_num and n > 20:
        raise ValueError(
            f"{func_name}: the target looks continuous ({n} distinct numeric "
            f"values). Use regress() for a numeric target.")
    return int(n)


def _classify_cv_folds(cv: int, y_series) -> int:
    """Clamp folds so StratifiedKFold has >= 1 sample of each class per fold."""
    counts = y_series.value_counts()
    min_class = int(counts.min()) if len(counts) else 0
    if min_class < 2:
        raise ValueError(
            "classify: every class needs >= 2 samples for cross-validation; "
            "the rarest class has fewer.")
    return int(max(2, min(cv, min_class)))


def _classify_estimator(method: str, standardize: bool):
    from sklearn.pipeline import Pipeline
    if method == "logistic":
        from sklearn.linear_model import LogisticRegression
        est = LogisticRegression(max_iter=2000)
    elif method == "tree":
        from sklearn.tree import DecisionTreeClassifier
        est = DecisionTreeClassifier(random_state=0)
    elif method == "forest":
        from sklearn.ensemble import RandomForestClassifier
        est = RandomForestClassifier(n_estimators=200, random_state=0)
    elif method == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        est = KNeighborsClassifier()
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"classify: unknown method {method!r}")
    steps = []
    if standardize:
        from sklearn.preprocessing import StandardScaler
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", est))
    return Pipeline(steps)


def _classify_hyperparams(method: str) -> dict:
    return {
        "logistic": {"max_iter": 2000},
        "tree": {"random_state": 0},
        "forest": {"n_estimators": 200, "random_state": 0},
        "knn": {"n_neighbors": 5},
    }[method]


def _clf_cv_metrics(estimator, X, y, cv_folds: int, binary: bool) -> dict:
    from sklearn.model_selection import StratifiedKFold, cross_validate
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=0)
    auc = "roc_auc" if binary else "roc_auc_ovr_weighted"
    scoring = {"accuracy": "accuracy", "f1": "f1_weighted", "roc_auc": auc}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cross_validate(estimator, X, y, cv=skf, scoring=scoring,
                             error_score=np.nan)
    return {
        "accuracy": _json_safe_num(np.nanmean(res["test_accuracy"])),
        "f1": _json_safe_num(np.nanmean(res["test_f1"])),
        "roc_auc": _json_safe_num(np.nanmean(res["test_roc_auc"])),
    }


def _clf_train_metrics(estimator, X, y, binary: bool) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    y_pred = estimator.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        proba = estimator.predict_proba(X)
        cls = list(estimator.classes_)
        if binary:
            y_bin = (np.asarray(y) == cls[1]).astype(int)
            auc = roc_auc_score(y_bin, proba[:, 1])
        else:
            auc = roc_auc_score(y, proba, multi_class="ovr",
                                average="weighted", labels=cls)
    except Exception:
        auc = None
    return {"accuracy": _json_safe_num(acc), "f1": _json_safe_num(f1),
            "roc_auc": _json_safe_num(auc)}


def classify(
    df: pd.DataFrame,
    y=None,
    cols: Optional[Sequence[str]] = None,
    method: str = "forest",
    *,
    cv: int = 5,
    standardize: Optional[bool] = None,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 13.0,
    fig_height: float = 5.0,
    dpi: int = 110,
):
    """Fit an instant, cross-validated classification baseline in one line.

    Mirrors :func:`regress` exactly (fit / apply / compare, hybrid artifact)
    for a categorical target. In FIT mode the predicted labels are appended as
    ``"<target>_pred"``; APPLY mode predicts with the saved estimator (no
    re-fit); COMPARE mode cross-validates every candidate and writes nothing.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    y : str or array-like
        The categorical target (a column name, or a Series/array aligned to
        ``df``). Not needed in apply mode.
    cols : sequence of str, optional
        Feature columns. Defaults to every numeric (non-boolean) column except
        the target. Ignored in apply mode.
    method : {'logistic', 'tree', 'forest', 'knn', 'compare'}
        The baseline classifier, or 'compare' to rank them all.
    cv : int, default 5
        Stratified cross-validation folds (clamped to the rarest class count).
    standardize : bool, optional
        Default: ``True`` for logistic/knn, ``False`` for tree/forest.
    params, return_params, show, plot, return_df, return_fig, decimals, df_name
        Standard dextra flags (see :func:`regress`).

    Returns
    -------
    pandas.DataFrame
        The input frame plus a predicted-label column, and -- when requested --
        the hybrid ``params`` artifact and/or the matplotlib figure.

    Examples
    --------
    >>> out, p = dx.classify(df_train, y='churn', method='forest', return_params=True)
    >>> preds = dx.classify(df_test, params=p)              # apply, no re-fit
    >>> dx.classify(df_train, y='churn', method='compare')  # rank baselines
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'df' must be a pandas DataFrame, got {type(df).__name__}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _classify_apply(df, params, show, plot, return_df,
                               return_params, return_fig, decimals, df_name,
                               fig_width, fig_height, dpi)

    # ---- FIT / COMPARE guards (before importing sklearn) ----------------
    if method not in _VALID_CLASSIFY_METHODS:
        raise ValueError(
            f"classify: 'method' must be one of {_VALID_CLASSIFY_METHODS}, "
            f"got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "classify: method='compare' fits no single model and writes no "
            "artifact; call with a concrete method to get params.")
    if cv < 2:
        raise ValueError(f"classify: 'cv' must be >= 2, got {cv}")

    name, y_series = _resolve_target(df, y, "classify")
    _classify_n_classes(y_series, "classify")          # early validation
    target_name = name if isinstance(y, str) else (name or "target")
    features = _resolve_features(df, cols, "classify", exclude=[target_name])

    _require_sklearn("classify")
    X, yv = _clean_xy_clf(df, features, y_series, "classify")
    n_classes = _classify_n_classes(yv, "classify")
    binary = (n_classes == 2)
    cv_folds = _classify_cv_folds(cv, yv)

    if method == "compare":
        return _classify_compare(df, X, yv, features, target_name, n_classes,
                                 binary, cv_folds, standardize, show, plot,
                                 return_df, return_fig, decimals, df_name,
                                 fig_width, fig_height, dpi)

    use_scaler = (method in _CLASSIFY_SCALED) if standardize is None \
        else bool(standardize)
    estimator = _classify_estimator(method, use_scaler)
    Xa = X.to_numpy(dtype=float)
    ya = yv.to_numpy()
    cv_m = _clf_cv_metrics(estimator, Xa, ya, cv_folds, binary)
    with warnings.catch_warnings():
        try:
            from sklearn.exceptions import ConvergenceWarning
            warnings.simplefilter("ignore", ConvergenceWarning)
        except Exception:
            pass
        estimator.fit(Xa, ya)
    train_m = _clf_train_metrics(estimator, Xa, ya, binary)

    out = df.copy()
    out.attrs = dict(df.attrs)
    pred_col = f"{target_name}_pred"
    full_X = (df[features].apply(pd.to_numeric, errors="coerce")
              .fillna(X.mean()).to_numpy(dtype=float))
    out[pred_col] = pd.Series(estimator.predict(full_X), index=df.index,
                              name=pred_col)

    params_out = {
        "function": "classify",
        "task": "classification",
        "algorithm": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "features": list(features),
        "target": target_name,
        "classes": [_json_safe_label(c) for c in estimator.classes_],
        "n_classes": int(n_classes),
        "hyperparams": _classify_hyperparams(method),
        "metrics": {"train": train_m, "cv": cv_m},
        "pred_col": pred_col,
        "metadata": {"n_train": int(len(X)), "n_features": len(features),
                     "standardize": bool(use_scaler), "cv_folds": cv_folds,
                     "binary": bool(binary)},
        "estimator": estimator,
    }

    kind = "binary" if binary else f"{n_classes}-class"
    decision = (
        f"'{method}' classification baseline ({kind}) -- CV accuracy="
        f"{_fmt_metric(cv_m['accuracy'], decimals)}, F1="
        f"{_fmt_metric(cv_m['f1'], decimals)}, ROC-AUC="
        f"{_fmt_metric(cv_m['roc_auc'], decimals)} ({cv_folds}-fold, "
        f"stratified). Train accuracy={_fmt_metric(train_m['accuracy'], decimals)}. "
        f"Predicted labels added as '{pred_col}'. "
        f"Persist with joblib.dump(params['estimator'], ...).")

    _append_audit(out, {
        "stage": "modeling",
        "function": "classify",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "features": list(features),
                   "target": target_name, "cv": cv_folds,
                   "n_classes": int(n_classes)},
        "decision": decision,
    })

    summary = _metrics_table(params_out["metrics"])
    if show:
        _print_header(f"Classification baseline for: {df_name}  "
                      f"(method={method}, target={target_name}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_classify(ya, estimator.predict(Xa), list(estimator.classes_),
                             method, target_name, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _classify_apply(df, params, show, plot, return_df, return_params,
                    return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "classify":
        got = (params.get("function") if isinstance(params, dict)
               else type(params).__name__)
        raise ValueError(f"params dict is not for 'classify' (function={got!r}).")
    estimator = params.get("estimator")
    if estimator is None or not hasattr(estimator, "predict"):
        raise ValueError(
            "classify apply: params has no fitted estimator. Re-fit and pass "
            "the full params (the artifact carries params['estimator']).")
    features = params["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(
            f"classify apply failed: params expects feature column(s) "
            f"{missing} which are not present in this DataFrame.")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError(
            "classify apply: feature columns contain NaN/non-numeric values. "
            "Clean / impute (Phase 3) before predicting.")
    pred_col = params.get("pred_col", f"{params.get('target', 'target')}_pred")
    out = df.copy()
    out.attrs = dict(df.attrs)
    out[pred_col] = pd.Series(estimator.predict(X.to_numpy(dtype=float)),
                              index=df.index, name=pred_col)

    decision = (f"Applied saved '{params.get('algorithm', '?')}' classifier "
                f"(fitted {params.get('fit_at', '?')}); predicted {len(out)} "
                f"row(s) into '{pred_col}' -- no re-fit, leakage-safe.")
    _append_audit(out, {
        "stage": "modeling",
        "function": "classify",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"algorithm": params.get("algorithm"),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })
    if show:
        _print_header(f"Classification prediction for: {df_name}  "
                      f"(algorithm={params.get('algorithm')}, mode=apply)")
        _display(_fmt_table(_metrics_table(params["metrics"]), decimals))
        print(f"\nDecision: {decision}\n")
    fig = None
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _classify_compare(df, X, yv, features, target_name, n_classes, binary,
                      cv_folds, standardize, show, plot, return_df, return_fig,
                      decimals, df_name, fig_width, fig_height, dpi):
    Xa = X.to_numpy(dtype=float)
    ya = yv.to_numpy()
    rows = {}
    for m in _CLASSIFY_METHODS:
        use_scaler = (m in _CLASSIFY_SCALED) if standardize is None \
            else bool(standardize)
        est = _classify_estimator(m, use_scaler)
        rows[m] = _clf_cv_metrics(est, Xa, ya, cv_folds, binary)
    summary = pd.DataFrame(
        {"CV_accuracy": {m: rows[m]["accuracy"] for m in _CLASSIFY_METHODS},
         "CV_f1": {m: rows[m]["f1"] for m in _CLASSIFY_METHODS},
         "CV_roc_auc": {m: rows[m]["roc_auc"] for m in _CLASSIFY_METHODS}})
    summary = summary.sort_values("CV_accuracy", ascending=False)
    best = summary.index[0]
    best_acc = summary.loc[best, "CV_accuracy"]

    out = df.copy()
    out.attrs = dict(df.attrs)
    kind = "binary" if binary else f"{n_classes}-class"
    decision = (f"Compared {len(_CLASSIFY_METHODS)} classifiers ({kind}) by "
                f"{cv_folds}-fold stratified CV; best by accuracy is '{best}' "
                f"(accuracy={_fmt_metric(best_acc, decimals)}). Nothing written "
                f"-- choose a method to fit.")
    _append_audit(out, {
        "stage": "modeling",
        "function": "classify",
        "timestamp": _now_iso(),
        "mode": "compare",
        "params": {"candidates": list(_CLASSIFY_METHODS), "cv": cv_folds,
                   "target": target_name, "n_classes": int(n_classes)},
        "decision": decision,
    })
    if show:
        _print_header(f"Classification model comparison for: {df_name}  "
                      f"(target={target_name}, {cv_folds}-fold stratified CV)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")
    fig = None
    if plot:
        fig = _plot_classify_compare(summary, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, None, fig, return_df, False, return_fig)


def _plot_classify(y_true, y_pred, classes, method, target_name,
                   fig_width, fig_height, dpi):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm_norm / row_sums
    labs = [str(c) for c in classes]
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Classification diagnostics -- {method} ({target_name})",
                 fontsize=13, fontweight="bold")
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
    return fig


def _plot_classify_compare(summary, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle("Classification baselines -- cross-validated comparison",
                 fontsize=13, fontweight="bold")
    order = list(summary.index)
    acc = [summary.loc[m, "CV_accuracy"] if summary.loc[m, "CV_accuracy"]
           is not None else 0.0 for m in order]
    f1 = [summary.loc[m, "CV_f1"] if summary.loc[m, "CV_f1"] is not None
          else 0.0 for m in order]
    axes[0].barh(order, acc, color="#2E75B6")
    axes[0].set_xlabel("CV accuracy (higher is better)")
    axes[0].set_xlim(0, 1)
    axes[0].invert_yaxis()
    axes[0].set_title("Accuracy by algorithm")
    axes[1].barh(order, f1, color="#2ca02c")
    axes[1].set_xlabel("CV F1 (weighted)")
    axes[1].set_xlim(0, 1)
    axes[1].invert_yaxis()
    axes[1].set_title("F1 by algorithm")
    return fig


# Short alias (consistent with the rest of dextra).
clf = classify


# ===========================================================================
# Stage 6.3 -- cluster  (instant clustering baseline: fit/apply/compare)
# ===========================================================================

# Unsupervised: never accepts a target y (see MODELING_PHILOSOPHY 2.3 / 4.7).
_CLUSTER_METHODS = ("kmeans", "agglomerative")
_VALID_CLUSTER_METHODS = _CLUSTER_METHODS + ("compare",)


def _clean_x(df, features, func_name):
    """Like :func:`_clean_xy` but target-free: coerce features, drop NaN rows.

    Clustering has no target, so this is the unsupervised cleaning helper the
    contract calls for. Needs >= 3 complete rows (silhouette requires at least
    2 clusters and ``2 <= n_labels <= n_samples - 1``).
    """
    X = df[features].apply(pd.to_numeric, errors="coerce")
    mask = X.notna().all(axis=1)
    X = X.loc[mask]
    if len(X) < 3:
        raise ValueError(
            f"{func_name}: need >= 3 complete (non-NaN) rows after cleaning; "
            f"got {len(X)}.")
    return X


def _cluster_hyperparams(method: str, k: int) -> dict:
    if method == "kmeans":
        return {"n_clusters": int(k), "n_init": 10, "random_state": 0}
    if method == "agglomerative":
        return {"n_clusters": int(k), "linkage": "ward"}
    raise ValueError(f"cluster: unknown method {method!r}")  # pragma: no cover


def _k_grid(k_range, n: int) -> list:
    """The candidate k values, clamped so silhouette stays well-defined."""
    lo, hi = int(k_range[0]), int(k_range[1])
    if lo < 2:
        raise ValueError(f"cluster: k_range lower bound must be >= 2, got {lo}")
    if hi < lo:
        raise ValueError(
            f"cluster: k_range must be (lo, hi) with hi >= lo, got {k_range}")
    hi = min(hi, n - 1)            # silhouette needs n_labels <= n_samples - 1
    if hi < lo:
        hi = lo
    return list(range(lo, hi + 1))


def _fit_cluster_core(method: str, k: int, Xs):
    """Fit one clusterer on already-scaled data.

    Returns ``(predictor, labels, inertia)``. ``agglomerative`` has no native
    ``predict``; a :class:`~sklearn.neighbors.NearestCentroid` is fitted on its
    labels so the persisted estimator can assign new points by the same
    nearest-centroid rule used to deploy hierarchical clustering. ``inertia`` is
    ``None`` for agglomerative (undefined) and rendered ``-`` / ``n/a``.
    """
    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=int(k), n_init=10, random_state=0)
        labels = km.fit_predict(Xs)
        return km, labels, _json_safe_num(km.inertia_)
    if method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import NearestCentroid
        ag = AgglomerativeClustering(n_clusters=int(k))
        labels = ag.fit_predict(Xs)
        nc = NearestCentroid()
        nc.fit(Xs, labels)
        return nc, labels, None
    raise ValueError(f"cluster: unknown method {method!r}")  # pragma: no cover


def _silhouette(Xs, labels):
    """Silhouette score, or ``None`` when undefined (< 2 distinct labels)."""
    if len(np.unique(labels)) < 2:
        return None
    from sklearn.metrics import silhouette_score
    try:
        return _json_safe_num(silhouette_score(Xs, labels))
    except Exception:
        return None


def _assemble_cluster_pipeline(scaler, predictor):
    """Wrap (optional fitted scaler) + fitted predictor in an sklearn Pipeline.

    Both steps are already fitted; the Pipeline is never re-fit. This keeps the
    artifact joblib-persistable and droppable into Pipeline/GridSearchCV, and
    gives a uniform ``predict`` for both kmeans and agglomerative.
    """
    from sklearn.pipeline import Pipeline
    steps = []
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("model", predictor))
    return Pipeline(steps)


def _select_k(method: str, Xs, k_grid):
    """Fit at every candidate k and pick the k with the best silhouette.

    Returns ``(best_k, best_labels, best_inertia, best_predictor, curve)`` where
    ``curve`` maps k -> {'silhouette', 'inertia'} for the elbow / silhouette
    plot. The winning predictor is kept so no re-fit is needed.
    """
    curve = {}
    best = None
    for k in k_grid:
        predictor, labels, inertia = _fit_cluster_core(method, k, Xs)
        sil = _silhouette(Xs, labels)
        curve[k] = {"silhouette": sil, "inertia": inertia}
        score = sil if sil is not None else -np.inf
        if best is None or score > best[0]:
            best = (score, k, labels, inertia, predictor)
    _, best_k, best_labels, best_inertia, best_predictor = best
    return int(best_k), best_labels, best_inertia, best_predictor, curve


def cluster(
    df: pd.DataFrame,
    cols: Optional[Sequence[str]] = None,
    method: str = "kmeans",
    *,
    k: Optional[int] = None,
    k_range: Sequence[int] = (2, 10),
    standardize: Optional[bool] = None,
    params: Optional[dict] = None,
    return_params: bool = False,
    show: bool = True,
    plot: bool = True,
    return_df: bool = True,
    return_fig: bool = False,
    decimals: int = 4,
    df_name: Optional[str] = None,
    fig_width: float = 13.0,
    fig_height: float = 5.0,
    dpi: int = 110,
):
    """Fit an instant clustering baseline in one line (unsupervised; no target).

    Mirrors :func:`regress` / :func:`classify` (fit / apply / compare, hybrid
    artifact) but takes **no** ``y`` -- clustering finds the data's own
    structure. In FIT mode the cluster label of each row is appended as
    ``"cluster"``; if ``k`` is ``None`` the number of clusters is chosen
    automatically by maximising the silhouette over ``k_range``. APPLY mode
    assigns clusters to new data with the saved estimator (no re-fit); COMPARE
    mode evaluates every candidate over ``k_range`` and writes nothing.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data. Never mutated.
    cols : sequence of str, optional
        Feature columns. Defaults to every numeric (non-boolean) column.
        Ignored in apply mode.
    method : {'kmeans', 'agglomerative', 'compare'}
        The baseline clusterer, or 'compare' to rank them all.
    k : int, optional
        Fixed number of clusters. If ``None`` (default) k is selected by
        silhouette over ``k_range``.
    k_range : (int, int), default (2, 10)
        Inclusive search range for k when ``k`` is ``None`` (upper bound is
        clamped to ``n_rows - 1``).
    standardize : bool, optional
        Whether to z-score features before clustering. Default ``True`` (both
        algorithms are distance-based).
    params, return_params, show, plot, return_df, return_fig, decimals, df_name
        Standard dextra flags (see :func:`regress`).

    Returns
    -------
    pandas.DataFrame
        The input frame plus a ``"cluster"`` label column, and -- when
        requested -- the hybrid ``params`` artifact and/or the figure.

    Examples
    --------
    >>> out, p = dx.cluster(df, method='kmeans', return_params=True)   # auto-k
    >>> labels = dx.cluster(df_new, params=p)            # apply, no re-fit
    >>> dx.cluster(df, method='compare')                 # rank baselines
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"'df' must be a pandas DataFrame, got {type(df).__name__}")
    if df_name is None:
        df_name = get_variable_name(df, depth=2)

    # ---- APPLY MODE -----------------------------------------------------
    if params is not None:
        return _cluster_apply(df, params, show, plot, return_df,
                              return_params, return_fig, decimals, df_name,
                              fig_width, fig_height, dpi)

    # ---- FIT / COMPARE guards (before importing sklearn) ----------------
    if method not in _VALID_CLUSTER_METHODS:
        raise ValueError(
            f"cluster: 'method' must be one of {_VALID_CLUSTER_METHODS}, "
            f"got {method!r}")
    if method == "compare" and return_params:
        raise ValueError(
            "cluster: method='compare' fits no single model and writes no "
            "artifact; call with a concrete method to get params.")
    if k is not None and int(k) < 2:
        raise ValueError(f"cluster: 'k' must be >= 2, got {k}")

    features = _resolve_features(df, cols, "cluster", exclude=[])

    _require_sklearn("cluster")
    X = _clean_x(df, features, "cluster")
    if k is not None and int(k) > len(X) - 1:
        raise ValueError(
            f"cluster: k={k} too large for {len(X)} usable rows (need "
            f"k <= n - 1).")
    Xa = X.to_numpy(dtype=float)

    use_scaler = True if standardize is None else bool(standardize)
    scaler = None
    if use_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(Xa)
    else:
        Xs = Xa

    k_grid = [int(k)] if k is not None else _k_grid(k_range, len(X))

    if method == "compare":
        return _cluster_compare(df, Xs, scaler, features, k_grid, use_scaler,
                                show, plot, return_df, return_fig, decimals,
                                df_name, fig_width, fig_height, dpi)

    k_selected_by = "user" if k is not None else "silhouette"
    best_k, labels, inertia, predictor, curve = _select_k(method, Xs, k_grid)
    sil = curve[best_k]["silhouette"]
    estimator = _assemble_cluster_pipeline(scaler, predictor)

    out = df.copy()
    out.attrs = dict(df.attrs)
    pred_col = "cluster"
    full_X = (df[features].apply(pd.to_numeric, errors="coerce")
              .fillna(X.mean()).to_numpy(dtype=float))
    out[pred_col] = pd.Series(estimator.predict(full_X), index=df.index,
                              name=pred_col)

    metrics = {"fit": {"silhouette": sil, "inertia": inertia,
                       "n_clusters": int(best_k)}}
    params_out = {
        "function": "cluster",
        "task": "clustering",
        "algorithm": method,
        "version": __version__,
        "fit_at": _now_iso(),
        "features": list(features),
        "target": None,
        "hyperparams": _cluster_hyperparams(method, best_k),
        "metrics": metrics,
        "pred_col": pred_col,
        "metadata": {"n": int(len(X)), "n_features": len(features),
                     "k": int(best_k), "standardize": bool(use_scaler),
                     "k_range": ([k_grid[0], k_grid[-1]] if k is None else None),
                     "k_selected_by": k_selected_by},
        "estimator": estimator,
    }

    inertia_txt = ("" if inertia is None
                   else f", inertia={_fmt_metric(inertia, decimals)}")
    decision = (
        f"'{method}' clustering baseline -- {best_k} clusters "
        f"(k by {k_selected_by}); silhouette={_fmt_metric(sil, decimals)}"
        f"{inertia_txt}. Cluster labels added as '{pred_col}'. "
        f"Persist with joblib.dump(params['estimator'], ...).")

    _append_audit(out, {
        "stage": "modeling",
        "function": "cluster",
        "timestamp": params_out["fit_at"],
        "mode": "fit",
        "params": {"method": method, "features": list(features),
                   "k": int(best_k), "k_selected_by": k_selected_by},
        "decision": decision,
    })

    summary = _metrics_table(params_out["metrics"])
    if show:
        _print_header(f"Clustering baseline for: {df_name}  "
                      f"(method={method}, k={best_k}, mode=fit)")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")

    fig = None
    if plot:
        fig = _plot_cluster(Xs, labels, curve, best_k, method, features,
                            fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params_out, fig, return_df, return_params, return_fig)


def _cluster_apply(df, params, show, plot, return_df, return_params,
                   return_fig, decimals, df_name, fig_width, fig_height, dpi):
    if not isinstance(params, dict) or params.get("function") != "cluster":
        got = (params.get("function") if isinstance(params, dict)
               else type(params).__name__)
        raise ValueError(f"params dict is not for 'cluster' (function={got!r}).")
    estimator = params.get("estimator")
    if estimator is None or not hasattr(estimator, "predict"):
        raise ValueError(
            "cluster apply: params has no fitted estimator. Re-fit and pass the "
            "full params (the artifact carries params['estimator']).")
    features = params["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(
            f"cluster apply failed: params expects feature column(s) {missing} "
            f"which are not present in this DataFrame.")
    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        raise ValueError(
            "cluster apply: feature columns contain NaN/non-numeric values. "
            "Clean / impute (Phase 3) before predicting.")
    pred_col = params.get("pred_col", "cluster")
    out = df.copy()
    out.attrs = dict(df.attrs)
    out[pred_col] = pd.Series(estimator.predict(X.to_numpy(dtype=float)),
                              index=df.index, name=pred_col)

    decision = (f"Applied saved '{params.get('algorithm', '?')}' clustering "
                f"model (fitted {params.get('fit_at', '?')}); assigned "
                f"{len(out)} row(s) into '{pred_col}' -- no re-fit, "
                f"leakage-safe.")
    _append_audit(out, {
        "stage": "modeling",
        "function": "cluster",
        "timestamp": _now_iso(),
        "mode": "apply",
        "params": {"algorithm": params.get("algorithm"),
                   "fit_at": params.get("fit_at")},
        "decision": decision,
    })
    if show:
        _print_header(f"Cluster assignment for: {df_name}  "
                      f"(algorithm={params.get('algorithm')}, mode=apply)")
        _display(_fmt_table(_metrics_table(params["metrics"]), decimals))
        print(f"\nDecision: {decision}\n")
    fig = None
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, params, fig, return_df, return_params, return_fig)


def _cluster_compare(df, Xs, scaler, features, k_grid, use_scaler, show, plot,
                     return_df, return_fig, decimals, df_name,
                     fig_width, fig_height, dpi):
    rows = {}
    curves = {}
    for m in _CLUSTER_METHODS:
        best_k, _labels, inertia, _pred, curve = _select_k(m, Xs, k_grid)
        rows[m] = {"silhouette": curve[best_k]["silhouette"],
                   "n_clusters": int(best_k), "inertia": inertia}
        curves[m] = curve
    summary = pd.DataFrame(
        {"silhouette": {m: rows[m]["silhouette"] for m in _CLUSTER_METHODS},
         "n_clusters": {m: rows[m]["n_clusters"] for m in _CLUSTER_METHODS},
         "inertia": {m: rows[m]["inertia"] for m in _CLUSTER_METHODS}})
    summary = summary.sort_values(
        "silhouette", ascending=False, na_position="last")
    best = summary.index[0]
    best_sil = summary.loc[best, "silhouette"]
    best_k = summary.loc[best, "n_clusters"]

    out = df.copy()
    out.attrs = dict(df.attrs)
    krange_txt = f"[{k_grid[0]}, {k_grid[-1]}]"
    decision = (
        f"Compared {len(_CLUSTER_METHODS)} clustering algorithms over k in "
        f"{krange_txt}; best by silhouette is '{best}' "
        f"(silhouette={_fmt_metric(best_sil, decimals)}, k={int(best_k)}). "
        f"Nothing written -- choose a method to fit.")
    _append_audit(out, {
        "stage": "modeling",
        "function": "cluster",
        "timestamp": _now_iso(),
        "mode": "compare",
        "params": {"candidates": list(_CLUSTER_METHODS),
                   "k_range": [k_grid[0], k_grid[-1]]},
        "decision": decision,
    })
    if show:
        _print_header(f"Clustering model comparison for: {df_name}  "
                      f"(k in {krange_txt})")
        _display(_fmt_table(summary, decimals))
        print(f"\nDecision: {decision}\n")
    fig = None
    if plot:
        fig = _plot_cluster_compare(summary, curves, fig_width, fig_height, dpi)
    _finalize_figure(fig, return_fig)
    return _ret_pack(out, None, fig, return_df, False, return_fig)


# ---------------------------------------------------------------------------
# Cluster plots
# ---------------------------------------------------------------------------

def _project_2d(Xs):
    """A 2-D view of the scaled features for the scatter (PCA if > 2 dims)."""
    Xs = np.asarray(Xs, dtype=float)
    if Xs.shape[1] == 1:
        return np.column_stack([Xs[:, 0], np.zeros(len(Xs))]), ("feature 1", "")
    if Xs.shape[1] == 2:
        return Xs[:, :2], ("feature 1", "feature 2")
    from sklearn.decomposition import PCA
    proj = PCA(n_components=2, random_state=0).fit_transform(Xs)
    return proj, ("PC 1", "PC 2")


def _plot_cluster(Xs, labels, curve, best_k, method, features,
                  fig_width, fig_height, dpi):
    proj, (xlab, ylab) = _project_2d(Xs)
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle(f"Clustering diagnostics -- {method} (k={best_k})",
                 fontsize=13, fontweight="bold")
    # (1) cluster scatter
    ax = axes[0]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", s=20,
                    alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(f"Clusters ({len(np.unique(labels))} found)")
    try:
        fig.colorbar(sc, ax=ax, label="cluster", fraction=0.046, pad=0.04)
    except Exception:
        pass
    # (2) silhouette (and inertia/elbow for kmeans) vs k
    ax2 = axes[1]
    ks = sorted(curve.keys())
    sil = [curve[k]["silhouette"] if curve[k]["silhouette"] is not None
           else np.nan for k in ks]
    ax2.plot(ks, sil, marker="o", color="#2E75B6", label="silhouette")
    ax2.axvline(best_k, color="#d62728", linestyle="--", linewidth=1.4,
                label=f"selected k={best_k}")
    ax2.set_xlabel("k (number of clusters)")
    ax2.set_ylabel("silhouette")
    ax2.set_title("Model selection across k")
    inertia = [curve[k]["inertia"] for k in ks]
    if any(v is not None for v in inertia):
        ax3 = ax2.twinx()
        iv = [v if v is not None else np.nan for v in inertia]
        ax3.plot(ks, iv, marker="s", color="#ff7f0e", alpha=0.7,
                 label="inertia (elbow)")
        ax3.set_ylabel("inertia")
    ax2.legend(loc="best", fontsize=9)
    return fig


def _plot_cluster_compare(summary, curves, fig_width, fig_height, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), dpi=dpi)
    fig.suptitle("Clustering baselines -- comparison",
                 fontsize=13, fontweight="bold")
    order = list(summary.index)
    sil = [summary.loc[m, "silhouette"] if summary.loc[m, "silhouette"]
           is not None else 0.0 for m in order]
    axes[0].barh(order, sil, color="#2E75B6")
    axes[0].set_xlabel("silhouette (higher is better)")
    axes[0].invert_yaxis()
    axes[0].set_title("Best silhouette by algorithm")
    for m in order:
        curve = curves[m]
        ks = sorted(curve.keys())
        sv = [curve[k]["silhouette"] if curve[k]["silhouette"] is not None
              else np.nan for k in ks]
        axes[1].plot(ks, sv, marker="o", label=m)
    axes[1].set_xlabel("k (number of clusters)")
    axes[1].set_ylabel("silhouette")
    axes[1].set_title("Silhouette across k")
    axes[1].legend(loc="best", fontsize=9)
    return fig


# Short alias (consistent with the rest of dextra).
clus = cluster
