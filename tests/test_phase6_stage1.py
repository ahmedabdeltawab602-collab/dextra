"""dextra Phase 6 Stage 6.1 -- regress (regression baseline).

Covers the fit / apply / compare contract, the hybrid artifact (JSON-safe
descriptor + fitted estimator), immutability, idempotency under apply,
guard errors, the audit trail, and every candidate algorithm.

scikit-learn is required to fit a model, so the whole module is skipped if it
is not installed (CI installs the 'ml' extra).
"""
from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)
pytest.importorskip("sklearn")

import dextra as dx

KW = dict(show=False, plot=False)


@pytest.fixture
def reg_df():
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "x3": rng.uniform(-1, 1, n),
    })
    df["price"] = 3.0 * df["x1"] - 2.0 * df["x2"] + 0.5 * df["x3"] \
        + rng.normal(0, 0.4, n)
    return df


def _descriptor(params):
    return {k: v for k, v in params.items() if k != "estimator"}


def test_exports_and_alias():
    assert hasattr(dx, "regress") and hasattr(dx, "reg")
    assert dx.reg is dx.regress


@pytest.mark.parametrize("method", ["linear", "ridge", "lasso", "tree", "forest"])
def test_fit_each_method(reg_df, method):
    out, params = dx.regress(reg_df, y="price", method=method,
                             return_params=True, **KW)
    # prediction column appended, original untouched
    assert "price_pred" in out.columns
    assert "price_pred" not in reg_df.columns
    assert len(out) == len(reg_df)
    # hybrid artifact shape
    assert params["function"] == "regress"
    assert params["task"] == "regression"
    assert params["algorithm"] == method
    assert params["features"] == ["x1", "x2", "x3"]
    assert params["target"] == "price"
    assert "train" in params["metrics"] and "cv" in params["metrics"]
    # descriptor (without estimator) is strict-JSON-serialisable
    json.loads(json.dumps(_descriptor(params)))
    # the fitted estimator is exposed and usable
    assert hasattr(params["estimator"], "predict")


def test_fit_audit_entry(reg_df):
    out = dx.regress(reg_df, y="price", method="linear", **KW)
    audit = out.attrs.get("dextra_audit", [])
    assert audit and audit[-1]["stage"] == "modeling"
    assert audit[-1]["mode"] == "fit"
    assert audit[-1]["function"] == "regress"


def test_apply_reproduces_and_is_idempotent(reg_df):
    train = reg_df.iloc[:150]
    test = reg_df.iloc[150:].copy()
    _, params = dx.regress(train, y="price", method="forest",
                           return_params=True, **KW)
    p1 = dx.regress(test, params=params, **KW)
    # apply does not need y and leaves original untouched
    assert "price_pred" in p1.columns
    assert "price_pred" not in test.columns
    # idempotent: applying again gives identical predictions
    p2 = dx.regress(p1.drop(columns=["price_pred"]), params=params, **KW)
    np.testing.assert_allclose(p1["price_pred"].to_numpy(),
                               p2["price_pred"].to_numpy())
    # apply audit entry
    assert p1.attrs["dextra_audit"][-1]["mode"] == "apply"


def test_apply_missing_feature_raises(reg_df):
    _, params = dx.regress(reg_df, y="price", method="linear",
                           return_params=True, **KW)
    with pytest.raises(KeyError):
        dx.regress(reg_df.drop(columns=["x2"]), params=params, **KW)


def test_apply_wrong_params_raises(reg_df):
    with pytest.raises(ValueError):
        dx.regress(reg_df, params={"function": "scale"}, **KW)


def test_compare_writes_nothing(reg_df):
    out = dx.regress(reg_df, y="price", method="compare", **KW)
    assert "price_pred" not in out.columns          # nothing written
    assert list(out.columns) == list(reg_df.columns)
    assert out.attrs["dextra_audit"][-1]["mode"] == "compare"


def test_compare_with_return_params_raises(reg_df):
    with pytest.raises(ValueError):
        dx.regress(reg_df, y="price", method="compare",
                   return_params=True, **KW)


def test_non_numeric_target_raises(reg_df):
    df = reg_df.copy()
    df["grade"] = np.random.default_rng(0).choice(list("abc"), len(df))
    with pytest.raises(ValueError):
        dx.regress(df, y="grade", cols=["x1", "x2"], **KW)


def test_bad_method_raises(reg_df):
    with pytest.raises(ValueError):
        dx.regress(reg_df, y="price", method="svm", **KW)


def test_standardize_override(reg_df):
    _, params = dx.regress(reg_df, y="price", method="forest",
                           standardize=True, return_params=True, **KW)
    assert params["metadata"]["standardize"] is True


def test_plot_returns_figure(reg_df):
    fig = dx.regress(reg_df, y="price", method="linear",
                     show=False, plot=True, return_fig=True, return_df=False)
    assert fig is not None


# ---------------------------------------------------------------------------
# Edge cases (audit before locking 6.1 as the 6.2 baseline)
# ---------------------------------------------------------------------------

def test_constant_feature_does_not_crash(reg_df):
    df = reg_df.copy()
    df["const"] = 5.0                      # zero-variance feature
    out, params = dx.regress(df, y="price", cols=["x1", "x2", "const"],
                             method="ridge", return_params=True, **KW)
    assert "price_pred" in out.columns
    assert "const" in params["features"]


def test_constant_target_is_safe(reg_df):
    # Degenerate target: R^2 is undefined -> metrics may be None, but the call
    # must not raise (the Decision sentence formats None as 'n/a').
    df = reg_df.copy()
    df["flat"] = 3.0
    out, params = dx.regress(df, y="flat", cols=["x1", "x2"], method="linear",
                             show=True, plot=False, return_params=True)
    assert "flat_pred" in out.columns
    assert "train" in params["metrics"] and "cv" in params["metrics"]


def test_nan_in_target_rows_dropped(reg_df):
    df = reg_df.copy()
    df.loc[df.index[:10], "price"] = np.nan
    out, params = dx.regress(df, y="price", cols=["x1", "x2"], method="linear",
                             return_params=True, **KW)
    # rows with a NaN target are dropped from training
    assert params["metadata"]["n_train"] == len(df) - 10
    # but every original row still gets an (in-sample) prediction column
    assert len(out) == len(df)


def test_all_nan_feature_raises(reg_df):
    df = reg_df.copy()
    df["bad"] = np.nan
    with pytest.raises(ValueError):
        dx.regress(df, y="price", cols=["bad"], method="linear", **KW)


def test_partial_nan_feature_imputed_in_fit(reg_df):
    df = reg_df.copy()
    df.loc[df.index[:5], "x1"] = np.nan
    out, params = dx.regress(df, y="price", cols=["x1", "x2"], method="linear",
                             return_params=True, **KW)
    # NaN rows excluded from the fit ...
    assert params["metadata"]["n_train"] == len(df) - 5
    # ... yet the in-sample prediction column is complete (NaN features imputed)
    assert out["price_pred"].notna().all()


def test_single_feature(reg_df):
    out = dx.regress(reg_df, y="price", cols=["x1"], method="forest", **KW)
    assert "price_pred" in out.columns


def test_skewed_outlier_dominated_target(reg_df):
    # Regression analogue of "severe imbalance": a target dominated by one value
    # with rare extreme outliers. Must run without crashing.
    df = reg_df.copy()
    y = np.zeros(len(df))
    y[:3] = 1e6
    df["rare"] = y
    out = dx.regress(df, y="rare", cols=["x1", "x2"], method="forest", **KW)
    assert "rare_pred" in out.columns


# ---------------------------------------------------------------------------
# No hidden coupling: the shared renderer is family-agnostic
# ---------------------------------------------------------------------------

def test_metrics_table_is_family_agnostic():
    from dextra.modeling import _metrics_table
    # regression / classification shape: {'train': {...}, 'cv': {...}}
    reg = _metrics_table({"train": {"r2": 0.9, "rmse": 1.2, "mae": 0.8},
                          "cv": {"r2": 0.85, "rmse": 1.4, "mae": 0.9}})
    assert list(reg.columns) == ["train", "cv"]
    assert set(reg.index) == {"R2", "RMSE", "MAE"}
    # classification metrics (different keys, same renderer, no code change)
    clf = _metrics_table({"train": {"accuracy": 0.95, "f1": 0.94, "roc_auc": 0.98},
                          "cv": {"accuracy": 0.91, "f1": 0.90, "roc_auc": 0.95}})
    assert set(clf.index) == {"ACCURACY", "F1", "ROC_AUC"}
    # clustering shape: a single 'fit' split
    clu = _metrics_table({"fit": {"silhouette": 0.55, "n_clusters": 3}})
    assert list(clu.columns) == ["fit"]
    assert set(clu.index) == {"SILHOUETTE", "N_CLUSTERS"}


def test_fmt_metric_handles_none_and_nan():
    from dextra.modeling import _fmt_metric
    assert _fmt_metric(None, 4) == "n/a"
    assert _fmt_metric(float("nan"), 4) == "n/a"
    assert _fmt_metric(0.1234, 2) == "0.12"


# ---------------------------------------------------------------------------
# Presentation unification: every metric display path is NaN/None safe
# ---------------------------------------------------------------------------

def test_degenerate_target_all_display_paths_safe(capsys):
    # A constant target makes R^2 undefined (NaN on real sklearn). Every
    # presentation path -- fit show, compare show, apply show -- must render
    # via the two sanctioned formatters and never raise.
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": rng.normal(0, 1, n)})
    df["flat"] = 7.0
    # fit (show=True)
    out, params = dx.regress(df, y="flat", cols=["x1", "x2"], method="linear",
                             show=True, plot=False, return_params=True)
    # compare (show=True)
    dx.regress(df, y="flat", cols=["x1", "x2"], method="compare",
               show=True, plot=False)
    # apply (show=True) reuses the same artifact
    dx.regress(df, params=params, show=True, plot=False)
    text = capsys.readouterr().out
    # missing metrics surface as the sanctioned sentinels, never a raw error
    assert ("n/a" in text) or ("-" in text)


def test_fmt_table_renders_missing_as_dash():
    from dextra.modeling import _fmt_table, _metrics_table
    tbl = _metrics_table({"train": {"r2": None, "rmse": 1.0, "mae": None},
                          "cv": {"r2": float("nan"), "rmse": 1.1, "mae": 0.9}})
    rendered = _fmt_table(tbl, 4)
    # both None and NaN collapse to the table sentinel "-"
    assert rendered.loc["R2", "train"] == "-"
    assert rendered.loc["R2", "cv"] == "-"
    assert rendered.loc["MAE", "train"] == "-"
    # real values still format normally
    assert rendered.loc["RMSE", "train"] == "1.0000"
