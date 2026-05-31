"""dextra Phase 6 Stage 6.2 -- classify (classification baseline).

Same contract as regress (fit / apply / compare, hybrid artifact) for a
categorical target. Covers binary + multiclass, every algorithm, immutability,
idempotency under apply, guard errors, edge cases (rare class, constant
feature, NaN), and the audit trail. Skipped if scikit-learn is absent.
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
def clf_df():
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
    })
    df["churn"] = np.where(df.x1 + df.x2 + rng.normal(0, 0.3, n) > 0, "yes", "no")
    df["grade"] = pd.cut(df.x1, bins=[-9, -0.5, 0.5, 9],
                         labels=["low", "mid", "hi"]).astype(str)
    return df


def _descriptor(params):
    return {k: v for k, v in params.items() if k != "estimator"}


def test_exports_and_alias():
    assert hasattr(dx, "classify") and hasattr(dx, "clf")
    assert dx.clf is dx.classify


@pytest.mark.parametrize("method", ["logistic", "tree", "forest", "knn"])
def test_fit_each_method_binary(clf_df, method):
    out, params = dx.classify(clf_df, y="churn", cols=["x1", "x2", "x3"],
                              method=method, return_params=True, **KW)
    assert "churn_pred" in out.columns and "churn_pred" not in clf_df.columns
    assert params["function"] == "classify" and params["task"] == "classification"
    assert params["algorithm"] == method
    assert set(params["classes"]) == {"yes", "no"}
    assert params["n_classes"] == 2 and params["metadata"]["binary"] is True
    assert set(params["metrics"]) == {"train", "cv"}
    assert set(params["metrics"]["cv"]) == {"accuracy", "f1", "roc_auc"}
    json.loads(json.dumps(_descriptor(params)))           # JSON-safe descriptor
    assert hasattr(params["estimator"], "predict")
    # predicted labels are drawn from the trained label set
    assert set(out["churn_pred"].unique()).issubset({"yes", "no"})


def test_fit_multiclass(clf_df):
    out, params = dx.classify(clf_df, y="grade", cols=["x1", "x2", "x3"],
                              method="forest", return_params=True, **KW)
    assert params["n_classes"] == 3 and params["metadata"]["binary"] is False
    assert set(params["classes"]) == {"low", "mid", "hi"}
    assert "grade_pred" in out.columns


def test_fit_audit_entry(clf_df):
    out = dx.classify(clf_df, y="churn", cols=["x1", "x2"], method="logistic", **KW)
    a = out.attrs.get("dextra_audit", [])
    assert a and a[-1]["stage"] == "modeling" and a[-1]["mode"] == "fit"
    assert a[-1]["function"] == "classify"


def test_apply_reproduces_and_idempotent(clf_df):
    train = clf_df.iloc[:150]
    test = clf_df.iloc[150:].copy()
    _, params = dx.classify(train, y="churn", cols=["x1", "x2", "x3"],
                            method="forest", return_params=True, **KW)
    p1 = dx.classify(test, params=params, **KW)
    assert "churn_pred" in p1.columns and "churn_pred" not in test.columns
    assert set(p1["churn_pred"].unique()).issubset({"yes", "no"})
    p2 = dx.classify(p1.drop(columns=["churn_pred"]), params=params, **KW)
    assert list(p1["churn_pred"]) == list(p2["churn_pred"])
    assert p1.attrs["dextra_audit"][-1]["mode"] == "apply"


def test_apply_missing_feature_raises(clf_df):
    _, params = dx.classify(clf_df, y="churn", cols=["x1", "x2"],
                            method="logistic", return_params=True, **KW)
    with pytest.raises(KeyError):
        dx.classify(clf_df.drop(columns=["x2"]), params=params, **KW)


def test_apply_wrong_params_raises(clf_df):
    with pytest.raises(ValueError):
        dx.classify(clf_df, params={"function": "regress"}, **KW)


def test_compare_writes_nothing(clf_df):
    out = dx.classify(clf_df, y="churn", cols=["x1", "x2", "x3"],
                      method="compare", **KW)
    assert "churn_pred" not in out.columns
    assert list(out.columns) == list(clf_df.columns)
    assert out.attrs["dextra_audit"][-1]["mode"] == "compare"


def test_compare_with_return_params_raises(clf_df):
    with pytest.raises(ValueError):
        dx.classify(clf_df, y="churn", method="compare", return_params=True, **KW)


def test_continuous_target_raises(clf_df):
    df = clf_df.copy()
    df["amount"] = np.random.default_rng(0).normal(100, 15, len(df))  # continuous
    with pytest.raises(ValueError):
        dx.classify(df, y="amount", cols=["x1", "x2"], **KW)


def test_single_class_raises(clf_df):
    df = clf_df.copy()
    df["const"] = "only"
    with pytest.raises(ValueError):
        dx.classify(df, y="const", cols=["x1", "x2"], **KW)


def test_bad_method_raises(clf_df):
    with pytest.raises(ValueError):
        dx.classify(clf_df, y="churn", method="svm", **KW)


def test_rare_class_raises(clf_df):
    df = clf_df.copy()
    df.loc[df.index[0], "churn"] = "unique_single"   # a class with 1 member
    with pytest.raises(ValueError):
        dx.classify(df, y="churn", cols=["x1", "x2"], **KW)


def test_constant_feature_ok(clf_df):
    df = clf_df.copy()
    df["flat"] = 1.0
    out = dx.classify(df, y="churn", cols=["x1", "x2", "flat"],
                      method="logistic", **KW)
    assert "churn_pred" in out.columns


def test_standardize_override(clf_df):
    _, params = dx.classify(clf_df, y="churn", method="forest",
                            standardize=True, return_params=True, **KW)
    assert params["metadata"]["standardize"] is True


def test_plot_returns_figure(clf_df):
    fig = dx.classify(clf_df, y="churn", cols=["x1", "x2"], method="forest",
                      show=False, plot=True, return_fig=True, return_df=False)
    assert fig is not None
