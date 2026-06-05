"""dextra Phase 7 -- evaluation (confusion_report / roc_pr / residual_analysis /
learning_curves).

Covers both input modes (label and artifact), binary + multiclass, the report
descriptor (JSON-safe, no estimator), figure rendering, immutability of the
input DataFrame, alias identity, and the guard-error paths. Skipped if
scikit-learn is absent.
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
KWP = dict(show=False, plot=True, return_fig=True)


@pytest.fixture
def clf_df():
    rng = np.random.default_rng(7)
    n = 150
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
    })
    df["churn"] = np.where(df.x1 + df.x2 + rng.normal(0, 0.3, n) > 0,
                           "yes", "no")
    df["grade"] = pd.cut(df.x1, bins=[-9, -0.5, 0.5, 9],
                         labels=["low", "mid", "hi"]).astype(str)
    return df


@pytest.fixture
def reg_df():
    rng = np.random.default_rng(11)
    n = 150
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(0, 1, n),
        "x3": rng.normal(0, 1, n),
    })
    df["price"] = 3 * df.x1 - 2 * df.x2 + rng.normal(0, 0.5, n)
    return df


def _json_ok(report):
    json.loads(json.dumps(report))


# ---------------------------------------------------------------------------
# aliases / exports
# ---------------------------------------------------------------------------

def test_exports_and_aliases():
    assert dx.confrep is dx.confusion_report
    assert dx.rocpr is dx.roc_pr
    assert dx.residan is dx.residual_analysis
    assert dx.learncv is dx.learning_curves


# ---------------------------------------------------------------------------
# confusion_report
# ---------------------------------------------------------------------------

def test_confusion_label_mode_binary(clf_df):
    rng = np.random.default_rng(1)
    df = clf_df.copy()
    flip = rng.random(len(df)) < 0.2
    df["pred"] = np.where(flip, np.where(df.churn == "yes", "no", "yes"),
                          df.churn)
    cols0 = list(df.columns)
    table, report = dx.confusion_report(df, y_true="churn", y_pred="pred",
                                        return_params=True, **KW)
    assert "accuracy" in table.index and "weighted avg" in table.index
    assert list(table.columns) == ["precision", "recall", "f1", "support"]
    assert report["function"] == "confusion_report"
    assert report["task"] == "classification"
    assert report["metadata"]["input_mode"] == "label"
    assert set(report["metrics"]["per_class"]) == {"yes", "no"}
    _json_ok(report)
    # immutability
    assert list(df.columns) == cols0
    assert "dextra_audit" not in df.attrs


def test_confusion_label_mode_multiclass_plots(clf_df):
    df = clf_df.copy()
    df["pred"] = df["grade"]
    table, report, fig = dx.confusion_report(df, y_true="grade", y_pred="pred",
                                             return_params=True, **KWP)
    assert fig is not None
    # perfect predictions -> accuracy 1.0
    assert report["metrics"]["overall"]["accuracy"] == pytest.approx(1.0)
    assert len(report["metrics"]["labels"]) == 3
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_confusion_artifact_mode(clf_df):
    _, p = dx.classify(clf_df, y="churn", cols=["x1", "x2", "x3"],
                       method="forest", return_params=True, **KW)
    table, report = dx.confusion_report(clf_df, params=p, return_params=True,
                                        **KW)
    assert report["metadata"]["input_mode"] == "artifact"
    assert report["target"] == "churn"
    _json_ok(report)


def test_confusion_custom_labels(clf_df):
    df = clf_df.copy()
    df["pred"] = df["churn"]
    table, _ = dx.confusion_report(df, y_true="churn", y_pred="pred",
                                   labels=["yes", "no"], return_params=True,
                                   **KW)
    assert list(table.index)[:2] == ["yes", "no"]


def test_confusion_errors(clf_df):
    with pytest.raises(ValueError):
        dx.confusion_report(clf_df, **KW)                 # no y_true / params
    with pytest.raises(KeyError):
        dx.confusion_report(clf_df, y_true="nope", y_pred="churn", **KW)
    # a clustering-style artifact (target None) must be rejected
    with pytest.raises(ValueError):
        dx.confusion_report(clf_df, params={"target": None}, **KW)


# ---------------------------------------------------------------------------
# residual_analysis
# ---------------------------------------------------------------------------

def test_residual_label_mode(reg_df):
    rng = np.random.default_rng(2)
    df = reg_df.copy()
    df["pred"] = df["price"] + rng.normal(0, 0.5, len(df))
    cols0 = list(df.columns)
    table, report = dx.residual_analysis(df, y_true="price", y_pred="pred",
                                         return_params=True, **KW)
    for k in ["n", "resid_mean", "rmse", "mae", "durbin_watson",
              "jarque_bera_p"]:
        assert k in table.index
    assert report["function"] == "residual_analysis"
    assert report["metrics"]["rmse"] is not None
    _json_ok(report)
    assert list(df.columns) == cols0
    assert "dextra_audit" not in df.attrs


def test_residual_artifact_mode_plots(reg_df):
    _, p = dx.regress(reg_df, y="price", cols=["x1", "x2", "x3"],
                      method="linear", return_params=True, **KW)
    table, report, fig = dx.residual_analysis(reg_df, params=p,
                                              return_params=True, **KWP)
    assert fig is not None
    assert report["metadata"]["input_mode"] == "artifact"
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_residual_too_few_rows():
    df = pd.DataFrame({"y": [1.0, 2.0], "p": [1.1, 1.9]})
    with pytest.raises(ValueError):
        dx.residual_analysis(df, y_true="y", y_pred="p", **KW)


# ---------------------------------------------------------------------------
# roc_pr
# ---------------------------------------------------------------------------

def test_roc_pr_binary_label_scores(clf_df):
    from sklearn.linear_model import LogisticRegression
    X = clf_df[["x1", "x2", "x3"]].to_numpy(float)
    y = clf_df["churn"].to_numpy()
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    pos_idx = list(lr.classes_).index("yes")
    proba = lr.predict_proba(X)[:, pos_idx]
    table, report, fig = dx.roc_pr(clf_df, y_true="churn", scores=proba,
                                   pos_label="yes", return_params=True, **KWP)
    assert report["metadata"]["binary"] is True
    assert report["metrics"]["macro_roc_auc"] is not None
    assert any(str(i).startswith("positive") for i in table.index)
    _json_ok(report)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_roc_pr_binary_scores_as_column(clf_df):
    from sklearn.linear_model import LogisticRegression
    X = clf_df[["x1", "x2", "x3"]].to_numpy(float)
    y = clf_df["churn"].to_numpy()
    lr = LogisticRegression(max_iter=2000).fit(X, y)
    df = clf_df.copy()
    df["proba_yes"] = lr.predict_proba(X)[:, list(lr.classes_).index("yes")]
    table, report = dx.roc_pr(df, y_true="churn", scores="proba_yes",
                              pos_label="yes", return_params=True, **KW)
    assert report["metrics"]["macro_avg_precision"] is not None


def test_roc_pr_multiclass_artifact_plots(clf_df):
    _, p = dx.classify(clf_df, y="grade", cols=["x1", "x2", "x3"],
                       method="forest", return_params=True, **KW)
    table, report, fig = dx.roc_pr(clf_df, params=p, return_params=True, **KWP)
    assert report["metadata"]["binary"] is False
    assert report["metadata"]["n_classes"] == 3
    assert "macro" in table.index
    _json_ok(report)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_roc_pr_missing_scores(clf_df):
    with pytest.raises(ValueError):
        dx.roc_pr(clf_df, y_true="churn", **KW)           # no scores / params


# ---------------------------------------------------------------------------
# learning_curves
# ---------------------------------------------------------------------------

def test_learning_curves_artifact_classification_plots(clf_df):
    _, p = dx.classify(clf_df, y="churn", cols=["x1", "x2", "x3"],
                       method="forest", return_params=True, **KW)
    table, report, fig = dx.learning_curves(clf_df, params=p,
                                            return_params=True, **KWP)
    assert list(table.columns) == ["train_mean", "train_std",
                                   "cv_mean", "cv_std"]
    assert report["metrics"]["scoring"] == "accuracy"
    assert report["metadata"]["input_mode"] == "artifact"
    assert report["metrics"]["final_gap"] is not None
    _json_ok(report)
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_learning_curves_artifact_regression(reg_df):
    _, p = dx.regress(reg_df, y="price", cols=["x1", "x2", "x3"],
                      method="forest", return_params=True, **KW)
    table, report = dx.learning_curves(reg_df, params=p, return_params=True,
                                       **KW)
    assert report["metrics"]["scoring"] == "r2"
    assert report["task"] == "regression"


def test_learning_curves_estimator_mode_infers_task(clf_df, reg_df):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    _, rc = dx.learning_curves(
        clf_df, y="churn", cols=["x1", "x2", "x3"],
        estimator=RandomForestClassifier(n_estimators=50, random_state=0),
        return_params=True, **KW)
    assert rc["task"] == "classification"
    _, rr = dx.learning_curves(
        reg_df, y="price", cols=["x1", "x2", "x3"],
        estimator=RandomForestRegressor(n_estimators=50, random_state=0),
        return_params=True, **KW)
    assert rr["task"] == "regression"


def test_learning_curves_errors(clf_df):
    with pytest.raises(ValueError):
        dx.learning_curves(clf_df, y="churn", cols=["x1", "x2", "x3"], **KW)
    with pytest.raises(ValueError):
        dx.learning_curves(clf_df.head(4), y="churn", cols=["x1"],
                           estimator=object(), **KW)
