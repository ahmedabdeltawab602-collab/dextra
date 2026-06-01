"""Hardening sprint -- scikit-learn compatibility wrappers (dextra.compat).

Verifies the wrappers honour the sklearn estimator API: get_params/set_params,
clone-ability, fit/transform/predict, predict_proba + classes_ for the
classifier, labels_ for the clusterer, and end-to-end use inside an
sklearn.pipeline.Pipeline. Skipped if scikit-learn is absent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

import dextra as dx  # noqa: E402
from dextra.compat import (  # noqa: E402
    DextraClassifier,
    DextraClusterer,
    DextraFeaturePipeline,
    DextraRegressor,
    DextraSelectPipeline,
)

WRAPPERS = [DextraFeaturePipeline, DextraSelectPipeline, DextraRegressor,
            DextraClassifier, DextraClusterer]


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 160
    X = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(5000, 900, n),
        "score": rng.normal(50, 12, n),
    })
    y_reg = 2 * X["age"] + 0.01 * X["income"] + rng.normal(0, 5, n)
    lin = (X["score"] - 50) / 12 + rng.normal(0, 0.4, n)
    y_clf = np.where(lin > 0, "yes", "no")
    return X, y_reg.to_numpy(), y_clf


def test_exported_on_namespace():
    for cls in WRAPPERS:
        assert hasattr(dx, cls.__name__)


@pytest.mark.parametrize("cls", WRAPPERS)
def test_get_set_params_and_clone(cls):
    est = cls()
    params = est.get_params()
    assert isinstance(params, dict)
    est.set_params(**params)
    clone(est)                        # sklearn clone must succeed


def test_feature_pipeline_transformer(data):
    X, _, _ = data
    fp = DextraFeaturePipeline(steps=[
        {"fn": "scale", "cols": ["age", "income"], "method": "standard"}])
    Xt = fp.fit_transform(X)
    assert isinstance(Xt, pd.DataFrame) and len(Xt) == len(X)
    assert hasattr(fp, "params_")


def test_select_pipeline_transformer(data):
    X, _, y_clf = data
    sp = DextraSelectPipeline(steps=[
        {"fn": "relevance", "method": "anova", "keep": 2}])
    Xt = sp.fit_transform(X, y_clf)
    assert isinstance(Xt, pd.DataFrame) and len(Xt) == len(X)


def test_regressor_fit_predict_score(data):
    X, y_reg, _ = data
    reg = DextraRegressor(method="forest").fit(X, y_reg)
    pred = reg.predict(X)
    assert len(pred) == len(X)
    assert hasattr(reg, "params_") and reg.estimator_ is not None
    assert isinstance(reg.score(X, y_reg), float)     # RegressorMixin -> R^2


def test_classifier_predict_proba_and_classes(data):
    X, _, y_clf = data
    clf = DextraClassifier(method="logistic").fit(X, y_clf)
    pred = clf.predict(X)
    assert len(pred) == len(X)
    assert set(clf.classes_) == {"yes", "no"}
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert 0.0 <= clf.score(X, y_clf) <= 1.0           # ClassifierMixin -> acc


def test_clusterer_fit_predict_labels(data):
    X, _, _ = data
    cl = DextraClusterer(method="kmeans", k=3)
    labels = cl.fit_predict(X)
    assert len(labels) == len(X)
    assert hasattr(cl, "labels_")
    assert len(np.unique(labels)) == 3


def test_inside_sklearn_pipeline(data):
    X, _, y_clf = data
    pipe = Pipeline([
        ("fe", DextraFeaturePipeline(steps=[
            {"fn": "scale", "cols": ["age", "income", "score"],
             "method": "standard"}])),
        ("clf", DextraClassifier(method="forest")),
    ])
    pipe.fit(X, y_clf)
    pred = pipe.predict(X)
    assert len(pred) == len(X)
