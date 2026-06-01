"""scikit-learn conformance for the dextra.compat wrappers (consistency sprint).

Full ``check_estimator`` is intentionally NOT used: it assumes tiny array-only
datasets that break CV-based baselines and ignores dextra's column-aware /
recipe-driven design. Instead these tests assert the conformance that matters
in practice -- the behaviours sklearn's Pipeline / GridSearchCV / joblib rely on:
``fit`` returns self, ``clone`` reproduces results, pickling round-trips, and
``repr`` works. Skipped if scikit-learn is absent.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402

from dextra.compat import (  # noqa: E402
    DextraClassifier,
    DextraClusterer,
    DextraFeaturePipeline,
    DextraRegressor,
    DextraSelectPipeline,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame({
        "age": rng.normal(40, 9, n),
        "income": rng.normal(5000, 800, n),
        "score": rng.normal(50, 12, n),
    })
    y_reg = 2 * X["age"] + 0.01 * X["income"] + rng.normal(0, 4, n)
    y_clf = np.where(X["score"] > 50, "yes", "no")
    return X, y_reg.to_numpy(), y_clf


def _fe():
    return DextraFeaturePipeline(steps=[
        {"fn": "scale", "cols": ["age", "income", "score"], "method": "standard"}])


# --- fit returns self -------------------------------------------------------

def test_fit_returns_self(data):
    X, y_reg, y_clf = data
    assert _fe().fit(X) .__class__.__name__ == "DextraFeaturePipeline"
    assert DextraRegressor(method="forest").fit(X, y_reg).__class__ is DextraRegressor
    est = DextraClusterer(method="kmeans", k=3)
    assert est.fit(X) is est


# --- clone reproduces predictions (deterministic estimators) ---------------

def test_clone_reproduces_predictions(data):
    X, y_reg, _ = data
    reg = DextraRegressor(method="forest").fit(X, y_reg)
    reg2 = clone(reg).fit(X, y_reg)
    assert np.allclose(reg.predict(X), reg2.predict(X))

    cl = DextraClusterer(method="kmeans", k=3).fit(X)
    cl2 = clone(cl).fit(X)
    assert np.array_equal(cl.labels_, cl2.labels_)


# --- pickle round-trip ------------------------------------------------------

def test_pickle_round_trip(data):
    X, _, y_clf = data
    clf = DextraClassifier(method="forest").fit(X, y_clf)
    restored = pickle.loads(pickle.dumps(clf))
    assert np.array_equal(clf.predict(X), restored.predict(X))

    fe = _fe().fit(X)
    fe_restored = pickle.loads(pickle.dumps(fe))
    pd.testing.assert_frame_equal(fe.transform(X), fe_restored.transform(X))


# --- repr -------------------------------------------------------------------

@pytest.mark.parametrize("est", [
    DextraFeaturePipeline(), DextraSelectPipeline(), DextraRegressor(),
    DextraClassifier(), DextraClusterer(),
])
def test_repr_contains_class_name(est):
    assert type(est).__name__ in repr(est)
