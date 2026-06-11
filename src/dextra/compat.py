"""scikit-learn compatible wrappers around dextra's pipelines and models.

These thin adapters expose dextra's leakage-safe fit/apply pipelines (Phase 4
``featpipe`` / Phase 5 ``selectpipe``) and its baseline models (Phase 6
``regress`` / ``classify`` / ``cluster``) through the **standard scikit-learn
estimator API** (``fit`` / ``transform`` / ``predict`` / ``get_params`` /
``set_params``). They therefore drop directly into
:class:`sklearn.pipeline.Pipeline`, :class:`~sklearn.model_selection.GridSearchCV`,
``cross_val_score`` and friends, while keeping dextra's inspectable JSON
``params`` artifact reachable via the ``params_`` attribute after ``fit``.

scikit-learn is an optional dependency (extra ``ml``). Importing dextra always
works; instantiating any wrapper without scikit-learn raises a helpful error.

Examples
--------
>>> from sklearn.pipeline import Pipeline
>>> from dextra.compat import DextraFeaturePipeline, DextraClassifier
>>> pipe = Pipeline([
...     ("fe", DextraFeaturePipeline(steps=[
...         {"fn": "scale", "cols": ["age", "income"], "method": "standard"}])),
...     ("clf", DextraClassifier(method="forest")),
... ])
>>> pipe.fit(X_train, y_train)          # doctest: +SKIP
>>> pipe.predict(X_test)                # doctest: +SKIP
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .features import featpipe
from .modeling import classify, cluster, regress
from .selection import selectpipe

try:  # scikit-learn is the optional `ml` extra
    from sklearn.base import (
        BaseEstimator,
        ClassifierMixin,
        ClusterMixin,
        RegressorMixin,
        TransformerMixin,
    )
    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover - exercised only without scikit-learn
    _HAS_SKLEARN = False

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class TransformerMixin:  # type: ignore[no-redef]
        pass

    class RegressorMixin:  # type: ignore[no-redef]
        pass

    class ClassifierMixin:  # type: ignore[no-redef]
        pass

    class ClusterMixin:  # type: ignore[no-redef]
        pass


__all__ = [
    "DextraFeaturePipeline",
    "DextraSelectPipeline",
    "DextraRegressor",
    "DextraClassifier",
    "DextraClusterer",
]

_KW = {"show": False, "plot": False}


def _check_sklearn() -> None:
    if not _HAS_SKLEARN:
        raise ImportError(
            "dextra.compat wrappers require scikit-learn, which is not "
            "installed. Install it with `pip install scikit-learn` (or "
            "`pip install dextra[ml]`)."
        )


def _as_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))


# ===========================================================================
# Transformers (Phase 4 / Phase 5 pipelines)
# ===========================================================================

class DextraFeaturePipeline(TransformerMixin, BaseEstimator):
    """sklearn transformer wrapping :func:`dextra.featpipe` (fit/apply, leak-safe).

    Parameters
    ----------
    steps : sequence of dict
        A ``featpipe`` recipe, e.g.
        ``[{"fn": "scale", "cols": ["age"], "method": "standard"}]``.
    """

    def __init__(self, steps: Optional[Sequence[dict]] = None):
        self.steps = steps

    def fit(self, X, y=None):
        _check_sklearn()
        _, params = featpipe(_as_frame(X), steps=self.steps,
                             return_params=True, **_KW)
        self.params_ = params
        return self

    def transform(self, X):
        return featpipe(_as_frame(X), params=self.params_, **_KW)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


class DextraSelectPipeline(TransformerMixin, BaseEstimator):
    """sklearn transformer wrapping :func:`dextra.selectpipe` (fit/apply, leak-safe).

    Parameters
    ----------
    steps : sequence of dict
        A ``selectpipe`` recipe, e.g.
        ``[{"fn": "relevance", "method": "anova", "keep": 8}]``.
    """

    def __init__(self, steps: Optional[Sequence[dict]] = None):
        self.steps = steps

    def fit(self, X, y=None):
        _check_sklearn()
        X = _as_frame(X)
        extra = {} if y is None else {"y": pd.Series(np.asarray(y), index=X.index)}
        _, params = selectpipe(X, steps=self.steps, return_params=True,
                               **extra, **_KW)
        self.params_ = params
        return self

    def transform(self, X):
        return selectpipe(_as_frame(X), params=self.params_, **_KW)

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X)


# ===========================================================================
# Estimators (Phase 6 models)
# ===========================================================================

class _BaseModelWrapper(BaseEstimator):
    """Shared fit/predict plumbing for the model wrappers."""

    def _feature_cols(self, X: pd.DataFrame):
        return list(self.cols) if self.cols is not None else list(X.columns)

    def _predict_matrix(self, X) -> np.ndarray:
        X = _as_frame(X)
        cols = list(self.feature_names_in_)
        return (X[cols].apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float))

    def predict(self, X):
        return self.estimator_.predict(self._predict_matrix(X))


class DextraRegressor(RegressorMixin, _BaseModelWrapper):
    """sklearn regressor wrapping :func:`dextra.regress` (one baseline algorithm)."""

    def __init__(self, method: str = "forest",
                 cols: Optional[Sequence[str]] = None, cv: int = 5,
                 standardize: Optional[bool] = None):
        self.method = method
        self.cols = cols
        self.cv = cv
        self.standardize = standardize

    def fit(self, X, y):
        _check_sklearn()
        X = _as_frame(X)
        cols = self._feature_cols(X)
        df = X.copy()
        tgt = "__dextra_y__"
        df[tgt] = np.asarray(y)
        _, params = regress(df, y=tgt, cols=cols, method=self.method,
                            cv=self.cv, standardize=self.standardize,
                            return_params=True, **_KW)
        self.params_ = params
        self.estimator_ = params["estimator"]
        self.feature_names_in_ = np.asarray(cols, dtype=object)
        self.n_features_in_ = len(cols)
        return self


class DextraClassifier(ClassifierMixin, _BaseModelWrapper):
    """sklearn classifier wrapping :func:`dextra.classify` (one baseline algorithm)."""

    def __init__(self, method: str = "forest",
                 cols: Optional[Sequence[str]] = None, cv: int = 5,
                 standardize: Optional[bool] = None):
        self.method = method
        self.cols = cols
        self.cv = cv
        self.standardize = standardize

    def fit(self, X, y):
        _check_sklearn()
        X = _as_frame(X)
        cols = self._feature_cols(X)
        df = X.copy()
        tgt = "__dextra_y__"
        df[tgt] = np.asarray(y)
        _, params = classify(df, y=tgt, cols=cols, method=self.method,
                             cv=self.cv, standardize=self.standardize,
                             return_params=True, **_KW)
        self.params_ = params
        self.estimator_ = params["estimator"]
        self.classes_ = np.asarray(params["classes"])
        self.feature_names_in_ = np.asarray(cols, dtype=object)
        self.n_features_in_ = len(cols)
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self._predict_matrix(X))


class DextraClusterer(ClusterMixin, _BaseModelWrapper):
    """sklearn clusterer wrapping :func:`dextra.cluster` (unsupervised, no y).

    ``labels_`` holds the labels assigned by the fit itself (sklearn
    convention); ``predict`` assigns data via the persisted estimator (for
    ``agglomerative`` that is a nearest-centroid rule, which may disagree
    with the fit labels on boundary points).
    """

    def __init__(self, method: str = "kmeans",
                 cols: Optional[Sequence[str]] = None, k: Optional[int] = None,
                 k_range: Sequence[int] = (2, 10),
                 standardize: Optional[bool] = None):
        self.method = method
        self.cols = cols
        self.k = k
        self.k_range = k_range
        self.standardize = standardize

    def fit(self, X, y=None):
        _check_sklearn()
        X = _as_frame(X)
        cols = self._feature_cols(X)
        _, params = cluster(X, cols=cols, method=self.method, k=self.k,
                           k_range=self.k_range, standardize=self.standardize,
                           return_params=True, **_KW)
        self.params_ = params
        self.estimator_ = params["estimator"]
        self.feature_names_in_ = np.asarray(cols, dtype=object)
        self.n_features_in_ = len(cols)
        fit_labels = getattr(self.estimator_[-1], "labels_", None)
        if fit_labels is not None and len(fit_labels) == len(X):
            # The labels the fit actually assigned (sklearn convention).
            # predict() may disagree for agglomerative, whose deployed
            # predictor re-assigns by nearest centroid (audit #13).
            self.labels_ = np.asarray(fit_labels)
        else:
            # Rows were dropped internally (e.g. NaN) or the estimator has
            # no labels_: fall back to assigning via the deployed predictor.
            self.labels_ = self.predict(X)
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_
