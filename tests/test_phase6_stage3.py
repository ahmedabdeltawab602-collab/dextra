"""dextra Phase 6 Stage 6.3 -- cluster (clustering baseline).

Same unified contract as regress / classify (fit / apply / compare, hybrid
artifact) but UNSUPERVISED: cluster never accepts a target y. Covers both
algorithms (kmeans, agglomerative), automatic and user-fixed k, immutability,
idempotency + no-refit under apply, the agglomerative NearestCentroid apply
path, guard errors, and edge cases (constant feature, NaN propagation, single
feature, degenerate sizes), plus the audit trail. Skipped if scikit-learn is
absent (offline sandbox); the full run is on CI / run_validation.ps1.
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
def blob_df():
    """Three well-separated Gaussian blobs in 3 numeric features."""
    rng = np.random.default_rng(11)
    centers = np.array([[0, 0, 0], [8, 8, 8], [-8, 8, -8]], dtype=float)
    parts = [rng.normal(c, 0.5, size=(60, 3)) for c in centers]
    X = np.vstack(parts)
    return pd.DataFrame(X, columns=["x1", "x2", "x3"])


def _descriptor(params):
    return {k: v for k, v in params.items() if k != "estimator"}


# --- API ------------------------------------------------------------------

def test_exports_and_alias():
    assert hasattr(dx, "cluster") and hasattr(dx, "clus")
    assert dx.clus is dx.cluster


def test_cluster_takes_no_target():
    """Unsupervised: y is not even a parameter (cannot leak a target)."""
    df = pd.DataFrame({"a": [0.0, 1, 2, 8, 9, 10], "b": [0.0, 1, 2, 8, 9, 10]})
    with pytest.raises(TypeError):
        dx.cluster(df, y="a", **KW)          # noqa: unexpected keyword


# --- fit -------------------------------------------------------------------

@pytest.mark.parametrize("method", ["kmeans", "agglomerative"])
def test_fit_each_method(blob_df, method):
    out, params = dx.cluster(blob_df, method=method, k=3,
                             return_params=True, **KW)
    assert "cluster" in out.columns and "cluster" not in blob_df.columns
    assert params["function"] == "cluster" and params["task"] == "clustering"
    assert params["algorithm"] == method
    assert params["target"] is None
    assert params["pred_col"] == "cluster"
    assert set(params["metrics"]) == {"fit"}
    assert set(params["metrics"]["fit"]) == {"silhouette", "inertia",
                                             "n_clusters"}
    assert params["metrics"]["fit"]["n_clusters"] == 3
    assert params["metadata"]["k"] == 3
    # estimator is a usable, predict-capable sklearn Pipeline
    assert hasattr(params["estimator"], "predict")
    # descriptor (estimator stripped) is strict-JSON-serialisable
    json.loads(json.dumps(_descriptor(params)))
    # original frame untouched
    assert "cluster" not in blob_df.columns
    assert out["cluster"].nunique() == 3


def test_inertia_only_for_kmeans(blob_df):
    _, pk = dx.cluster(blob_df, method="kmeans", k=3, return_params=True, **KW)
    _, pa = dx.cluster(blob_df, method="agglomerative", k=3,
                       return_params=True, **KW)
    assert pk["metrics"]["fit"]["inertia"] is not None
    assert pa["metrics"]["fit"]["inertia"] is None      # undefined -> None


def test_auto_k_selection(blob_df):
    out, params = dx.cluster(blob_df, method="kmeans", k_range=(2, 6),
                             return_params=True, **KW)
    md = params["metadata"]
    assert md["k_selected_by"] == "silhouette"
    assert md["k_range"] == [2, 6]
    k = params["metrics"]["fit"]["n_clusters"]
    assert 2 <= k <= 6
    # three clean blobs -> silhouette should clearly prefer k = 3
    assert k == 3
    assert params["metrics"]["fit"]["silhouette"] > 0.5


def test_user_k_overrides_search(blob_df):
    _, params = dx.cluster(blob_df, method="kmeans", k=4,
                           return_params=True, **KW)
    assert params["metadata"]["k_selected_by"] == "user"
    assert params["metadata"]["k_range"] is None
    assert params["metrics"]["fit"]["n_clusters"] == 4


def test_fit_audit_entry(blob_df):
    out = dx.cluster(blob_df, method="kmeans", k=3, **KW)
    audit = out.attrs["dextra_audit"]
    last = audit[-1]
    assert last["stage"] == "modeling" and last["function"] == "cluster"
    assert last["mode"] == "fit"


# --- apply -----------------------------------------------------------------

@pytest.mark.parametrize("method", ["kmeans", "agglomerative"])
def test_apply_reproduces_and_is_idempotent(blob_df, method):
    _, params = dx.cluster(blob_df, method=method, k=3,
                           return_params=True, **KW)
    est = params["estimator"]
    a = dx.cluster(blob_df, params=params, **KW)
    b = dx.cluster(blob_df, params=params, **KW)
    assert np.array_equal(a["cluster"].to_numpy(), b["cluster"].to_numpy())
    # apply must NOT re-fit: estimator identity preserved
    assert params["estimator"] is est
    # apply mode records mode=apply
    assert a.attrs["dextra_audit"][-1]["mode"] == "apply"


def test_apply_assigns_new_rows(blob_df):
    """Agglomerative apply works via the NearestCentroid path (no native predict)."""
    _, params = dx.cluster(blob_df, method="agglomerative", k=3,
                           return_params=True, **KW)
    new = pd.DataFrame({"x1": [0.0, 8, -8], "x2": [0.0, 8, 8],
                        "x3": [0.0, 8, -8]})
    out = dx.cluster(new, params=params, **KW)
    assert "cluster" in out.columns
    assert len(out) == 3
    # the three points sit on the three blob centres -> three distinct clusters
    assert out["cluster"].nunique() == 3


def test_apply_missing_feature_raises(blob_df):
    _, params = dx.cluster(blob_df, method="kmeans", k=3,
                           return_params=True, **KW)
    with pytest.raises(KeyError):
        dx.cluster(blob_df.drop(columns=["x2"]), params=params, **KW)


def test_apply_wrong_params_raises(blob_df):
    bogus = {"function": "regress", "estimator": None}
    with pytest.raises(ValueError):
        dx.cluster(blob_df, params=bogus, **KW)


def test_apply_nan_feature_raises(blob_df):
    _, params = dx.cluster(blob_df, method="kmeans", k=3,
                           return_params=True, **KW)
    bad = blob_df.copy()
    bad.loc[0, "x1"] = np.nan
    with pytest.raises(ValueError):
        dx.cluster(bad, params=params, **KW)


# --- compare ---------------------------------------------------------------

def test_compare_writes_nothing(blob_df):
    out = dx.cluster(blob_df, method="compare", k_range=(2, 5), **KW)
    assert "cluster" not in out.columns
    assert out.attrs["dextra_audit"][-1]["mode"] == "compare"


def test_compare_with_return_params_raises(blob_df):
    with pytest.raises(ValueError):
        dx.cluster(blob_df, method="compare", return_params=True, **KW)


# --- guards ----------------------------------------------------------------

def test_bad_method_raises(blob_df):
    with pytest.raises(ValueError):
        dx.cluster(blob_df, method="dbscan", **KW)


def test_k_too_small_raises(blob_df):
    with pytest.raises(ValueError):
        dx.cluster(blob_df, method="kmeans", k=1, **KW)


def test_k_too_large_raises():
    tiny = pd.DataFrame({"a": [0.0, 1, 2, 3], "b": [0.0, 1, 2, 3]})
    with pytest.raises(ValueError):
        dx.cluster(tiny, method="kmeans", k=10, **KW)


def test_standardize_override(blob_df):
    _, params = dx.cluster(blob_df, method="kmeans", k=3, standardize=False,
                           return_params=True, **KW)
    assert params["metadata"]["standardize"] is False


def test_plot_returns_figure(blob_df):
    fig = dx.cluster(blob_df, method="kmeans", k=3, return_df=False,
                     return_fig=True, show=False, plot=True)
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


# --- edge cases ------------------------------------------------------------

def test_constant_feature_does_not_crash(blob_df):
    df = blob_df.copy()
    df["const"] = 5.0
    out, params = dx.cluster(df, method="kmeans", k=3,
                             return_params=True, **KW)
    assert "const" in params["features"]
    assert "cluster" in out.columns


def test_nan_rows_dropped_in_fit_but_all_rows_labelled(blob_df):
    df = blob_df.copy()
    df.loc[0, "x1"] = np.nan
    out, params = dx.cluster(df, method="kmeans", k=3,
                             return_params=True, **KW)
    # the NaN row is excluded from training
    assert params["metadata"]["n"] == len(df) - 1
    # but every row (including the imputed one) still gets a label
    assert out["cluster"].notna().all()
    assert len(out) == len(df)


def test_all_nan_feature_raises(blob_df):
    df = blob_df.copy()
    df["x1"] = np.nan
    with pytest.raises(ValueError):
        dx.cluster(df, method="kmeans", k=3, **KW)


def test_single_feature():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"x": np.concatenate([rng.normal(0, 0.4, 40),
                                            rng.normal(10, 0.4, 40)])})
    out, params = dx.cluster(df, method="kmeans", k=2,
                             return_params=True, **KW)
    assert params["metrics"]["fit"]["n_clusters"] == 2
    assert "cluster" in out.columns
