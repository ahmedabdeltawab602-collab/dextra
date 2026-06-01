"""Breadth tests to raise coverage of the lower-covered modules.

Exercises cleaning and stats_advanced (the two lowest-covered modules) plus a
variety of feature-engineering methods, in silent mode (show=False, plot=False).
Assertions are deliberately loose -- the goal is to execute many code paths
without crashing, not to re-assert numerical results already covered by the
dedicated suites. scikit-learn-only paths are guarded individually.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import dextra as dx  # noqa: E402

KW = dict(show=False, plot=False)


@pytest.fixture
def data():
    rng = np.random.default_rng(7)
    n = 160
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.lognormal(9, 0.4, n),          # strictly positive
        "score": rng.normal(50, 12, n),
        "spend": rng.normal(200, 40, n),
        "city": rng.choice(["Cairo", "Giza", "Alex"], n),
        "gender": rng.choice(["M", "F"], n),
        "signup": pd.to_datetime("2023-01-01")
        + pd.to_timedelta(rng.integers(0, 500, n), unit="D"),
    })
    df["price"] = 3 * df["age"] + 0.001 * df["income"] + rng.normal(0, 8, n)
    df["churn"] = np.where(df["score"] > 50, "yes", "no")
    return df


@pytest.fixture
def messy():
    return pd.DataFrame({
        " First Name ": ["  Ann", "Bob", "Bob", "CARLA", None, "Dan", "Dan"],
        "AGE": ["25", "30", "30", "40", "45", "50", "50"],
        "Income($)": [5000.0, 6000, 6000, 7000, np.nan, 999999, 8000],
        "City": ["Cairo", "Giza", "Giza", "Alex", "Cairo", "Cairo", "Cairo"],
    })


# --- cleaning ---------------------------------------------------------------

def test_cleaning_paths(data, messy):
    df_nan = data.copy()
    df_nan.loc[df_nan.sample(15, random_state=1).index, "income"] = np.nan
    assert dx.clean_report(messy, return_df=True, **KW) is not None
    assert dx.standardize_columns(messy, return_df=True, **KW) is not None
    assert dx.cast_types(messy, return_df=True, **KW) is not None
    assert dx.handle_missing(df_nan, return_df=True, **KW) is not None
    assert dx.dedupe(messy, return_df=True, **KW) is not None
    assert dx.clip_outliers(data, cols=["income"], return_df=True, **KW) is not None
    assert dx.na_show(df_nan, return_df=True, **KW) is not None
    assert dx.dup_show(messy, return_df=True, **KW) is not None
    assert dx.out_show(data, cols=["income"], return_df=True, **KW) is not None
    rules = [{"name": "age_ok", "check": "age.between(0, 120)"},
             {"name": "price_pos", "check": "price >= 0"}]
    assert dx.validate_rules(data, rules, return_df=True, **KW) is not None


# --- stats_advanced ---------------------------------------------------------

def test_stats_descriptive(data):
    assert dx.z_scores(data, cols=["age", "score"], return_df=True, **KW) is not None
    assert dx.pearson_skewness(data, cols=["income"], return_df=True, **KW) is not None
    assert dx.empirical_rule_check(data, cols=["score"], return_df=True, **KW) is not None
    assert dx.outliers_report(data, cols=["income"], method="zscore",
                              return_df=True, **KW) is not None


def test_stats_bivariate(data):
    assert dx.correlation_matrix(data, cols=["age", "income", "score", "price"],
                                 return_df=True, **KW) is not None
    assert dx.simple_linear_regression(data, x="score", y="price",
                                       return_df=True, **KW) is not None
    assert dx.frequency_table(data, col="city", return_df=True, **KW) is not None
    assert dx.cross_tab(data, row="city", col="gender", return_df=True, **KW) is not None
    assert dx.group_compare(data, group_col="city", value_cols=["price", "score"],
                            return_df=True, **KW) is not None


def test_stats_inference(data):
    assert dx.confidence_interval_mean(data["score"], return_df=True, **KW) is not None
    assert dx.confidence_interval_proportion(40, 160, return_df=True, **KW) is not None
    assert dx.sample_size_mean(2.0, 12.0, return_df=True, **KW) is not None
    assert dx.sample_size_proportion(0.05, return_df=True, **KW) is not None
    assert dx.normality_test(data["score"], return_df=True, **KW) is not None
    assert dx.t_test_one_sample(data["score"], popmean=50, return_df=True, **KW) is not None
    m = data.loc[data.gender == "M", "score"]
    f = data.loc[data.gender == "F", "score"]
    assert dx.t_test_two_sample(m, f, return_df=True, **KW) is not None
    assert dx.t_test_paired(data["score"], data["score"] + 1.0,
                            return_df=True, **KW) is not None
    assert dx.anova_oneway(data, group_col="city", value_col="price",
                           return_df=True, **KW) is not None
    assert dx.chi_square_independence(data, row="city", col="gender",
                                      return_df=True, **KW) is not None
    assert dx.vif_scores(data, cols=["age", "income", "score", "spend"],
                         return_df=True, **KW) is not None
    assert dx.class_imbalance(data["churn"], return_df=True, **KW) is not None


# --- feature-engineering method variety ------------------------------------

@pytest.mark.parametrize("method", ["log1p", "sqrt", "yeojohnson"])
def test_transform_methods(data, method):
    assert dx.transform(data, cols=["income"], method=method, **KW) is not None


@pytest.mark.parametrize("method", ["standard", "minmax", "robust", "maxabs"])
def test_scale_methods(data, method):
    assert dx.scale(data, cols=["age", "income"], method=method, **KW) is not None


@pytest.mark.parametrize("method", ["equal_width", "quantile", "kmeans"])
def test_bin_methods(data, method):
    assert dx.bin(data, cols=["income"], method=method, n_bins=4, **KW) is not None


@pytest.mark.parametrize("method", ["onehot", "ordinal", "frequency"])
def test_encode_methods(data, method):
    assert dx.encode(data, cols=["city"], method=method, **KW) is not None


def test_encode_target(data):
    # target (mean) encoding needs a NUMERIC target
    assert dx.encode(data, cols=["city"], method="target",
                     y=data["price"], **KW) is not None


@pytest.mark.parametrize("method", ["ratio", "product", "diff"])
def test_cross_methods(data, method):
    assert dx.cross(data, pairs=[("age", "score")], method=method, **KW) is not None


@pytest.mark.parametrize("agg", ["mean", "median", "sum", "max"])
def test_aggfeat_methods(data, agg):
    assert dx.aggfeat(data, group="city", value="income", agg=agg, **KW) is not None


@pytest.mark.parametrize("method", ["calendar", "cyclical", "both"])
def test_dtfeats_methods(data, method):
    assert dx.dtfeats(data, cols=["signup"], method=method, **KW) is not None


# --- selection / modeling (scikit-learn) -----------------------------------

def test_selection_models(data):
    pytest.importorskip("sklearn")
    num = ["age", "income", "score", "spend"]
    assert dx.importance(data, y="churn", cols=num, method="tree", keep=3, **KW) is not None
    assert dx.rfe(data, y="churn", cols=num, keep=2, estimator="tree", **KW) is not None


@pytest.mark.parametrize("method", ["linear", "ridge", "lasso", "tree"])
def test_regress_methods(data, method):
    pytest.importorskip("sklearn")
    num = ["age", "income", "score", "spend"]
    assert dx.regress(data, y="price", cols=num, method=method, **KW) is not None


@pytest.mark.parametrize("method", ["logistic", "tree", "knn"])
def test_classify_methods(data, method):
    pytest.importorskip("sklearn")
    num = ["age", "income", "score", "spend"]
    assert dx.classify(data, y="churn", cols=num, method=method, **KW) is not None


def test_cluster_agglomerative(data):
    pytest.importorskip("sklearn")
    assert dx.cluster(data, cols=["age", "income", "score"],
                      method="agglomerative", k=3, **KW) is not None
