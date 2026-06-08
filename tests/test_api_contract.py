"""API-contract & consistency tests (consistency sprint).

Enforces the cross-module contract by introspection only (no execution):
every exported name resolves, public callables are documented, the
fit/apply families share the same standard flags, and every short alias is
the *same object* as its long-named function.
"""
from __future__ import annotations

import inspect

import dextra as dx

# fit/apply families must all expose these standard flags (the unified contract)
_STANDARD_FLAGS = {"params", "return_params", "show", "plot",
                   "return_df", "return_fig", "decimals", "df_name"}
_FAMILIES = ["transform", "scale", "bin", "encode", "dtfeats", "cross",
             "aggfeat", "featpipe", "redundancy", "relevance", "importance",
             "rfe", "selectpipe", "regress", "classify", "cluster",
             # Phase 1 EDA brought up to the unified contract (audit #4).
             "describe_numeric", "plot_histograms", "plot_boxplots",
             # Phase 2 batch 1 (audit #4).
             "pearson_skewness", "empirical_rule_check", "outliers_report",
             "missing_report", "frequency_table",
             # Phase 2 batch 2 (audit #4).
             "z_scores", "correlation_matrix", "simple_linear_regression",
             "group_compare", "anova_oneway", "chi_square_independence",
             "vif_scores", "cross_tab",
             # Phase 2 batch 3 — inference/test family (audit #4, name-> df_name).
             "confidence_interval_mean", "confidence_interval_proportion",
             "sample_size_mean", "sample_size_proportion", "normality_test",
             "t_test_one_sample", "t_test_two_sample", "t_test_paired",
             "class_imbalance"]

# short alias -> canonical function name
_ALIASES = {
    "numdesc": "describe_numeric", "hister": "plot_histograms", "boxpl": "plot_boxplots",
    "zsc": "z_scores", "pskew": "pearson_skewness", "emprule": "empirical_rule_check",
    "outrep": "outliers_report", "corrmat": "correlation_matrix",
    "slr": "simple_linear_regression", "missrep": "missing_report",
    "freqtab": "frequency_table", "xtab": "cross_tab", "gcmp": "group_compare",
    "cim": "confidence_interval_mean", "cip": "confidence_interval_proportion",
    "ssm": "sample_size_mean", "ssp": "sample_size_proportion",
    "normtest": "normality_test", "t1": "t_test_one_sample",
    "t2": "t_test_two_sample", "tpair": "t_test_paired", "aov1": "anova_oneway",
    "chi2ind": "chi_square_independence", "vif": "vif_scores",
    "imbalance": "class_imbalance",
    "cleanrep": "clean_report", "audit": "clean_report",
    "stdcols": "standardize_columns", "tidycols": "standardize_columns",
    "col_clean": "standardize_columns", "col_fix": "standardize_columns",
    "cast": "cast_types", "recast": "cast_types", "type_fix": "cast_types",
    "vrules": "validate_rules", "verify": "validate_rules", "rule_check": "validate_rules",
    "fillna_smart": "handle_missing", "impute": "handle_missing", "na_fix": "handle_missing",
    "dedup": "dedupe", "dup_fix": "dedupe",
    "clipout": "clip_outliers", "winsor": "clip_outliers", "out_fix": "clip_outliers",
    "nascan": "na_show", "dupscan": "dup_show", "outscan": "out_show",
    "redun": "redundancy", "relev": "relevance", "imps": "importance",
    "selpipe": "selectpipe", "reg": "regress", "clf": "classify", "clus": "cluster",
}


def test_all_exports_resolve():
    for name in dx.__all__:
        assert hasattr(dx, name), f"{name} is in __all__ but not importable"


def test_public_callables_have_docstrings():
    missing = []
    for name in dx.__all__:
        obj = getattr(dx, name, None)
        if callable(obj) and not (obj.__doc__ or "").strip():
            missing.append(name)
    assert not missing, f"public callables without a docstring: {missing}"


def test_fit_apply_families_share_standard_flags():
    for fn in _FAMILIES:
        params = set(inspect.signature(getattr(dx, fn)).parameters)
        missing = _STANDARD_FLAGS - params
        assert not missing, f"{fn} is missing standard flags: {missing}"


def test_aliases_are_identity():
    for alias, canonical in _ALIASES.items():
        assert hasattr(dx, alias), f"alias {alias} not exported"
        assert getattr(dx, alias) is getattr(dx, canonical), (
            f"alias {alias} is not the same object as {canonical}")
