"""dextra - lightweight exploratory-data-analysis helpers.

Quick start
-----------
>>> import dextra as dx
>>> dx.clean_rep(df)      # master audit
>>> dx.na_show(df)        # show missing
>>> dx.na_fix(df)         # fix missing

Phase 2: 22 statistical helpers.
Phase 3 v2: 10 cleaning helpers (3 inspectors + 7 actors, short names).
"""

from ._version import __version__
from ._utils import DEFAULT_BOX_COLORS
from .stats import describe_numeric, numdesc
from .plots import plot_histograms, plot_boxplots, hister, boxpl
from .stats_advanced import (
    z_scores, pearson_skewness, empirical_rule_check, outliers_report,
    correlation_matrix, simple_linear_regression,
    missing_report, frequency_table, cross_tab, group_compare,
    confidence_interval_mean, confidence_interval_proportion,
    sample_size_mean, sample_size_proportion,
    normality_test, t_test_one_sample, t_test_two_sample,
    t_test_paired, anova_oneway, chi_square_independence,
    vif_scores, class_imbalance,
    zsc, pskew, emprule, outrep, corrmat, slr,
    missrep, freqtab, xtab, gcmp,
    cim, cip, ssm, ssp,
    normtest, t1, t2, tpair, aov1, chi2ind,
    vif, imbalance,
)
from .cleaning import (
    # v1 long names (kept for backward compat)
    clean_report, standardize_columns, cast_types, validate_rules,
    handle_missing, dedupe, clip_outliers,
    # v1 aliases
    cleanrep, stdcols, cast, vrules,
    fillna_smart, dedup, clipout,
    # v2 inspectors (NEW — Stage 3.4)
    na_show, dup_show, out_show,
    # v2 actor short aliases
    na_fix, dup_fix, out_fix,
    rule_check, type_fix, col_clean, col_fix, clean_rep,
    # Professional underscore-free aliases (polars/tidyverse style)
    audit, nascan, dupscan, outscan,
    tidycols, recast, impute, winsor, verify,
)
from .features import (
    # Phase 4 Stage 4.1 - feature engineering
    transform, scale,
    # Phase 4 Stage 4.2 - feature engineering
    bin, encode,
    # Phase 4 Stage 4.3 - feature engineering
    dtfeats, cross, aggfeat,
    # Phase 4 Stage 4.4 - pipeline wrapper
    featpipe,
)
from .selection import (
    # Phase 5 Stage 5.1 - feature selection (Filter family)
    redundancy, relevance,
    # Phase 5 Stage 5.2 - feature selection (Embedded + Wrapper families)
    importance, rfe,
    # Phase 5 Stage 5.3 - selection pipeline wrapper
    selectpipe,
    # Phase 5 aliases
    redun, relev, imps, selpipe,
)
from .modeling import (
    # Phase 6 Stage 6.1 - regression baseline
    regress, reg,
)



def functions() -> None:
    """Print every public dextra function with its one-line summary.

    A zero-dependency discoverability aid: ``import dextra as dx;
    dx.functions()`` lists the whole public API and what each entry does.
    """
    import dextra as _dx
    for _name in __all__:
        _obj = getattr(_dx, _name, None)
        if callable(_obj):
            _doc = (_obj.__doc__ or "").strip().splitlines()
            _summary = _doc[0].strip() if _doc else ""
            print(f"{_name:<26} {_summary}")


__all__ = [
    "__version__", "DEFAULT_BOX_COLORS", "functions",
    "describe_numeric", "plot_histograms", "plot_boxplots",
    "numdesc", "hister", "boxpl",
    # Phase 2
    "z_scores", "pearson_skewness", "empirical_rule_check", "outliers_report",
    "correlation_matrix", "simple_linear_regression",
    "missing_report", "frequency_table", "cross_tab", "group_compare",
    "confidence_interval_mean", "confidence_interval_proportion",
    "sample_size_mean", "sample_size_proportion",
    "normality_test", "t_test_one_sample", "t_test_two_sample",
    "t_test_paired", "anova_oneway", "chi_square_independence",
    "vif_scores", "class_imbalance",
    # Phase 3 v1 (kept for compat)
    "clean_report", "standardize_columns", "cast_types", "validate_rules",
    "handle_missing", "dedupe", "clip_outliers",
    # Phase 3 v2 (NEW)
    "na_show", "dup_show", "out_show",
    "na_fix", "dup_fix", "out_fix",
    "rule_check", "type_fix", "col_clean", "col_fix", "clean_rep",
    # Aliases
    "zsc", "pskew", "emprule", "outrep", "corrmat", "slr",
    "missrep", "freqtab", "xtab", "gcmp",
    "cim", "cip", "ssm", "ssp",
    "normtest", "t1", "t2", "tpair", "aov1", "chi2ind",
    "vif", "imbalance",
    "cleanrep", "stdcols", "cast", "vrules",
    "fillna_smart", "dedup", "clipout",
    # Professional underscore-free aliases
    "audit", "nascan", "dupscan", "outscan",
    "tidycols", "recast", "impute", "winsor", "verify",
    # Phase 4 Stage 4.1
    "transform", "scale",
    # Phase 4 Stage 4.2
    "bin", "encode",
    # Phase 4 Stage 4.3
    "dtfeats", "cross", "aggfeat",
    # Phase 4 Stage 4.4
    "featpipe",
    # Phase 5 Stage 5.1
    "redundancy", "relevance",
    # Phase 5 Stage 5.2
    "importance", "rfe",
    # Phase 5 Stage 5.3
    "selectpipe",
    "redun", "relev", "imps", "selpipe",
    # Phase 6 Stage 6.1
    "regress", "reg",
]
