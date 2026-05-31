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

from ._utils import DEFAULT_BOX_COLORS
from ._version import __version__
from .cleaning import (
    # Professional underscore-free aliases (polars/tidyverse style)
    audit,
    cast,
    cast_types,
    clean_rep,
    # v1 long names (kept for backward compat)
    clean_report,
    # v1 aliases
    cleanrep,
    clip_outliers,
    clipout,
    col_clean,
    col_fix,
    dedup,
    dedupe,
    dup_fix,
    dup_show,
    dupscan,
    fillna_smart,
    handle_missing,
    impute,
    # v2 actor short aliases
    na_fix,
    # v2 inspectors (NEW — Stage 3.4)
    na_show,
    nascan,
    out_fix,
    out_show,
    outscan,
    recast,
    rule_check,
    standardize_columns,
    stdcols,
    tidycols,
    type_fix,
    validate_rules,
    verify,
    vrules,
    winsor,
)
from .features import (
    aggfeat,
    # Phase 4 Stage 4.2 - feature engineering
    bin,
    cross,
    # Phase 4 Stage 4.3 - feature engineering
    dtfeats,
    encode,
    # Phase 4 Stage 4.4 - pipeline wrapper
    featpipe,
    scale,
    # Phase 4 Stage 4.1 - feature engineering
    transform,
)
from .modeling import (
    reg,
    # Phase 6 Stage 6.1 - regression baseline
    regress,
)
from .plots import boxpl, hister, plot_boxplots, plot_histograms
from .selection import (
    # Phase 5 Stage 5.2 - feature selection (Embedded + Wrapper families)
    importance,
    imps,
    # Phase 5 aliases
    redun,
    # Phase 5 Stage 5.1 - feature selection (Filter family)
    redundancy,
    relev,
    relevance,
    rfe,
    # Phase 5 Stage 5.3 - selection pipeline wrapper
    selectpipe,
    selpipe,
)
from .stats import describe_numeric, numdesc
from .stats_advanced import (
    anova_oneway,
    aov1,
    chi2ind,
    chi_square_independence,
    cim,
    cip,
    class_imbalance,
    confidence_interval_mean,
    confidence_interval_proportion,
    correlation_matrix,
    corrmat,
    cross_tab,
    empirical_rule_check,
    emprule,
    freqtab,
    frequency_table,
    gcmp,
    group_compare,
    imbalance,
    missing_report,
    missrep,
    normality_test,
    normtest,
    outliers_report,
    outrep,
    pearson_skewness,
    pskew,
    sample_size_mean,
    sample_size_proportion,
    simple_linear_regression,
    slr,
    ssm,
    ssp,
    t1,
    t2,
    t_test_one_sample,
    t_test_paired,
    t_test_two_sample,
    tpair,
    vif,
    vif_scores,
    xtab,
    z_scores,
    zsc,
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
