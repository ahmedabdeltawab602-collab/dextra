"""dextra - lightweight exploratory-data-analysis helpers.

Quick start
-----------
>>> import dextra as dx
>>> dx.audit(df)          # master audit
>>> dx.na_show(df)        # show missing
>>> dx.impute(df)         # fix missing

Phase 2: 22 statistical helpers.
Phase 3 v2: 10 cleaning helpers (3 inspectors + 7 actors, short names).
"""

from ._loader import dload, dpeek, load, peek
from ._utils import DEFAULT_BOX_COLORS
from ._version import __version__
from .cleaning import (
    audit,
    cast_types,
    clean_report,
    clip_outliers,
    dedup,
    dedupe,
    dup_show,
    dupscan,
    handle_missing,
    impute,
    na_show,
    nascan,
    out_show,
    outscan,
    recast,
    standardize_columns,
    tidycols,
    validate_rules,
    verify,
    winsor,
)
from .compat import (
    DextraClassifier,
    DextraClusterer,
    DextraFeaturePipeline,
    DextraRegressor,
    DextraSelectPipeline,
)
from .dashboard import dash, dashapp
from .evaluation import (
    confrep,
    confusion_report,
    learncv,
    learning_curves,
    residan,
    residual_analysis,
    roc_pr,
    rocpr,
)
from .features import (
    aggfeat,
    bin,
    binize,
    cross,
    dtfeats,
    encode,
    featpipe,
    scale,
    transform,
)
from .modeling import (
    classify,
    clf,
    clus,
    cluster,
    reg,
    regress,
)
from .plots import boxpl, hister, plot_boxplots, plot_histograms
from .report import edarep, edareport
from .selection import (
    importance,
    imps,
    redun,
    redundancy,
    relev,
    relevance,
    rfe,
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
from .timeseries import tsdecomp, tsfcast, tsstat

_PHASE_LABELS = {
    "dextra._loader": "Phase 11 - loader (entry layer)",
    "dextra.stats": "Phase 1 - EDA",
    "dextra.plots": "Phase 1 - EDA",
    "dextra.stats_advanced": "Phase 2 - statistics",
    "dextra.cleaning": "Phase 3 - cleaning",
    "dextra._features_numeric": "Phase 4 - features",
    "dextra._features_discretize": "Phase 4 - features",
    "dextra._features_derive": "Phase 4 - features",
    "dextra._features_pipeline": "Phase 4 - features",
    "dextra.features": "Phase 4 - features",
    "dextra.selection": "Phase 5 - selection",
    "dextra.modeling": "Phase 6 - modeling",
    "dextra.evaluation": "Phase 7 - evaluation",
    "dextra.timeseries": "Phase 8 - timeseries",
    "dextra.report": "Phase 9 - report",
    "dextra.dashboard": "Phase 10 - dashboard",
    "dextra.compat": "scikit-learn compat",
}

_PHASE_ORDER = [
    "Phase 11 - loader (entry layer)",
    "Phase 1 - EDA", "Phase 2 - statistics", "Phase 3 - cleaning",
    "Phase 4 - features", "Phase 5 - selection", "Phase 6 - modeling",
    "Phase 7 - evaluation", "Phase 8 - timeseries", "Phase 9 - report",
    "Phase 10 - dashboard", "scikit-learn compat", "other",
]


def functions() -> None:
    """Print every public dextra function, grouped by phase, with a summary.

    A zero-dependency discoverability aid: ``import dextra as dx;
    dx.functions()`` lists the whole public API (functions and their short
    aliases) organised by phase, each with its one-line docstring summary.

    Examples
    --------
    >>> import dextra as dx
    >>> dx.functions()
    """
    import dextra as _dx
    groups: dict = {}
    for _name in __all__:
        _obj = getattr(_dx, _name, None)
        if not callable(_obj):
            continue
        _label = _PHASE_LABELS.get(getattr(_obj, "__module__", ""), "other")
        groups.setdefault(_label, []).append((_name, _obj))
    for _label in _PHASE_ORDER:
        if _label not in groups:
            continue
        print(f"\n# {_label}")
        for _name, _obj in groups[_label]:
            _doc = (_obj.__doc__ or "").strip().splitlines()
            _summary = _doc[0].strip() if _doc else ""
            print(f"  {_name:<26} {_summary}")


__all__ = [
    "__version__", "DEFAULT_BOX_COLORS", "functions",
    # Phase 11 - loader (entry layer)
    "load", "dload", "peek", "dpeek",
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
    # Phase 3 v2 inspectors
    "na_show", "dup_show", "out_show",
    # Aliases
    "zsc", "pskew", "emprule", "outrep", "corrmat", "slr",
    "missrep", "freqtab", "xtab", "gcmp",
    "cim", "cip", "ssm", "ssp",
    "normtest", "t1", "t2", "tpair", "aov1", "chi2ind",
    "vif", "imbalance",
    "dedup",
    "audit", "nascan", "dupscan", "outscan",
    "tidycols", "recast", "impute", "winsor", "verify",
    # Phase 4
    "transform", "scale",
    "bin", "binize", "encode",
    "dtfeats", "cross", "aggfeat",
    "featpipe",
    # Phase 5
    "redundancy", "relevance",
    "importance", "rfe",
    "selectpipe",
    "redun", "relev", "imps", "selpipe",
    # Phase 6 Stage 6.1
    "regress", "reg",
    # Phase 6 Stage 6.2
    "classify", "clf",
    # Phase 6 Stage 6.3
    "cluster", "clus",
    # Phase 7 - evaluation
    "confusion_report", "roc_pr", "residual_analysis", "learning_curves",
    "confrep", "rocpr", "residan", "learncv",
    # Phase 8 - timeseries
    "tsdecomp", "tsstat", "tsfcast",
    # Phase 9 - report
    "edareport", "edarep",
    # Phase 10 - dashboard
    "dash", "dashapp",
    # Hardening sprint - scikit-learn compatible wrappers (dextra.compat)
    "DextraFeaturePipeline", "DextraSelectPipeline",
    "DextraRegressor", "DextraClassifier", "DextraClusterer",
]


def __getattr__(name):  # PEP 562 -- deprecated Phase 3 aliases (audit #10)
    from . import cleaning as _cleaning
    if name in _cleaning._DEPRECATED_ALIASES:
        return getattr(_cleaning, name)  # warns there, names the official alias
    raise AttributeError(f"module 'dextra' has no attribute {name!r}")
