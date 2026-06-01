# Changelog

All notable changes to `dextra` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

These changes are merged but not yet released. The next release should be
tagged **0.2.0** (new features, backward-compatible).

### Added
- **Phase 2 - `stats_advanced`:** 22 statistical functions (+ 22 aliases):
  `z_scores`, `pearson_skewness`, `empirical_rule_check`, `outliers_report`,
  `correlation_matrix`, `simple_linear_regression`, `missing_report`,
  `frequency_table`, `cross_tab`, `group_compare`, the four confidence-interval
  / sample-size helpers, six hypothesis tests (`normality_test`, three t-tests,
  `anova_oneway`, `chi_square_independence`), `vif_scores`, `class_imbalance`.
- **Phase 3 - `cleaning`:** 10 cleaning helpers (inspectors + actors) covering
  the DAMA-DMBOK stages, with an audit trail in `df.attrs['dextra_audit']`.
- **Phase 4 - `features`:** 8 leakage-safe feature-engineering functions with a
  fit/apply contract and JSON-serialisable `params` dict: `transform`, `scale`,
  `bin`, `encode`, `dtfeats`, `cross`, `aggfeat`, `featpipe`.
- **Phase 5 - `selection`:** 5 feature-selection functions (Filter / Embedded /
  Wrapper): `redundancy`, `relevance`, `importance`, `rfe`, `selectpipe`.
- **Phase 6 Stage 6.1 - `modeling`:** `regress` (alias `reg`) - one-line
  cross-validated regression baseline (linear / ridge / lasso / tree / forest /
  compare) with a hybrid artifact (JSON descriptor + fitted sklearn estimator).
  Edge-case-audited (constant target/feature, NaN propagation, skewed target);
  degenerate-metric formatting hardened (`None`/NaN -> `n/a`). See PHASE6_TEST_MATRIX.md.
- **Phase 6 Stage 6.2 - `classify`** (alias `clf`) - one-line cross-validated
  classification baseline (logistic / tree / forest / knn / compare), binary and
  multiclass, with accuracy / F1 / ROC-AUC and the same hybrid artifact contract.
- **Phase 6 Stage 6.3 - `cluster`** (alias `clus`) - one-line UNSUPERVISED
  clustering baseline (kmeans / agglomerative / compare) with automatic k
  selection by silhouette (elbow / inertia for kmeans). Never accepts a target;
  appends a `"cluster"` label column; same hybrid artifact contract (JSON
  descriptor + fitted sklearn Pipeline). Agglomerative apply assigns new rows via
  a NearestCentroid fitted on its cluster labels. **Phase 6 complete.**
- `dx.functions()` - prints every public function with its one-line summary.
- `describe_numeric(..., ddof=1)` - choose sample (1) or population (0) std/var.
- `plot_histograms(..., bins='auto')` - bins now default to NumPy's automatic
  rule and accept any NumPy bin string ('fd', 'sturges', ...).

### Changed
- `describe_numeric` no longer floods the `modes` row on continuous data:
  all-unique columns show `-` and long mode lists are capped with `... (N more)`.
- Plain-terminal output caps column width so wide tables stay readable.

### Testing
- The staged verification suites for Phases 2-5 (~600 assertions) are vendored
  into `tests/legacy/` and now run under pytest in CI, alongside a coverage gate.

### Planned
- `describe_categorical` - summaries for object/category columns.
- Phase 6 `modeling` (`regress`, `classify`, `cluster`) - see MODELING_PHILOSOPHY.md.
- Phases 7-10: `evaluation`, `timeseries`, `report` (one-call HTML/PDF), `dashboard`.

## [0.1.0] - 2026-04-24

### Added
- Initial public release.
- `describe_numeric` - rich numeric summary with 21 metrics per column
  (mean, std, var, coefficient of variation, IQR, Tukey bounds, outlier
  counts, skewness, kurtosis, modes, and more).
- `plot_histograms` - matplotlib histograms with KDE overlay and a
  monospace statistics panel beside each plot.
- `plot_boxplots` - interactive Plotly box-plots with annotated Tukey
  bounds and per-row statistics.
- Backwards-compatible aliases: `numdesc`, `hister`, `boxpl`.
- `DEFAULT_BOX_COLORS` palette exported at top level.
- Flags on every function: `show=`, `return_df=`, `return_fig=`, `raw=`.

### Dependencies
- Python 3.9+
- numpy >= 1.23
- pandas >= 1.5
- matplotlib >= 3.6
- seaborn >= 0.12
- plotly >= 5.10

[Unreleased]: https://github.com/ahmedabdeltawab602-collab/dextra/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ahmedabdeltawab602-collab/dextra/releases/tag/v0.1.0