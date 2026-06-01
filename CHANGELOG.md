# Changelog

All notable changes to `dextra` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

_Nothing yet._

## [0.2.0] - 2026-06-01

Completes the modeling phase (Phase 6) and a hardening sprint (Phase 6.5):
scikit-learn-compatible wrappers, optional Plotly, PEP 561 typing markers,
property-based tests and an informational mypy gate. Backward-compatible.

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
- **Phase 6.5 - `dextra.compat`:** scikit-learn-compatible wrappers so dextra's
  pipelines and models drop into `sklearn.pipeline.Pipeline` / `GridSearchCV`:
  `DextraFeaturePipeline`, `DextraSelectPipeline`, `DextraRegressor`,
  `DextraClassifier`, `DextraClusterer` (fit/transform/predict, get/set_params,
  clone-able). scikit-learn stays an optional (`ml`) extra.
- **PEP 561:** the package now ships a `py.typed` marker so downstream type
  checkers consume dextra's annotations.
- **polars / pyarrow input (extra `perf`):** the data-processing and modeling
  entry points now accept any table exposing `.to_pandas()` (polars DataFrame,
  pyarrow Table) and convert it at the boundary via the shared `_ensure_pandas`
  helper; pandas frames pass through unchanged (zero-copy).
- **Benchmarks:** a `benchmarks/` micro-benchmark suite (pytest-benchmark) and a
  non-blocking CI job track performance of hot paths over time.
- `dx.functions()` - prints every public function with its one-line summary.
- `describe_numeric(..., ddof=1)` - choose sample (1) or population (0) std/var.
- `plot_histograms(..., bins='auto')` - bins now default to NumPy's automatic
  rule and accept any NumPy bin string ('fd', 'sturges', ...).

### Changed
- `describe_numeric` no longer floods the `modes` row on continuous data:
  all-unique columns show `-` and long mode lists are capped with `... (N more)`.
- Plain-terminal output caps column width so wide tables stay readable.
- **Plotly is now optional** (extra `viz`), not a core dependency, so headless
  / server installs are lighter. `plot_boxplots` imports Plotly lazily and
  raises a clear error if it is missing.

### Fixed
- `z_scores` no longer raises `KeyError` on a constant / zero-variance column:
  a fully degenerate selection now reports "no extremes" instead of crashing
  (found by the new Hypothesis property tests).

### Testing
- The staged verification suites for Phases 2-5 (~600 assertions) are vendored
  into `tests/legacy/` and now run under pytest in CI, alongside a coverage gate.
- Phase 6 stage suites (`regress`/`classify`/`cluster`), `tests/test_compat.py`
  for the sklearn wrappers, and Hypothesis property-based tests
  (`tests/test_property.py`). mypy runs informationally (non-blocking) in CI.

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

[Unreleased]: https://github.com/ahmedabdeltawab602-collab/dextra/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ahmedabdeltawab602-collab/dextra/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ahmedabdeltawab602-collab/dextra/releases/tag/v0.1.0