# dextra — Roadmap

> **Vision.** A complete data-analysis toolkit built around one core idea:
> **one line of user code → rich numeric results + comprehensive visual
> output + a one-sentence decision.**
> Every module — from descriptive statistics to a one-line interactive
> dashboard — must honour that contract.

---

## Design contract (applies to every function in every phase)

1. **One-line invocation.** `dx.something(df)` is enough to get a useful result.
2. **Rich tabular output.** Each function returns a DataFrame densely packed
   with related metrics (matches the spirit of `describe_numeric`).
3. **Comprehensive visual.** Every analytical function ships with a multi-panel
   figure (plot + side annotation panel of stats), matching the
   `plot_histograms` layout.
4. **Decision sentence.** Every function prints a short English sentence
   summarising the result (`Decision: ...`).
5. **Standard flags.** `show=True`, `return_df`, `return_fig`, `decimals`,
   `df_name`, `raw` — consistent across the library.
6. **Aliases.** Every long-named function has a 3–8 letter alias
   (`numdesc`, `zsc`, `pskew`, etc.).
7. **No new heavy dependencies without explicit approval.**
   Required: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`.
   Optional: `scikit-learn` (extra `ml`; Phase 5/6 + `dextra.compat`), `plotly` (extra `viz`; interactive box-plots).

---

## The 10-phase roadmap

| #  | Module                | Purpose                                                                                  | Status      |
|----|-----------------------|------------------------------------------------------------------------------------------|-------------|
| 1  | `stats.py` + `plots.py` | `describe_numeric`, `plot_histograms`, `plot_boxplots` — the foundational EDA helpers. | ✅ Complete |
| 2  | `stats_advanced.py`   | 22 statistical functions: descriptive extensions, bivariate, EDA market tools, inference, hypothesis tests, ML diagnostics. | ✅ Complete (22 funcs + 22 aliases) |
| 3  | `cleaning.py`         | 7 cleaning helpers across all 8 DAMA-DMBOK stages: `clean_report`, `standardize_columns`, `cast_types`, `validate_rules`, `handle_missing`, `dedupe`, `clip_outliers`. | ✅ Complete |
| 4  | `features.py`         | 8 feature-engineering functions: `transform`, `scale`, `bin`, `encode`, `dtfeats`, `cross`, `aggfeat`, `featpipe` — fit/apply framework, leakage-safe, choice-first. | ✅ Complete (8 funcs) |
| 5  | `selection.py`        | 5 feature-selection functions: `redundancy`, `relevance`, `importance`, `rfe`, `selectpipe` — Filter + Embedded + Wrapper families, fit/apply, leakage-safe. | ✅ Complete (5 funcs) |
| 6  | `modeling.py`         | `regress`, `classify`, `cluster` — instant baseline models with one call. | ✅ Complete (6.1 regress + 6.2 classify + 6.3 cluster) |
| 7  | `evaluation.py`       | `confusion_report`, `roc_pr`, `residual_analysis`, `learning_curves` — multi-metric evaluation panels. | 📅 Planned |
| 8  | `timeseries.py`       | `ts_decompose`, `ts_stationarity`, `ts_quick_forecast` — time-series basics. (Optional.)| 📅 Planned |
| 9  | `report.py`           | `eda_report(df, out="report.html")` — one-call full HTML/PDF report. | 📅 Planned |
| 10 | `dashboard.py`        | `dx.dash(df)` — auto-generated **Streamlit dashboard** in the browser from a single line. | 📅 Planned (final goal) |

---

## Phase 2 — Statistics Advanced (current focus)

Split into 6 staged deliveries; each must pass tests before the next begins.

| Stage | Functions                                                                                                | Status      |
|-------|----------------------------------------------------------------------------------------------------------|-------------|
| 1     | `z_scores`, `pearson_skewness`, `empirical_rule_check`, `outliers_report`                                | ✅ Complete |
| 2     | `correlation_matrix`, `simple_linear_regression`                                                         | ✅ Complete |
| 3     | `missing_report`, `frequency_table`, `cross_tab`, `group_compare` (EDA market tools)                     | ✅ Complete |
| 4     | `confidence_interval_mean`, `confidence_interval_proportion`, `sample_size_mean`, `sample_size_proportion` | ✅ Complete |
| 5     | `normality_test`, `t_test_one_sample`, `t_test_two_sample`, `t_test_paired`, `anova_oneway`, `chi_square_independence` | ✅ Complete |
| 6     | `vif_scores`, `class_imbalance`                                                                          | ✅ Complete |

---

## Phase 4 — Feature Engineering

Blueprint: `FEATURES_PHILOSOPHY.md`. Split into 4 staged deliveries; each
passed its test script before the next began. Every function follows the
fit/transform contract: parameters are learned on training data only and
replayed verbatim on held-out data via a JSON-serialisable `params` dict —
the technical safeguard against data leakage.

| Stage | Functions                                       | Status      |
|-------|-------------------------------------------------|-------------|
| 4.1   | `transform`, `scale`                            | ✅ Complete |
| 4.2   | `bin`, `encode`                                 | ✅ Complete |
| 4.3   | `dtfeats`, `cross`, `aggfeat`                   | ✅ Complete |
| 4.4   | `featpipe` (pipeline wrapper + round-trip test) | ✅ Complete |

**Phase 4 complete: 8/8 feature-engineering functions, 182/182 test checks
passing (45 + 57 + 45 + 35).**

---

## Phase 5 — Feature Selection

Blueprint: `SELECTION_PHILOSOPHY.md`. Split into 3 staged deliveries; each
passed its test script before the next began. Every function follows the
fit/apply contract: the kept-column set is learned on training data only and
replayed verbatim on held-out data -- the safeguard against selection leakage.
Selection removes existing columns; it never creates a new one. scikit-learn
is imported lazily, only where the model-based selectors need it.

| Stage | Functions | Family |
|-------|-----------|--------|
| 5.1   | `redundancy`, `relevance`  | Filter (target-free + target-based) |
| 5.2   | `importance`, `rfe`        | Embedded + Wrapper (lazy scikit-learn) |
| 5.3   | `selectpipe`               | Pipeline wrapper + round-trip test |

**Phase 5 complete: 5/5 feature-selection functions, 129/129 test checks
passing (47 + 45 + 37).**

---

## Source of truth for formulas

All statistical formulas implemented in Phase 2 are matched against
`Statistics Course Build/formulas.json` (28 formulas). Each implementation
carries an inline comment with its formula ID (e.g. `F-M03-L05-02`) to make
auditing trivial.

---

## Update history

| Date       | Milestone                                                |
|------------|----------------------------------------------------------|
| 2026-05-17 | Phase 2 Stage 1 delivered (4 functions).                 |
| 2026-05-17 | Roadmap document created.                                |
| 2026-05-17 | Phase 2 Stage 2 delivered (2 functions: bivariate).      |
| 2026-05-17 | scipy added as a required dependency in pyproject.toml.  |
| 2026-05-17 | Phase 2 Stage 3 delivered (4 EDA market tools).          |
| 2026-05-17 | Phase 2 Stage 4 delivered (4 inference helpers: CI + n). |
| 2026-05-17 | Phase 2 Stage 5 delivered (6 hypothesis tests).          |
| 2026-05-17 | Phase 2 Stage 6 delivered (VIF + class imbalance).       |
| 2026-05-17 | **Phase 2 complete: 22/22 advanced statistical helpers.** |
| 2026-05-18 | CLEANING_PHILOSOPHY.md authored as Phase 3 blueprint.    |
| 2026-05-18 | Phase 3 Stage 1: clean_report + standardize_columns.     |
| 2026-05-18 | Phase 3 Stage 2: cast_types + validate_rules.            |
| 2026-05-18 | Phase 3 Stage 3: handle_missing + dedupe + clip_outliers. |
| 2026-05-18 | **Phase 3 complete: 7/7 cleaning helpers, 8/8 DAMA stages.** |
| 2026-05-18 | FEATURES_PHILOSOPHY.md authored as Phase 4 blueprint.    |
| 2026-05-21 | Phase 4 Stage 4.1: transform + scale.                    |
| 2026-05-21 | Phase 4 Stage 4.2: bin + encode.                         |
| 2026-05-21 | Phase 4 Stage 4.3: dtfeats + cross + aggfeat.            |
| 2026-05-23 | Phase 4 Stage 4.4: featpipe pipeline wrapper.            |
| 2026-05-23 | **Phase 4 complete: 8/8 feature-engineering functions.** |
| 2026-05-23 | SELECTION_PHILOSOPHY.md authored as Phase 5 blueprint.   |
| 2026-05-23 | Phase 5 Stage 5.1: redundancy + relevance (Filter).      |
| 2026-05-23 | Phase 5 Stage 5.2: importance + rfe (Embedded + Wrapper).|
| 2026-05-23 | Phase 5 Stage 5.3: selectpipe pipeline wrapper.          |
| 2026-05-23 | **Phase 5 complete: 5/5 feature-selection functions.**   |
| 2026-05-31 | MODELING_PHILOSOPHY.md authored as Phase 6 blueprint.        |
| 2026-05-31 | Phase 5.5 consolidation: tests vendored to CI; README/CHANGELOG synced; IMPROVEMENTS #1/#3/#5/#8 applied. |
| 2026-05-31 | Phase 6 Stage 6.1 delivered: regress (linear/ridge/lasso/tree/forest + compare), hybrid artifact, tests. |
| 2026-05-31 | Phase 6 Stage 6.2 delivered: classify (logistic/tree/forest/knn + compare), binary+multiclass, accuracy/F1/ROC-AUC, tests. |
| 2026-06-01 | Phase 6 Stage 6.3 delivered: cluster (kmeans/agglomerative + compare), automatic k by silhouette, unsupervised, hybrid artifact, tests. **Phase 6 complete: 3/3 modeling functions.** |
| 2026-06-01 | Phase 6.5 hardening sprint: `dextra.compat` sklearn wrappers; Plotly made optional (`viz` extra) + lazy; PEP 561 `py.typed`; Hypothesis property tests; non-blocking mypy in CI; mkdocs-material docs scaffold; version bumped to 0.2.0. |
| 2026-06-01 | Performance sprint 1A: polars/pyarrow input accepted at all data/model entry points via `_ensure_pandas` (extra `perf`); `benchmarks/` suite + non-blocking CI bench job. |
