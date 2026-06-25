# Changelog

All notable changes to `dextra` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased] — hardening before release

### Added
- **Leakage-safe fit/apply for cleaning -- `handle_missing` / `impute`
  and `clip_outliers` / `winsor` (eval M-5):** both functions now
  accept the standard `params` / `return_params` contract. FIT mode (`return_params=True`)
  freezes the per-column fill values learned from the fit data
  (mean/median/mode/constant store the value; random_uniform /
  random_normal store the fitted distribution; `auto` resolves its
  per-column choice on the fit data and freezes it); APPLY mode
  (`params=<dict>`) replays those values verbatim -- never recomputed --
  so held-out data is filled with train statistics. `drop_cols` replays
  the fitted column drop; order-based strategies
  (ffill/bfill/interpolate) re-run by nature (documented); columns with
  missing values but no fitted fill are left as NaN with a UserWarning.
  Apply on a frame missing fitted column(s) raises a clear KeyError;
  foreign params raise ValueError; `dry_run` cannot fit or apply.
  `clip_outliers` freezes the per-column lower/upper bounds (Tukey
  IQR or z-score) computed on the fit data and clips/drops held-out
  rows at those train bounds verbatim (unfittable columns --
  all-NaN, zero spread -- are excluded and disclosed in the
  docstring). Return order follows the unified contract:
  `(df, params[, fig])`;
  bare calls are unchanged. Audit entries carry `mode: fit/apply`.
  Both functions are now valid `featpipe` steps (`{"fn": "handle_missing"}` / `{"fn": "clip_outliers"}`), so a full clean -> engineer recipe fits on train and replays verbatim from one combined params artifact; both also join the `_FAMILIES` API-contract check. Regression tests: `tests/test_cleaning_fit_apply.py`.
- **Explicit model task (audit #7):** `edareport` / `dash` accept
  `task="auto" | "regression" | "classification"` for the model section.
  `"auto"` (the default, behaviour unchanged) infers regression when the
  target dtype is numeric and it has more than 10 unique values, otherwise
  classification; an explicit value overrides the inference. The dashboard
  exposes the choice as a "Model task" sidebar control, forwards it to the
  generated app and records it in the reproducibility manifest; both
  functions record it in their `return_params` manifest and audit entry.
  An unknown value raises `ValueError`.
- **`binize` (audit #9):** new official alternative name for `bin` (which
  shadows the Python builtin). `bin` remains a fully supported alias --
  nothing breaks; both names point to the same function.
- **Macro-benchmark (productionization):** new `benchmarks/macro_bench.py`
  times the documented workflow on a ~1M-row dirty synthetic frame (load ->
  full cleaning chain -> featpipe) and prints a ready-to-paste markdown
  table; methodology and measured results live in `docs/benchmarks.md` (new
  nav page). Measured: the cleaning + features chain runs in ~3.8 s for 10^6
  rows; `dx.load`'s measured-inference pass dominates (~55 s) as the
  disclosed one-time ingestion cost.

### Changed
- **Explicit frame labels (audit #11):** `peek` now exposes `df_name=`
  explicitly in its signature (previously it was only forwarded through
  `**load_kwargs`). A new API-contract test enforces that every public
  frame-first function accepts an explicit `df_name=` / `name=`;
  `get_variable_name` inspection remains a last-resort fallback only
  (verified across all call sites).
- **DRY (audit #5):** `now_iso` / `append_audit` now live only in `_utils`;
  the six local copies (`_compose`, `_features_common`, `cleaning`,
  `dashboard`, `modeling`, `selection`) were removed, and `cleaning`'s unused
  `_append_audit` / `_ensure_audit` dead code deleted. No behaviour change.
- **Unified contract — Phase 1 (audit #4):** `describe_numeric`,
  `plot_histograms` and `plot_boxplots` now expose the full standard flag set
  (`params`, `return_params`, `show`, `plot`, `return_df`, `return_fig`,
  `decimals`, `df_name`), attach a `dextra_audit` trail to their summary
  frame, and return a JSON-safe `params` manifest when `return_params=True`.
  Existing return order is preserved (plotters stay figure-first; a bare call
  still returns `None`); the manifest is appended last. For the plotters
  `plot=True` simply gates rendering; for `describe_numeric` `plot` /
  `return_fig` are documented no-ops (it has no native figure).
- **Unified contract — Phase 2 (audit #4 complete):** all 22 `stats_advanced`
  functions now expose the full standard flag set, attach a `dextra_audit`
  trail and return a JSON-safe `params` manifest when `return_params=True`.
  Existing return order and the special escape-hatch returns
  (`return_zscores`, `return_p`, `return_residuals`, `return_test`,
  `return_rows`) are preserved; a bare call still returns `None`. The
  inference/test helpers are pure functions of their explicit arguments, so
  `params=` is accepted and recorded but there is nothing to replay.
- **Coverage gate raised 68 -> 78** in `run_validation.ps1` and all CI test
  jobs, locking in the coverage sprint (total 76.43% -> 81.20%; cleaning.py
  61% -> 98% via the new visual/show-path + strategy-matrix tests).

### Deprecated
- `name=` is deprecated across the Phase 2 single-label helpers
  (`confidence_interval_mean`, `confidence_interval_proportion`,
  `normality_test`, `t_test_one_sample`, `class_imbalance`); pass `df_name=`
  instead. `name=` still works and maps onto `df_name`, emitting a
  `DeprecationWarning`. (The two-label helpers keep `name1`/`name2` and
  `name_before`/`name_after`, which are group labels, not the frame name.)
- **Phase 3 alias slimming (audit #10):** each cleaning function now has ONE
  official short alias -- `audit`, `tidycols`, `recast`, `verify`, `impute`,
  `dedup`, `winsor` (plus the inspectors `nascan` / `dupscan` / `outscan`).
  The older synonyms (`cleanrep`, `clean_rep`, `stdcols`, `col_clean`,
  `col_fix`, `cast`, `type_fix`, `vrules`, `rule_check`, `fillna_smart`,
  `na_fix`, `dup_fix`, `clipout`, `out_fix`) still work and resolve to the
  same function objects via a module `__getattr__` (identity checks keep
  holding), but emit a `DeprecationWarning` and were removed from `__all__`.
  In-library messages and the docs now recommend the official names.

### Fixed
- **Leading-zero identifiers survive the loader (eval B-1, Blocker):**
  `load`'s measured inference used to coerce pure-digit columns into
  datetime ("001" -> year 1) or int64 ("01001" -> 1001), silently dropping
  leading zeros at parse_rate=1.0 and marking the decision `confirmed` --
  so no `on_ambiguous` policy could catch the corruption (the high-risk
  path only fired when some values failed to parse). Two guards now apply:
  pure-digit strings are never tried as datetime (no date separator means
  no date), and a pure-digit column containing at least one leading-zero
  value ("001", "01001", "0512345678") is kept as text and flagged
  `ambiguous-high-risk` (id/key/target-like name) or `ambiguous`, so
  `warn`/`raise`/`plan` all surface it. Lossless conversions are
  untouched: digit columns without leading zeros still become int64
  `confirmed`, real dates with separators still parse, and a lone "0" is
  still a number. Regression tests:
  `tests/test_loader_b1_leading_zeros.py`.
- **Arabic / legacy encodings no longer corrupted silently (eval M-6):**
  encoding auto-detection used to stamp a detector's first guess
  `confirmed`: legacy cp1256 Arabic came back as cp932/mac_latin2 mojibake,
  and even valid utf-8 Arabic could come back as gb18030 -- with no warning
  under any policy. Detection now tries strict utf-8 first (confirmed only
  when the bytes really decode, tolerating a multi-byte char cut at the
  sample boundary), prefers cp1256 when non-utf-8 bytes decode to
  Arabic-looking letter runs, and reports **every** non-utf-8 guess
  (including charset-normalizer's) as `ambiguous`, so
  `on_ambiguous="warn"/"raise"/"plan"` all surface it. `encoding=` is
  the documented deterministic way out and stays `confirmed`. Regression
  tests: `tests/test_loader_m6_encoding.py`.
- **Audit trail surfaced (audit #6):** `edareport` / `dash` appended their
  audit entry to an internal copy that was immediately discarded, so the
  trail was unreachable. The manifest returned with `return_params=True` now
  carries `dextra_audit` (the input frame's history plus this run's entry)
  as its last key. The input DataFrame stays unmutated and all other return
  values are unchanged.
- **Clusterer fit labels (audit #13):** `DextraClusterer.labels_` (and so
  `fit_predict`) now holds the labels the underlying fit actually assigned
  (sklearn clusterer convention) instead of re-predicting the training data
  through the persisted estimator. For `method="agglomerative"` the deployed
  predictor is a nearest-centroid stand-in whose re-predictions can disagree
  with the fit labels on boundary points; `predict` on *new* data still uses
  it, unchanged. The stand-in now also carries the fit labels as `labels_`.

### Docs
- `docs/api.md` now documents the Phase 11 loader (`load` / `peek`): the five
  source families, the `on_ambiguous` policy, the replayable plan, the
  security model and the audit trail. `getting-started` and `index` lead with
  `dx.load(...)` as the entry point.
- **`df.eval` trust assumption (audit #12):** the `validate_rules` docstring
  now states that string `check` rules run through `df.eval` and callables
  receive the full frame -- rules are code and must come from a trusted
  source, never from untrusted end-user input. No functional change.

## [0.4.0] — 2026-06-07 — Phase 11 complete: the smart loader

`dx.load` / `dx.peek` now cover csv/tsv + Excel + parquet + JSON/NDJSON +
safe SQL under one contract: full disclosure, measured type inference,
an `on_ambiguous` policy, and a replayable JSON-safe load plan.

### Release chores

- Delimiter stability check is now quote-aware and ignores preamble rows
  (closes the known 11.1 cosmetic limit: no more false "delimiter
  ambiguous" warnings on quoted commas / title rows).
- CI: new `perf-extras` job (pyarrow + polars path, audit #8) and a
  non-blocking `pandas-matrix` job across pandas 2.1/2.2/3.0 with matching
  numpy pins (audit #8b).

### Phase 11.3b — safe SQL (parametrised only, read-only, row guard)

#### Added
- `dx.load` / `dx.peek` run **safe SQL**: one parametrised statement only
  (stacked statements refused; values bound via `sql_params=` — named
  `:name` + dict or positional `?` + tuple — never string-formatted).
  SQLite files (`.db` / `.sqlite` / `.sqlite3`) are opened **read-only**
  (URI `mode=ro`); caller DB-API connections are accepted with read-only
  disclosed as not enforced. Results are capped by `max_rows=` or a default
  1,000,000-row guard (truncation disclosed; default-guard hits are flagged
  ambiguous under `on_ambiguous`). SQL errors surface as clear
  `DextraLoaderError`s; plans are replayable (`load(src, params=plan)`
  re-executes the stored statement and re-applies dtypes).
- Shared disclosure tail `_disclose_and_finish` (report → policy → audit)
  now serves every source kind — one contract, zero duplication.
- 12 SQL tests in `tests/test_phase11.py`.

### Phase 11.3a — parquet + JSON/NDJSON

#### Added
- `dx.load` / `dx.peek` read **parquet** (`.parquet` / `.pq`): typed
  pass-through (already-typed columns confirmed at parse_rate 1.0; text
  columns re-inferred measurably) via a lazy engine (pyarrow — `perf`
  extra), with `max_rows` guard and replayable plans.
- `dx.load` / `dx.peek` read **JSON/NDJSON** (`.json` / `.jsonl` /
  `.ndjson`): records-array and one-object-per-line forms detected and
  disclosed (`json_form` decision; a lone top-level object loads as one
  record and is flagged ambiguous), unparsable lines skipped + counted,
  nested object/array values serialised to JSON strings + disclosed per
  column, `na_values` / `max_rows` honoured, plans replayable.
- 11 new tests in `tests/test_phase11.py`.

### Phase 11.2 — Excel (xlsx/xlsm)

#### Added
- `dx.load` / `dx.peek` read Excel (`.xlsx` / `.xlsm`) via openpyxl, lazily
  (`io` extra): sheet listing + selection (`sheet=` name or 0-based index;
  several sheets -> flagged ambiguous under `on_ambiguous`), data-block
  detection (leading empty rows/columns skipped and disclosed), multi-row
  merged headers detected and combined into `top_bottom` names
  (`header_rows=` to force), **values not formulas** (cached values only;
  macros never executed), native cell types honoured (dates / booleans /
  numbers confirmed at parse_rate 1.0) with measured inference for text
  columns, and fully replayable Excel load plans. Legacy `.xls` is refused
  with clear guidance. New `load` params: `sheet=`, `header_rows=`.
- `openpyxl>=3.1` added to the `io` extra (and to `dev` so CI exercises it).
- 14 Excel tests in `tests/test_phase11.py`.

#### Fixed
- `load`/`peek`: source-kind detection now uses the real file name; `peek`
  used to mis-detect the kind because the inferred display name replaced
  the path before the extension check.

## [0.3.0] — 2026-06-07 — Phase 11.1 (loader, entry layer) + release fixes

### Added
- `dx.load` / `dx.peek` (aliases `dload` / `dpeek`): a smart, transparent,
  replayable loader for messy CSV/TSV — encoding + delimiter + header
  detection, measured per-column type inference, categorical confidence
  with reasons, a JSON-safe replayable **load plan**, and an
  `on_ambiguous` policy (`warn` default / `raise` / `plan`).
- Security by default: pickle sources refused unless `allow_pickle=True`.
- New optional extra `io` (charset-normalizer, clevercsv), lazy with
  pure-stdlib fallbacks; added to `dev` so CI exercises the high-quality path.
- Shared contract helpers `now_iso` / `append_audit` / `json_safe` in
  `dextra._utils` (single source of truth; addresses audit finding #5).
- `tests/test_phase11.py`.

### Changed
- `dx.dash`: default sidecar `data_format` is now `"auto"` — resolves to
  `parquet` when an engine (pyarrow/fastparquet) is importable, else `csv`.
  Pickle is no longer the default.

### Security
- `dx.dash(data_format='pickle')` is opt-in and emits a `UserWarning`:
  pickle sidecars can execute arbitrary code when loaded.

### Removed
- `src/dextra/stats_advanced.py.bak` no longer tracked/shipped
  (`*.bak` added to `.gitignore`).

### Phases 7–10 — evaluation, time series, report, dashboard (shipped in 0.3.0; were previously stranded in a second Unreleased section)

#### Added
- **Phase 7 - `evaluation`:** four deep, multi-metric evaluation functions
  (each with a short alias), built on the Phase-6 hybrid artifact and the same
  one-line contract (dense metrics table + multi-panel figure + `Decision:`
  sentence + audit trail). Each accepts BOTH a label mode (`y_true`/`y_pred`/
  `scores`) AND an artifact mode (a Phase-6 `params` dict); `return_params=True`
  yields a JSON-safe evaluation report descriptor (no estimator).
  - `confusion_report` (`confrep`) - per-class precision/recall/F1/support plus
    accuracy and macro/weighted averages; confusion-matrix + row-normalised
    (recall) + per-class-F1 panels.
  - `roc_pr` (`rocpr`) - per-class ROC-AUC and average-precision (binary and
    multiclass one-vs-rest, plus macro); ROC + Precision-Recall curve panels.
  - `residual_analysis` (`residan`) - residual mean/std/skew/kurtosis, R2/RMSE/
    MAE, Durbin-Watson, a heteroscedasticity hint and a Jarque-Bera normality
    p-value; residual-vs-fitted + distribution + Normal Q-Q + scale-location
    panels (scipy.stats lazily; no statsmodels).
  - `learning_curves` (`learncv`) - train vs cross-validated score across
    training sizes with a bias/variance verdict; learning-curve + gap panels.
    Re-fits on CV folds of subsets by design (the only Phase-7 re-fit).
  See EVALUATION_PHILOSOPHY.md.
- **Phase 8 - `timeseries` (Stage 8.1):** `tsdecomp` - one-line trend /
  seasonal / residual decomposition on the same contract (dense components
  frame + four-panel figure + `Decision:` sentence + audit trail). Two input
  modes: SERIES (`value` (+ `time`), with the seasonal `period` inferred from a
  datetime index) and ARTIFACT (`params` from a prior call, replayed without
  re-deciding). Classical decomposition (centred 2xm moving-average trend,
  period-averaged seasonal, additive or multiplicative) is dependency-free;
  `method="stl"` uses a lazy statsmodels import. Hyndman trend / seasonal
  strengths reported; the input DataFrame is never mutated. Underscore-free
  public name. See TIMESERIES_PHILOSOPHY.md.
- **Phase 8 - `timeseries` (Stage 8.2):** `tsstat` - one-line stationarity
  diagnosis. Runs the ADF (null: unit root) and KPSS (null: stationarity) tests
  via a lazy statsmodels import, reports both with the classic four-case
  verdict, and suggests a differencing order `d` by differencing until ADF
  rejects a unit root AND KPSS fails to reject stationarity (capped at
  `max_diff`). Two-row test table + three-panel figure (series + rolling mean,
  rolling std, a dependency-free ACF) + `Decision:` sentence + audit trail;
  series / artifact modes; JSON-safe descriptor. Underscore-free public name.
- **Phase 8 - `timeseries` (Stage 8.3):** `tsfcast` - one-line baseline
  forecast. Trains `naive` / `snaive` / `drift` / `mean` on every point before a
  held-out tail, scores on that tail (MASE / RMSE / MAE / MAPE; no look-ahead),
  then re-fits on the full series to project `horizon` steps with an approximate
  95% band. `method='auto'` picks seasonal-naive when a period is available else
  naive; `method='compare'` ranks every baseline by MASE and writes no artifact.
  Two-panel figure, series / artifact modes, JSON-safe descriptor, dependency-
  free. **Phase 8 complete: 3/3 time-series functions.**
- **Phase 9 - `report`:** `edareport` (`edarep`) - a one-call,
  self-contained HTML EDA report that composes the tested functions of Phases
  1-8 (it computes nothing new). Sections: Overview, Data quality (missing /
  duplicates / outliers), Univariate (numeric summary + histograms + top
  categorical frequencies), Bivariate (correlation + class balance), and an
  optional target-aware Baseline model & evaluation section
  (`include_model=True`, lazy scikit-learn). Figures are embedded as base64
  PNGs and tables inline -- one portable file, no sidecar assets, no new
  dependency (PDF export deferred to an optional `report` extra). Sections are
  isolated (a failing section is skipped with a reason; the report still
  renders); the input DataFrame is never mutated; `return_params=True` yields a
  JSON-safe build manifest. Underscore-free public name. See
  REPORT_PHILOSOPHY.md.
- **Phase 10 - `dashboard` (final goal):** `dash` (`dashapp`) - generates a
  self-contained interactive Streamlit app (`dashboard_app.py` + a sidecar data
  file) that re-runs dextra's analyses live under sidebar controls. It computes
  nothing new: every tab is a Phase-9 section builder (Overview / Data quality /
  Univariate / Bivariate / optional Model), rendered to Streamlit widgets.
  Streamlit is the optional, lazy `dash` extra (the renderer is testable with a
  stub, so the base install and CI are unaffected); `launch=False` by default
  (generate only); the input DataFrame is never mutated; tabs are isolated;
  `return_params=True` yields a JSON-safe manifest. Underscore-free public name.
  See DASHBOARD_PHILOSOPHY.md. **This completes the 10-phase Roadmap.**
- **Phase 9/10 hardening:** the report/dashboard section builders were
  extracted to a neutral `dextra._compose` layer -- a single source of truth
  that the HTML renderer (`edareport`) and the Streamlit renderer (`dash`) both
  sit on (the dashboard no longer depends on the report module). `dash` gained
  `output_dir=` (collect app + data + metadata in one folder), a `parquet`
  `data_format` (lazy engine, clear error if absent), a `*_meta.json`
  reproducibility manifest (dextra / Python / pandas versions + settings), and
  up-front dependency / data-file checks in the generated app.

#### Testing
- `tests/test_phase7.py` - covers all four evaluation functions in both label
  and artifact modes (binary + multiclass), the JSON-safe report descriptor,
  figure rendering, input-DataFrame immutability, alias identity and the
  guard-error paths, restoring the coverage gate.
- `tests/test_phase8.py` - covers `tsdecomp` in series and artifact modes,
  classical additive + multiplicative reconstruction, period inference, the
  JSON-safe descriptor, figure rendering, immutability, idempotency and the
  guard-error paths; the STL path is skipped when statsmodels is absent.
- `tests/test_phase8.py` (Stage 8.2) - covers `tsstat` on white-noise
  (stationary, `d=0`) and random-walk (needs differencing) series, the
  four-case verdict, the JSON-safe descriptor, immutability, artifact mode,
  figure rendering and the guard-error paths; the ADF / KPSS tests are skipped
  when statsmodels is absent.
- `tests/test_phase8.py` (Stage 8.3) - covers `tsfcast` baselines (exact
  naive / snaive / drift / mean forecasts), `auto` resolution, the held-out
  validation metrics, the `compare` leaderboard (and its no-artifact rule), the
  forward datetime / integer index, immutability, artifact mode, figure
  rendering and the guard-error paths. 60 Phase-8 tests in total.
- `tests/test_phase9.py` - covers `edareport`: a self-contained HTML file
  with embedded figures and no sidecar assets, the JSON-safe manifest, section
  isolation (skipped-but-still-written), immutability, the sections subset, and
  the optional model section (classification + regression; skipped without
  scikit-learn).
- `tests/test_phase10.py` - covers `dash`: a runnable generated app + a
  dtype-preserving sidecar (round-trip), the JSON-safe manifest, immutability,
  the CSV format and the guard paths; and the renderer `_build_dashboard` with a
  stubbed Streamlit (tab rendering, model-tab toggle, tab isolation).

#### Dependencies
- New **optional** extra `ts` (`statsmodels>=0.14`), lazy-imported, for the
  Phase 8 STL decomposition and (upcoming) ADF / KPSS stationarity tests. The
  base install is unchanged; `import dextra` still needs only numpy / pandas /
  matplotlib / seaborn / scipy.
- New **optional** extra `dash` (`streamlit>=1.30`), lazy-imported, for the
  Phase 10 interactive dashboard. The base install is unchanged and CI does not
  require Streamlit (the renderer is tested with a stub).


## [0.2.0] - 2026-06-01

Completes the modeling phase (Phase 6) and a post-Phase-6 hardening sprint:
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
- **Hardening sprint - `dextra.compat`:** scikit-learn-compatible wrappers so dextra's
  pipelines and models drop into `sklearn.pipeline.Pipeline` / `GridSearchCV`:
  `DextraFeaturePipeline`, `DextraSelectPipeline`, `DextraRegressor`,
  `DextraClassifier`, `DextraClusterer` (fit/transform/predict, get/set_params,
  clone-able). scikit-learn stays an optional (`ml`) extra.
- **PEP 561:** the package now ships a `py.typed` marker so downstream type
  checkers consume dextra's annotations.
- **Docs site:** a MkDocs-Material site with an auto-generated API reference
  (mkdocstrings) and a `docs` GitHub Actions workflow that deploys to GitHub
  Pages. `CITATION.cff` and a PyPI version badge added.
- **CI `build` job:** builds the sdist + wheel and runs `twine check` so the
  package stays PyPI-publishable on every push.
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
- **`dx.functions()` now groups the API by phase** (and lists aliases) for
  easier discoverability.
- **`features.py` refactored into focused modules** (`_features_common`,
  `_features_numeric`, `_features_discretize`, `_features_derive`,
  `_features_pipeline`); `dextra.features` is now a thin re-export facade. No
  public API change -- imports and behaviour are identical.

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
- `tests/test_io_backends.py` (polars/pyarrow input) and `tests/test_coverage_boost.py`
  (breadth tests across cleaning / stats / features / modeling) raise coverage.
- `tests/test_api_contract.py` (exports resolve, docstrings present, fit/apply
  families share the 8 standard flags, 55 aliases verified identical) and
  `tests/test_compat_conformance.py` (fit-returns-self, clone, pickle round-trip,
  repr) lock in the cross-module contract and sklearn conformance.

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

[Unreleased]: https://github.com/ahmedabdeltawab602-collab/dextra/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/ahmedabdeltawab602-collab/dextra/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ahmedabdeltawab602-collab/dextra/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ahmedabdeltawab602-collab/dextra/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ahmedabdeltawab602-collab/dextra/releases/tag/v0.1.0