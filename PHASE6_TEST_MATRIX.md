# Phase 6 Stage 6.1 (`regress`) — Contract & Edge-Case Test Matrix

> Evidence that the unified return contract holds **and** that per-model
> scientific accuracy was not sacrificed for uniformity. Suite:
> `tests/test_phase6_stage1.py` — 21 test functions / 25 parametrised cases.
> Run on real scikit-learn via `run_phase6_tests.ps1` or CI (`[dev,ml]`).

## 1. Contract coverage

| Test | Category | What it proves |
|------|----------|----------------|
| `test_exports_and_alias` | API | `regress` + `reg` exported; alias identity |
| `test_fit_each_method` (×5) | Fit | linear/ridge/lasso/tree/forest each fit; artifact shape; **descriptor JSON-serialisable**; estimator exposed |
| `test_fit_audit_entry` | Audit | `dextra_audit` entry with stage=modeling, mode=fit |
| `test_apply_reproduces_and_is_idempotent` | Apply | predict-only; idempotent; **no re-fit** (estimator identity preserved); leakage-safe |
| `test_apply_missing_feature_raises` | Apply guard | KeyError when a fitted feature is absent |
| `test_apply_wrong_params_raises` | Apply guard | ValueError on a non-regress artifact |
| `test_compare_writes_nothing` | Compare | df unchanged, no pred column, audit mode=compare |
| `test_compare_with_return_params_raises` | Compare guard | ValueError (compare fits no single model) |
| `test_non_numeric_target_raises` | Guard | categorical target rejected → directs to `classify` |
| `test_bad_method_raises` | Guard | unknown method rejected before sklearn import |
| `test_standardize_override` | Option | explicit `standardize` honoured in metadata |
| `test_plot_returns_figure` | Display | `return_fig` returns a Matplotlib Figure |

## 2. Edge cases (imbalance / constant features / NaN propagation)

| Test | Edge case | Expected & verified behaviour |
|------|-----------|-------------------------------|
| `test_constant_feature_does_not_crash` | Zero-variance feature | StandardScaler guards σ=0; fit succeeds, feature kept |
| `test_constant_target_is_safe` | Degenerate (constant) target → R² undefined | **No crash**: metric → `None`, table shows `-`, Decision shows `n/a` |
| `test_skewed_outlier_dominated_target` | Outlier-dominated target (regression analogue of severe imbalance) | Fits and predicts without error |
| `test_nan_in_target_rows_dropped` | NaN in target | rows dropped from fit (`n_train` reduced); all rows still get a prediction |
| `test_all_nan_feature_raises` | Feature entirely NaN | clean ValueError (< 2 usable rows) |
| `test_partial_nan_feature_imputed_in_fit` | Partial NaN feature | NaN rows excluded from fit; in-sample preds complete (train-mean imputation) |
| `test_single_feature` | One feature | works |

NaN policy is explicit and asymmetric by design: **fit** is lenient (drops NaN
rows for training, train-mean-imputes for the in-sample prediction column);
**apply** is strict (raises on NaN features) so production data must be cleaned
(Phase 3) before scoring — no silent imputation in production.

A robustness bug was found and fixed during this audit: degenerate inputs make
R² `None` (JSON-safe NaN), which crashed the `Decision` f-string. Fixed with
`_fmt_metric` (None/NaN → `"n/a"`), covered by `test_fmt_metric_handles_none_and_nan`
and exercised end-to-end (fit + compare) under sklearn-like NaN behaviour.

## 3. No hidden coupling with `classify` / `cluster`

| Test | What it proves |
|------|----------------|
| `test_metrics_table_is_family_agnostic` | `_metrics_table` renders regression keys (R2/RMSE/MAE), classification keys (ACCURACY/F1/ROC_AUC), and clustering keys (SILHOUETTE/N_CLUSTERS) — same renderer, zero code change |
| `test_fmt_metric_handles_none_and_nan` | Shared formatter is value-based, family-independent |

`modeling.py` imports nothing from `classify`/`cluster` (they do not yet exist).
The shared, family-agnostic helpers ready for reuse by 6.2 / 6.3 are:
`_resolve_target`, `_resolve_features`, `_clean_xy`, `_effective_cv`,
`_metrics_table`, `_fmt_metric`, `_descriptor`, `_ret_pack`, `_append_audit`,
`_require_sklearn`.

## 4. Unification vs. scientific accuracy

The unification is **structural, not metric-level**. The contract fixes the
*shape* (`metrics = {split: {metric: value}}`) and the return order, but each
family computes its own scientifically appropriate metrics:

- regress: R² / RMSE / MAE (correct for any regressor — the 5 algorithms
  legitimately share the same regression metrics).
- classify (6.2): accuracy / F1 / ROC-AUC.
- cluster (6.3): silhouette / inertia / n_clusters under a single `fit` split.

`_metrics_table` renders whatever metrics a family supplies; it never imposes
regression metrics on classification or clustering. Standardisation is applied
only to the linear family (auto), never to trees/forests — preserving each
model's correct preprocessing.

## 5. Presentation unification (NaN/None safety across every path)

Audited: `modeling.py` contains **no raw metric formatting** (no
`f"{metric:.4f}"`). Every metric-to-string conversion passes through exactly
one of two sanctioned renderers, and both detect `None` and `NaN` via the same
check:

| Renderer | Context | Missing value renders as | Notes |
|----------|---------|--------------------------|-------|
| `_fmt_table` | tables (fit / apply / compare) | `-` | library-wide convention (also used by selection / features) |
| `_fmt_metric` | prose `Decision:` sentences | `n/a` | reads naturally in a sentence |

The sentinel token differs by context (a `-` cell vs. `R^2=n/a` in prose) but
the **null-handling is unified**: there is no path where a `None`/`NaN` metric
is formatted raw, so no display path can raise.

| Test | Proves |
|------|--------|
| `test_degenerate_target_all_display_paths_safe` | fit + compare + apply `show=True` all safe on a constant target; sentinels present |
| `test_fmt_table_renders_missing_as_dash` | `_fmt_table` collapses both `None` and `NaN` to `-`; real values format normally |
| `test_fmt_metric_handles_none_and_nan` | `_fmt_metric` collapses `None`/`NaN` to `n/a` |

This is the contract-vs-presentation separation in practice: the artifact
stores `None` (JSON-safe NaN); presentation never crashes on it.

## 6. Status of the two lock-in checks

1. **NaN -> sentinel unified across all paths** — DONE & tested (Section 5).
2. **Real-sklearn green CI (constant-target R^2 = NaN -> no divergence)** —
   requires execution on real scikit-learn, outside the offline sandbox.
   Run `run_phase6_tests.ps1` locally, or push to trigger the CI matrix
   (`pip install -e ".[dev,ml]"` then `pytest`). The suite exercises the
   degenerate path on real sklearn and asserts no exception + sentinel
   presence, **without** asserting a version-specific R^2 value — so it is
   robust to sklearn's NaN-vs-0 differences yet fails loudly on any real
   divergence or crash.
