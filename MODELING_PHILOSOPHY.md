# dextra Phase 6 — Modeling Philosophy & Framework

> **Purpose.** This document defines the philosophy, the globally-recognised
> framework, and the design constraints for every modeling function we add to
> dextra. It is the blueprint for Phase 6. Any modeling function we ship must
> satisfy this reference; if it does not, we either fix the function or extend
> this document.

---

## 1. Definition and boundary

**Modeling = learning a function from the selected features to a target (or to
the data's own structure), then producing instant, well-diagnosed baseline
predictions with a single call.**

Boundaries with adjacent phases:
- **Phase 3 (cleaning) — done.** Recovers truth from corrupted data.
- **Phase 4 (features) — done.** Creates new signals from clean data.
- **Phase 5 (selection) — done.** Keeps only the signals that help.
- **Phase 6 (this) — modeling.** Learns from the selected signals.
- **Phase 7 (evaluation) — next.** Judges the learned model in depth.

dextra modeling is **baseline-first**, not a tuning framework. Its job is to
get a defensible, leakage-safe baseline — and a comparison of candidate
algorithms — in one line, so the user knows where the floor is before
investing in heavy tuning. Deep evaluation (ROC/PR, residual analysis,
learning curves) is deliberately deferred to Phase 7.

---

## 2. The non-negotiable principles

### 2.1 Learn on TRAIN only, predict on TEST
The model and any preprocessing it owns are fitted on training data only and
applied verbatim to held-out / production data. Fitting on train+test together
is modeling leakage and inflates every reported score.

### 2.2 Train/test split happens BEFORE modeling
```
Load -> Clean (P3) -> Features (P4) -> Select (P5) -> SPLIT  <- inviolable
                                                       |
                                            fit model on train
                                                       |
                                          predict on train AND test
```

### 2.3 No leakage — the modeling-specific leaks to refuse
| Leak | Where it sneaks in | How dextra prevents it |
|------|---------------------|-------------------------|
| Preprocessing leak | Scaling/encoding fit on full data before modeling | Modeling owns no global fit; it consumes already-split, already-fitted P4/P5 output, or fits strictly on the train df passed. |
| Evaluation leak | Reporting train score as if it were generalisation | Both train and CV/holdout metrics are reported side by side; the decision sentence names which is which. |
| Target leak in clustering | Using the target to build clusters | `cluster` is unsupervised: it never accepts `y`. |

### 2.4 The hybrid reproducible artifact (the Phase 6 adaptation)
Phases 4–5 used a pure JSON `params` dict. A trained estimator is **not**
JSON-serialisable, so Phase 6 uses a **hybrid contract**:
- A JSON-serialisable `params` dict: `function`, `task`, `algorithm`,
  `features`, `target`, `hyperparams`, `metrics`, `version`, `fit_at`,
  `metadata`.
- The **fitted scikit-learn estimator** itself, returned to the user so it can
  be persisted with `joblib`/`pickle` and dropped into an
  `sklearn.Pipeline` / `GridSearchCV`. dextra never hides the estimator.

This closes the long-standing "no sklearn interoperability" gap while keeping
the inspectable JSON descriptor dextra is known for.

### 2.5 Don't decide — give choices
Every family exposes `method='compare'`: it trains every candidate baseline
algorithm, ranks them on a fair cross-validated metric, prints the table and
plots the comparison, **but fits nothing as "the" model and writes no
artifact**. The user picks the algorithm; dextra never silently anoints one.

### 2.6 Post-fit validation
After every fit: at least one feature is used; the estimator converged (or a
warning is surfaced, never swallowed); metrics are computed on the same split
they are labelled with; and the original DataFrame is never mutated.

---

## 3. The three families and their compare scope

| Family | dextra function (alias) | Task | compare candidates (expanded baseline) |
|--------|--------------------------|------|-----------------------------------------|
| Regression | `regress` (`reg`) | numeric target | linear, ridge, lasso, decision-tree, random-forest |
| Classification | `classify` (`clf`) | categorical target | logistic, decision-tree, random-forest, KNN |
| Clustering | `cluster` (`clus`) | no target (unsupervised) | kmeans, agglomerative + k selection (silhouette / elbow) |

scikit-learn is imported lazily inside these functions (extra `ml`), exactly
as in Phase 5's model-based selectors.

---

## 4. The dextra contract — every Phase 6 function must satisfy this

### 4.1 Dual mode: fit-and-model OR apply-with-artifact
```python
# Mode A: fit on training data. Returns the frame with a prediction column
# appended, plus the hybrid artifact. The fitted estimator lives at
# params['estimator'] (persist with joblib; drop into an sklearn Pipeline).
out_train, params = dx.regress(df_train, y='price', method='forest',
                               return_params=True)
# Mode B: apply the fitted model to held-out data (predict, no re-fit)
out_test = dx.regress(df_test, params=params)
```
When `params` (carrying the fitted estimator) is supplied the function
**does not re-fit** — it only predicts. This is the leakage safeguard,
mirroring Phases 4–5 apply mode. Method keys are short and underscore-free:
`linear`, `ridge`, `lasso`, `tree`, `forest` (and `compare`).

### 4.2 Universal flags
`df`, `cols` (feature selector), `y` (target; absent for `cluster`),
`method`, `params=None`, `return_params=False`, plus the standard dextra flags
`show`, `plot`, `return_df`, `return_fig`, `decimals`, `df_name`.

### 4.3 Universal outputs
- The input DataFrame with a prediction column appended (`<target>_pred`), original untouched; the fitted estimator is exposed via `params['estimator']`.
- A hybrid `params` artifact (JSON descriptor + fitted estimator).
- A printed metrics table: train vs cross-validated metrics side by side
  (regression: R²/RMSE/MAE; classification: accuracy/F1/ROC-AUC;
  clustering: silhouette/inertia/n_clusters).
- A multi-panel figure (e.g. predicted-vs-actual + residuals; confusion +
  metric bars; cluster scatter + silhouette/elbow).
- A one-line `Decision:` sentence naming the metric and the split.
- An append to `df.attrs['dextra_audit']`.

### 4.4 compare mode
`method='compare'` trains every candidate with cross-validation, ranks them,
prints the leaderboard and plots it, writes no artifact, and raises
`ValueError` if combined with `return_params`. Identical convention to
Phases 4–5.

### 4.5 Idempotency under apply mode
Predicting with the same fitted artifact twice yields identical output.
Verified in the test script.

---

### 4.7 The formal return contract (locked in 6.1, extensible to 6.2 / 6.3)

The hybrid `params` artifact has a fixed key set. Every family fills the same
keys; only the *values* differ. This is what lets `classify` and `cluster`
build on `regress` without a contract change.

| Key | regress (6.1) | classify (6.2) | cluster (6.3) |
|-----|---------------|----------------|---------------|
| `function` | `"regress"` | `"classify"` | `"cluster"` |
| `task` | `"regression"` | `"classification"` | `"clustering"` |
| `algorithm` | linear/ridge/lasso/tree/forest | logistic/tree/forest/knn | kmeans/agglomerative |
| `features` | feature columns | feature columns | feature columns |
| `target` | target name | target name | `None` (unsupervised) |
| `hyperparams` | algo hyper-params | algo hyper-params | algo hyper-params (incl. k) |
| `metrics` | `{split: {metric: value}}` | same shape | same shape |
| `pred_col` | `"<target>_pred"` | `"<target>_pred"` | `"cluster"` |
| `metadata` | n_train, n_features, cv_folds, ... | same + classes | n, n_features, k, ... |
| `estimator` | fitted sklearn Pipeline | fitted Pipeline | fitted Pipeline |

`metrics` is a mapping of **split-name to a metric dict**, rendered by the
shared `_metrics_table`. The splits vary by family but the renderer does not:

- regress / classify: `{"train": {...}, "cv": {...}}`
- cluster: `{"fit": {"silhouette": ..., "n_clusters": ...}}`

The **return shape** is identical in all three families and was verified across
seven scenarios (explicit cols, `y` as a Series, tiny `n`, NaN handling,
extra/re-ordered columns on apply, every `return_*` flag combination, and
compare):

- fit -> the input DataFrame with the prediction column appended, plus the
  hybrid `params` when `return_params=True` (order: `df`, `params`, `fig`).
- apply -> the input DataFrame with the prediction column appended (no re-fit).
- compare -> the input DataFrame unchanged; never returns `params`.

Shared, family-agnostic helpers already in place for reuse by 6.2 / 6.3:
`_resolve_target`, `_resolve_features`, `_clean_xy`, `_effective_cv`,
`_metrics_table`, `_descriptor`, `_ret_pack`, `_append_audit`, `_require_sklearn`.
`cluster` will add a target-free `_clean_x`; everything else is reused verbatim.

## 5. Anti-patterns we will refuse
| Anti-pattern | Why it is fatal |
|---|---|
| Fitting scaler/encoder/model on train+test together | Modeling/preprocessing leakage; scores are optimistic. |
| Reporting only the training score | Hides overfitting; meaningless as generalisation. |
| Hiding the estimator behind an opaque object | Blocks joblib persistence and Pipeline/GridSearchCV use. |
| Passing `y` to clustering | Turns unsupervised structure-finding into target leak. |
| Auto-selecting "the best" model silently | Violates "don't decide, give choices". |
| Treating a baseline as a tuned final model | Phase 6 is a floor, not a finish line. |

---

## 6. Where modeling sits in the professional pipeline
```
Cleaned (P3) -> Engineered (P4) -> Selected (P5)
        |
   Train / Test split
        |
   FIT model on X_train (+ y_train)         <- Phase 6
        |
   hybrid artifact: JSON descriptor + fitted estimator (saved via joblib)
        |
   PREDICT on X_test  ->  Deep evaluation     <- Phase 7
        |
   Production: load artifact -> predict
```

---

## 7. Reference tools we draw from
| Tool | Idea we adopt |
|------|----------------|
| **scikit-learn** | The estimator API, `cross_val_score`, baseline algorithms; the fitted estimator is exposed, not hidden. |
| **PyCaret / LazyPredict** | One-call training and an algorithm leaderboard (our `compare`). |
| **joblib** | The chosen persistence path for the fitted estimator artifact. |
| **statsmodels** | Diagnostic mindset: report assumptions and fit quality, not just a score. |

---

## 8. Phase 6 staged delivery plan
Three staged deliveries, each tested before the next begins:

| Stage | Function | Family |
|-------|----------|--------|
| 6.1 | `regress` (`reg`) | Regression + compare | ✅ done |
| 6.2 | `classify` (`clf`) | Classification + compare | ✅ done |
| 6.3 | `cluster` (`clus`) | Clustering + compare (k selection) | ✅ done |

Every public name is underscore-free and short; each has a short alias,
consistent with Phases 2–5.

---

## 9. The golden test for any Phase 6 function
A function passes the philosophy when:
1. It has both fit and apply modes; apply only predicts, never re-fits.
2. Its hybrid artifact carries a JSON descriptor AND the fitted estimator,
   persistable with joblib and usable in an sklearn Pipeline.
3. Applying a saved artifact reproduces identical predictions on the same data.
4. The original DataFrame is never mutated.
5. Train and cross-validated/holdout metrics are reported side by side, each
   correctly labelled.
6. `cluster` never accepts a target; `regress`/`classify` never leak it.
7. `method='compare'` ranks every candidate and writes nothing.
8. The decision sentence names the chosen metric and the split it was measured on.

Any function violating any of these is rewritten before shipping.

---

## Update history
| Date       | Note                                                          |
|------------|----------------------------------------------------------------|
| 2026-05-31 | Document created as Phase 6 blueprint. 3 functions confirmed: regress / classify / cluster (+ reg/clf/clus). |
| 2026-05-31 | Hybrid artifact contract locked (JSON descriptor + fitted estimator). Expanded-baseline compare scope set. 3-stage delivery plan set. |
| 2026-05-31 | Stage 6.1 (regress) shipped. Return contract formalised (4.7) and verified across 7 scenarios; metrics standardised to {split:{metric}}; folds moved to metadata. |
| 2026-05-31 | Stage 6.2 (classify) shipped on the same contract: logistic/tree/forest/knn, binary+multiclass, accuracy/F1/ROC-AUC, StratifiedKFold; reused all shared helpers. |
| 2026-06-01 | Stage 6.3 (cluster) shipped — Phase 6 complete: kmeans/agglomerative + compare, automatic k by silhouette (elbow/inertia for kmeans), unsupervised (no y), `pred_col="cluster"`. Added target-free `_clean_x`; reused every shared helper. Agglomerative apply uses a NearestCentroid fitted on its labels so the persisted Pipeline predicts uniformly. |
