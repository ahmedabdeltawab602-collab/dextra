# dextra Phase 7 — Evaluation Philosophy & Framework

> **Purpose.** This document defines the philosophy, the globally-recognised
> framework, and the design constraints for every evaluation function we add to
> dextra. It is the blueprint for Phase 7. Any evaluation function we ship must
> satisfy this reference; if it does not, we either fix the function or extend
> this document. It is the Phase-7 sibling of `MODELING_PHILOSOPHY.md` and
> inherits the same contract (§4.7 there).

---

## 1. Definition and boundary

**Evaluation = judging an already-trained model in depth, across multiple
metrics and multi-panel diagnostics, so the user knows not just *how well* a
baseline scored but *where and why* it succeeds or fails.**

Evaluation does **not** train a new model. The one apparent exception —
`learning_curves` — re-fits the estimator on progressively larger *subsets* of
the data on purpose: producing the curve *is* its job, not building a model to
keep.

Boundaries with adjacent phases:
- **Phase 5 (selection) — done.** Keeps only the signals that help.
- **Phase 6 (modeling) — done.** Learns a baseline and emits a hybrid artifact
  (JSON descriptor + fitted scikit-learn estimator).
- **Phase 7 (this) — evaluation.** Consumes that artifact (or raw
  predictions) and judges it deeply.
- **Phase 9 (report) — later.** Aggregates these judgements into a one-call
  HTML/PDF report.

dextra evaluation is **diagnosis-first**. Phase 6 answers "what is the floor?";
Phase 7 answers "is this floor trustworthy, for which rows, and what is the
failure mode?" — confusion structure, ROC/PR trade-offs, residual assumptions,
and bias/variance behaviour.

---

## 2. The non-negotiable principles

### 2.1 Two input modes (the Phase-7 adaptation of fit/apply)
Every evaluation function accepts its truth in one of two ways:

- **Label mode.** `df + y_true + y_pred` (plus probability columns / `scores`
  for `roc_pr`). A pure, estimator-free evaluation: the user already has
  predictions from anywhere (dextra, sklearn, a CSV) and just wants the
  diagnosis.
- **Artifact mode.** `df + params`, where `params` is a Phase-6 modeling
  artifact. Evaluation reads `target`, `features`, `pred_col` and the fitted
  `estimator` from the artifact and derives `y_true`, `y_pred` (and, for
  `roc_pr`, `predict_proba` / `decision_function` scores) from `df`.

`learning_curves` additionally accepts a bare `estimator=` (with `y`/`cols`),
because a curve needs an estimator to re-fit on subsets.

### 2.2 No leakage — evaluate, never silently re-fit
In artifact mode the saved estimator is consumed **verbatim**; predictions are
read from `pred_col` when present, otherwise produced by a single `predict`
call. No estimator is re-fitted to manufacture scores (the lone, explicit
exception is `learning_curves`, whose re-fits are on held-out CV folds of
subsets — the honest learning signal, not a leak).

### 2.3 Honesty about the split
When both an in-sample and an out-of-sample signal exist, evaluation reports
them side by side and the `Decision:` sentence names which split each metric
came from. `learning_curves` always separates the training score from the
cross-validated score — that gap *is* the diagnosis.

### 2.4 Immutability
The input DataFrame is never mutated. Evaluation appends nothing to `df`'s
columns; it returns a freshly-built metrics frame and only writes to a *copy*'s
`df.attrs['dextra_audit']`.

### 2.5 Don't decide for the user — surface the trade-off
Evaluation never picks a threshold, never declares a model "good enough". It
exposes the full curve / confusion structure / residual diagnostics and a
single honest `Decision:` sentence; the user owns the verdict.

---

## 3. The four functions and their scope

| Family | dextra function (alias) | Task | Truth it needs | Multi-panel figure |
|--------|--------------------------|------|----------------|--------------------|
| Confusion | `confusion_report` (`confrep`) | classification | `y_true`, `y_pred` | confusion counts + row-normalised (recall) + per-class F1 bars |
| Ranking | `roc_pr` (`rocpr`) | classification | `y_true`, `scores` (proba/decision) | ROC curve(s) + Precision–Recall curve(s) |
| Residuals | `residual_analysis` (`residan`) | regression | `y_true`, `y_pred` | residual-vs-fitted + residual distribution + Normal Q–Q + scale–location |
| Curves | `learning_curves` (`learncv`) | regression / classification | `estimator` + `X` + `y` | learning curve (train vs CV, ±1σ bands) + train–CV gap |

scikit-learn is imported lazily inside these functions (extra `ml`), exactly as
in Phase 5/6. `scipy.stats` (a core dependency) is imported lazily inside
`residual_analysis` for the Q–Q plot and the Jarque–Bera normality test. No new
heavy dependency is introduced (Durbin–Watson is computed directly, without
statsmodels).

---

## 4. The dextra contract — every Phase 7 function must satisfy this

### 4.1 Dual input mode
```python
# Label mode: bring your own predictions
dx.confusion_report(df, y_true='churn', y_pred='churn_pred')

# Artifact mode: hand it the Phase-6 artifact and the data to judge on
_, p = dx.classify(df_train, y='churn', method='forest', return_params=True)
dx.confusion_report(df_test, params=p)     # derives y_true / y_pred from p
dx.roc_pr(df_test, params=p)               # derives proba scores from p
dx.learning_curves(df_train, params=p)     # re-fits p['estimator'] on subsets
```

### 4.2 Universal flags
The standard dextra flags, identical to Phase 6:
`params=None`, `return_params=False`, `show`, `plot`, `return_df`,
`return_fig`, `decimals`, `df_name`, plus `fig_width`, `fig_height`, `dpi`.
Per-function inputs: `y_true`/`y_pred` (confusion, residual), `scores`
(`roc_pr`), `y`/`cols`/`estimator`/`scoring`/`cv`/`train_sizes`
(`learning_curves`).

### 4.3 Universal outputs (the locked return shape)
- A **metrics DataFrame** densely packed with the relevant diagnostics
  (per-class precision/recall/F1/support; per-class AUC/AP; the residual
  diagnostic block; the per-train-size score table).
- A **multi-panel matplotlib figure** (the table above names the panels).
- A printed one-line **`Decision:`** sentence naming the headline metric and
  the split / mode it was measured on.
- An append to a copy's **`df.attrs['dextra_audit']`**.
- When `return_params=True`, a **JSON-safe evaluation report descriptor**
  (NOT a fitted estimator — evaluation builds no model): `function`, `task`,
  `target`, the computed `metrics`, `metadata` (`n`, `input_mode`, classes /
  folds as applicable), `version`, `evaluated_at`. Reusing the `return_params`
  flag (rather than a new `return_report`) keeps the standard-flags contract in
  `tests/test_api_contract.py` green.

### 4.4 Return packing order
Identical to Phase 6 via the shared `_ret_pack`: `df_or_report`, `params`,
`fig`. Here the leading element is the **metrics report frame** (evaluation
appends no prediction column), so `return_df=True` yields the diagnostics
table.

### 4.5 Idempotency
Evaluating the same inputs twice yields an identical report and figure
(`learning_curves` fixes `random_state` so its CV is reproducible).

---

## 5. Anti-patterns we will refuse

| Anti-pattern | Why it is fatal |
|---|---|
| Re-fitting the estimator to fabricate "test" scores | That is modeling leakage wearing an evaluation mask. |
| Reporting a single accuracy number for an imbalanced problem | Hides the minority-class failure; confusion + PR exist for exactly this. |
| Picking the classification threshold for the user | Violates "don't decide, give choices"; we plot the whole ROC/PR curve. |
| Reading residual normality off a histogram alone | Q–Q + a numeric normality test are required; eyeballing misleads. |
| Showing only the final CV score, hiding the curve | The bias/variance signal lives in the *shape*, not the endpoint. |
| Mutating `df` (e.g. adding a residual column to the input) | Evaluation is read-only; it returns a new frame. |

---

## 6. Where evaluation sits in the professional pipeline
```
Cleaned (P3) -> Engineered (P4) -> Selected (P5)
        |
   Train / Test split
        |
   FIT baseline on train  ->  hybrid artifact (JSON + fitted estimator)   <- Phase 6
        |
   PREDICT on test
        |
   DEEP EVALUATION on test:                                                <- Phase 7
     confusion_report / roc_pr  (classification)
     residual_analysis          (regression)
     learning_curves            (bias / variance, both tasks)
        |
   One-call report (HTML/PDF)                                              <- Phase 9
```

---

## 7. Reference tools we draw from

| Tool | Idea we adopt |
|------|----------------|
| **scikit-learn** | `confusion_matrix`, `precision_recall_fscore_support`, `roc_curve`, `precision_recall_curve`, `learning_curve`; the estimator is consumed, never hidden. |
| **Yellowbrick** | The visual-diagnostics mindset: ROC/PR, residual plots, and learning curves as first-class multi-panel figures. |
| **statsmodels** | The diagnostic checklist for residuals (normality, autocorrelation via Durbin–Watson, heteroscedasticity) — reproduced dependency-free. |
| **scipy.stats** | `probplot` (Q–Q) and `jarque_bera` (residual normality). |

---

## 8. Phase 7 staged delivery plan
Four staged deliveries, each verified before the next begins:

| Stage | Function | Family |
|-------|----------|--------|
| 7.1 | `confusion_report` (`confrep`) | classification — confusion structure |
| 7.2 | `residual_analysis` (`residan`) | regression — residual diagnostics |
| 7.3 | `roc_pr` (`rocpr`) | classification — ROC / PR ranking |
| 7.4 | `learning_curves` (`learncv`) | both — bias / variance curves |

Every public name is underscore-free and short; each has a 3–8 letter alias,
consistent with Phases 2–6.

---

## 9. The golden test for any Phase 7 function
A function passes the philosophy when:
1. It accepts BOTH label mode (`y_true`/`y_pred`/`scores`) AND artifact mode
   (`params`), and derives truth correctly from each.
2. It never re-fits to manufacture scores (only `learning_curves` re-fits, and
   only on CV folds of subsets, by design).
3. It reports the diagnostic densely as a DataFrame AND as a multi-panel figure.
4. The original DataFrame is never mutated.
5. In-sample and out-of-sample signals, when both exist, are labelled distinctly.
6. The decision sentence names the headline metric and the split / mode.
7. `return_params=True` returns a JSON-safe report descriptor (no estimator).
8. Re-running on identical input reproduces an identical report.

Any function violating any of these is rewritten before shipping.

---

## Update history
| Date       | Note                                                          |
|------------|----------------------------------------------------------------|
| 2026-06-05 | Document created as Phase 7 blueprint. 4 functions confirmed: confusion_report / roc_pr / residual_analysis / learning_curves (+ confrep / rocpr / residan / learncv). Two-input-mode contract (label / artifact) locked; return shape inherits Phase-6 §4.7 (report descriptor replaces the fitted-estimator artifact). |
