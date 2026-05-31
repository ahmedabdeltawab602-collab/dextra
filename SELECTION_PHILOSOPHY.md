# dextra Phase 5 — Feature Selection Philosophy & Framework

> **Purpose.** This document defines the philosophy, the globally-recognised
> framework, and the design constraints for every feature-selection function
> we add to dextra. It is the blueprint for Phase 5. Any selection function
> we ship must satisfy this reference; if it does not, we either fix the
> function or extend this document.

---

## 1. Definition and boundary

**Feature selection = keeping the subset of existing features that helps a
model, and dropping the rest — without creating any new feature.**

Boundaries with adjacent phases:
- **Phase 3 (cleaning) — done.** Recovers truth from corrupted data.
- **Phase 4 (features) — done.** Creates new signals from clean data.
- **Phase 5 (this) — selection.** Keeps only the signals that actually help.
- **Phase 6 (modeling) — next.** Learns from the selected signals.

A function belongs to Phase 5 if its output is **a DataFrame with the same
rows but fewer columns** (a chosen subset), plus a record of *why* each
column was kept or dropped. Phase 4 *adds* columns; Phase 5 *removes* them.

Why select at all:
- **Less overfitting.** Fewer irrelevant features means less noise to fit.
- **Speed and cost.** Smaller matrices train and serve faster.
- **Interpretability.** A model on 12 features is explainable; on 1 200 it is not.
- **Curse of dimensionality.** Distance and density estimates degrade as
  dimensionality grows relative to sample size.
- **Multicollinearity.** Redundant correlated features destabilise linear
  model coefficients (we already measure this with `vif` from Phase 2).

---

## 2. The non-negotiable principles

### 2.1 Selection is a learned decision — fit on TRAIN only

The set of kept columns is a parameter. It is learned on the training data
and applied verbatim to test / production data.

```
fit(X_train, y_train)   -> learns which columns to keep
transform(X)            -> subsets ANY data to those exact columns
```

Choosing features by inspecting the whole dataset (train + test together)
is **selection leakage**: the held-out score becomes optimistic because the
selector already "saw" the test rows.

### 2.2 Train/test split happens BEFORE selection

```
Load -> Clean (P3) -> Features (P4) -> SPLIT  <- the inviolable boundary
                                        |
                              fit selector on train
                                        |
                       transform train AND test to the kept columns
```

### 2.3 No leakage — the three selection-specific leaks to refuse

| Leak | Where it sneaks in | How dextra prevents it |
|------|---------------------|-------------------------|
| Statistic leak | Variance / correlation / VIF computed on train+test | Fit-mode statistics use the passed `df` only; apply mode never re-computes. |
| Target leak | Univariate scores / importances computed on full data | Target-based functions take `y` and are fitted only on the data given; apply just subsets. |
| Selection-CV leak | Selecting features outside the cross-validation loop | The params dict makes the selection an explicit, inspectable artifact so it can be placed *inside* a CV fold. |

### 2.4 Selection is a reproducible artifact, not a script

Every selection must be saveable (the kept-column list + scores), loadable
in production, applied verbatim to new rows, and versioned.

### 2.5 Don't decide — give choices

Consistent with the whole library: every function exposes a `method='compare'`
mode that **ranks features by every available criterion and writes nothing**,
so the user chooses the cut. The function never silently drops columns the
user did not ask it to drop.

### 2.6 Post-selection validation

After every selection:
- At least one feature survives (never return an empty feature matrix).
- The target column, ID columns, and any user-protected columns are never
  dropped.
- The kept set is stable enough to report (selection that flips on a tiny
  data perturbation is flagged, not hidden).

---

## 3. The universal taxonomy of feature selection

The industry-standard categorisation is three families, mapped to the five
dextra Phase 5 functions:

| Family | What it does | Cost | dextra function |
|--------|--------------|------|-----------------|
| **Filter** | Score features by an intrinsic statistical property, independent of any model. | Cheap | `redundancy` (target-free), `relevance` (target-based) |
| **Embedded** | Read the selection a model performs *while* it trains (L1 zeros, tree importances). | Medium | `importance` |
| **Wrapper** | Search feature subsets, each evaluated by actually training a model. | Expensive | `rfe` |

And one composition tool:

| Tool | What it does | dextra function |
|------|--------------|-----------------|
| **Pipeline** | Chain several selectors; collect one combined, versioned params. | `selectpipe` |

### 3.1 Filter — `redundancy` (no target needed)

Drops features that carry little or duplicated information, judged without
ever looking at the target:
- `variance` — drop quasi-constant features (variance below a threshold).
- `correlation` — within each pair of highly-correlated features, drop one.
- `vif` — iteratively drop the highest-VIF feature until all VIF <= threshold.

### 3.2 Filter — `relevance` (target needed)

Ranks each feature by its univariate association with the target and keeps
the top ones:
- `anova` — ANOVA F-test (numeric feature vs categorical target, or the
  regression F-test for a numeric target).
- `chi2` — chi-squared test (non-negative feature vs categorical target).
- `mutualinfo` — mutual information; captures non-linear dependence.

### 3.3 Embedded — `importance`

Trains one model and reads the selection it implies:
- `tree` — a tree ensemble's `feature_importances_`.
- `l1` — L1-penalised model (Lasso / L1 logistic): coefficients driven to 0.
- `linear` — magnitude of standardised linear-model coefficients.

### 3.4 Wrapper — `rfe`

Recursive Feature Elimination: fit a model, drop the weakest feature(s),
refit, repeat until the requested number of features remains.

### 3.5 Pipeline — `selectpipe`

Chains the selectors above into one ordered recipe, collecting each step's
params into a single JSON-serialisable artifact, exactly as `featpipe` does
for Phase 4.

Out of scope for Phase 5: dimensionality *reduction* (PCA / UMAP build new
axes — that is feature *extraction*, not selection), and automated feature
search across tables.

---

## 4. The dextra contract — every Phase 5 function must satisfy this

### 4.1 Dual mode: fit-and-select OR apply-with-params

```python
# Mode A: fit on training data, learn the kept columns
df_train_sel, params = dx.relevance(df_train, y='churn', method='anova',
                                     keep=10, return_params=True)

# Mode B: apply the learned column set to held-out data
df_test_sel = dx.relevance(df_test, params=params)   # subset, no re-score
```

When `params` is supplied the function **does not re-score** — it only
subsets the DataFrame to `params['kept']`. This is the leakage safeguard.

### 4.2 Universal flags

Every Phase 5 function accepts: `df`, `cols` (candidate-feature selector),
`y` (target, where the family needs one), `params=None`, `return_params=False`,
and the standard dextra flags `show`, `plot`, `return_df`, `return_fig`,
`decimals`, `df_name`.

### 4.3 Universal outputs

- A new DataFrame with the kept columns (immutable; original untouched).
  Non-candidate columns (IDs, the target, untouched columns) pass through
  unchanged — only candidate features are ever dropped.
- A `params` dict carrying everything needed to reproduce the selection.
- A printed kept/dropped summary table with the score behind each decision.
- A multi-panel figure (ranked-score bar, kept-vs-dropped, and a redundancy
  heatmap where relevant).
- A one-line `Decision:` sentence.
- An append to `df.attrs['dextra_audit']`.

### 4.4 The params dict shape

```python
{
    "function": "relevance",
    "method": "anova",
    "version": "0.1.0",
    "fit_at": "2026-05-23T...Z",
    "target": "churn",            # None for target-free redundancy
    "candidates": [...],          # the feature pool considered
    "kept": [...],                # candidate features retained
    "dropped": [...],             # candidate features removed
    "scores": {"feat": 12.4, ...},# the criterion value per candidate
    "metadata": {...},            # function-specific extras + cut rule
}
```

JSON-serialisable, so it can be saved with `json.dump` and applied on a
different machine / day.

### 4.5 Idempotency under apply mode

Subsetting to `params['kept']` twice yields the same DataFrame. Verified for
every function in its test script.

### 4.6 compare mode

`method='compare'` ranks the candidate features by every criterion in the
family and prints the table, but **writes nothing and drops nothing**. It
raises `ValueError` if combined with `return_params` (there is nothing to
reproduce). Identical to the Phase 4 convention.

---

## 5. Anti-patterns we will refuse

| Anti-pattern | Why it is fatal |
|---|---|
| Selecting features on train+test together | Selection leakage; held-out score is optimistic. |
| `SelectKBest` then a train/test split | Same leak — the scores already saw the test rows. |
| Dropping a feature for low correlation with the target | Univariate filters miss features useful only in interaction. |
| One-shot correlation drop without a rule for which of the pair to keep | Non-deterministic, non-reproducible selection. |
| Returning zero features | A degenerate matrix; downstream modelling cannot proceed. |
| Selecting inside, but scaling outside, the CV loop (or vice versa) | Inconsistent preprocessing; leaks across folds. |
| Treating tree importances as ground truth | Importances are biased toward high-cardinality features; report, don't worship. |

---

## 6. Where selection sits in the professional pipeline

```
Cleaned + engineered data (Phase 4 output)
        |
   Train / Test split
        |
   +────┴────+
   |         |
 X_train   X_test
   |
 FIT selector(s) on X_train (+ y_train)
   |
   ┌────┴────┐ params: the kept-column list (saved to disk)
   |
   v
 X_train_sel ──→ SUBSET (same columns) ──→ X_test_sel
   |                                            |
   +────────────────┬───────────────────────────+
                    v
             Model training (Phase 6)
                    |
                    v
       Production: load params → subset → predict
```

In dextra the params dict is the saved selector; the user chooses how to
persist it (json, pickle, parquet).

---

## 7. Reference tools we draw from

| Tool | Idea we adopt |
|------|----------------|
| **scikit-learn** | `VarianceThreshold`, `SelectKBest`, `RFE`, `SelectFromModel` — the canonical selector API. |
| **feature-engine** | `DropConstantFeatures`, `DropCorrelatedFeatures`, `SmartCorrelatedSelection` — DataFrame-in / DataFrame-out, named columns kept. |
| **statsmodels** | VIF as the multicollinearity diagnostic. |
| **Boruta / stability selection** | Selection is a decision to be reported with its evidence, not a silent filter. |
| **mlxtend** | Sequential / wrapper search framed as an explicit, inspectable process. |

scikit-learn is imported lazily, only inside `importance`, `rfe`, and the
`mutualinfo` branch of `relevance` — exactly the pattern used for SciPy in
Phase 4. Filter methods (`variance`, `correlation`, `vif`, `anova`, `chi2`)
need only NumPy / SciPy and work without scikit-learn installed.

---

## 8. Phase 5 staged delivery plan

Three staged deliveries, each tested before the next begins:

| Stage | Functions | Family |
|-------|-----------|--------|
| 5.1 | `redundancy`, `relevance` | Filter (target-free + target-based) |
| 5.2 | `importance`, `rfe` | Embedded + Wrapper (lazy scikit-learn) |
| 5.3 | `selectpipe` | Pipeline wrapper + round-trip integration test |

Every public name is underscore-free and short; each long descriptive name
also gets a short alias, consistent with Phase 2 / Phase 3.

---

## 9. The golden test for any Phase 5 function

A function passes the philosophy when:

1. It has both `fit` and `apply` modes; apply only subsets, never re-scores.
2. Its params dict is JSON-serialisable and lists `kept` / `dropped` / `scores`.
3. Applying saved params reproduces the same column subset on different data.
4. The original DataFrame is never mutated.
5. It never drops the target, never returns zero features, and never drops a
   column outside the declared candidate pool.
6. Idempotency holds under apply mode.
7. `method='compare'` ranks everything and writes nothing.
8. Its decision sentence states how many features were kept vs dropped and on
   what criterion.

Any function violating any of these is rewritten before shipping.

---

## Update history

| Date       | Note                                                          |
|------------|----------------------------------------------------------------|
| 2026-05-23 | Document created as Phase 5 blueprint. 5 functions confirmed. |
| 2026-05-23 | Fit/apply contract locked; 3-stage delivery plan set.         |
