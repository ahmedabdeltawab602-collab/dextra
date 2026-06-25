# dextra Phase 4 — Feature Engineering Philosophy & Framework

> **Purpose.** This document defines the philosophy, the globally-recognised
> framework, and the design constraints for every feature-engineering function
> we add to dextra. It is the blueprint for Phase 4. Any feature-engineering
> function we ship must satisfy this reference; if it does not, we either fix
> the function or extend this document.

---

## 1. Definition and boundary

**Feature engineering = transforming cleaned data into representations that
make a model closer to the truth without inventing the truth.**

Boundaries with adjacent phases:
- **Phase 3 (cleaning) — already done.** Recovers truth from corrupted data.
- **Phase 4 (this) — features.** Creates new signals from clean data.
- **Phase 5 (selection) — next.** Keeps only signals that actually help.
- **Phase 6 (modeling) — later.** Learns from selected signals.

A function belongs to Phase 4 if its output is **a new column or transformed
column intended to feed a model**, and **not** a quality fix.

---

## 2. The six non-negotiable principles

### 2.1 Fit / Transform separation — THE foundational principle

```
fit(X_train)              → learns parameters (mean, std, encoding map, quantiles)
transform(X)              → applies the learned parameters to ANY data
fit_transform(X_train)    → both, on training data only
```

Hard rule: parameters are learned **only on training data** and applied
verbatim to test / production data. Computing the mean on train+test together
is data leakage.

### 2.2 Train/test split must happen BEFORE feature engineering

```
Load → Clean (Phase 3) → SPLIT ← the inviolable boundary
                          ↓
                 fit on train → transform train and test
```

Functions in dextra Phase 4 must make this easy and obvious, not optional.

### 2.3 No data leakage — three classic leaks to refuse

| Leak | Where it sneaks in | How dextra prevents it |
|------|---------------------|-------------------------|
| Statistic leak  | `scaler.fit_transform(all_data)` then split    | Functions expose `params` so you must opt in to reuse stats. |
| Target leak     | Target encoding without held-out folds          | `encode_categorical(method='target')` requires `y` and uses K-fold by default. |
| Temporal leak   | Lag/aggregation features that peek at the future | Aggregation functions take an `as_of` column or refuse without it. |

### 2.4 Pipelines are reproducible artifacts, not scripts

Every transformation chain must be:
- Saveable (`joblib.dump`)
- Loadable in production
- Applied verbatim to new rows
- Versioned (the params dict carries a `version` field)

### 2.5 Documentation per feature

Every feature created has:
- A semantically meaningful name (not `f7`)
- A creation rule (the params dict captures this)
- An expected range / dtype
- Provenance (which raw columns it came from)

### 2.6 Post-transform validation

After every transformation:
- No new Inf or unexpected NaN
- Cardinality didn't explode (One-Hot didn't create 10 000 columns)
- Distribution sanity (no value occupies > 99% of the column)
- VIF didn't jump (we already have `dx.vif_scores` from Phase 2)

---

## 3. The universal taxonomy of feature engineering

Industry-standard categorisation, mapped to our 7 dextra functions:

| # | Category               | What it does                                    | dextra function          |
|---|------------------------|-------------------------------------------------|--------------------------|
| 1 | Numerical transforms   | Reshape distributions: log, sqrt, Box-Cox, Yeo-Johnson | `transform_numeric`     |
| 2 | Scaling                | Place values on a common scale (z, MinMax, robust) | `scale_numeric`         |
| 3 | Binning                | Discretise continuous values into ordered bins  | `bin_numeric`            |
| 4 | Categorical encoding   | Turn categories into numbers without losing info | `encode_categorical`     |
| 5 | Temporal features      | Extract calendar + cyclical signals from datetimes | `datetime_features`     |
| 6 | Interactions           | Cross columns: ratio, product, difference, polynomial | `interaction_features`  |
| 7 | Aggregations (groupby) | Group statistics, with leakage warnings         | `aggregate_features`     |

Out of scope for Phase 4:
- **Text/NLP features** (separate domain, deferred)
- **Image features** (separate domain, deferred)
- **Spatial / geo features** (separate domain, deferred)
- **Feature selection** (Phase 5: `selection.py`)

---

## 4. The dextra contract — every Phase 4 function must satisfy this

### 4.1 Dual mode: fit-and-transform OR apply-with-params

```python
# Mode A: fit and transform together (on training)
df_train_fe, params = dx.scale_numeric(df_train, cols=['price'], return_params=True)

# Mode B: apply pre-learned params to new data (on test or production)
df_test_fe = dx.scale_numeric(df_test, params=params)
```

When `params` is supplied, the function **does not re-fit** — it only applies.
This is the technical safeguard against leakage.

### 4.2 Universal flags

Every Phase 4 function must accept:
- `df` (DataFrame), `cols` (subset selector)
- `params=None` (apply mode if provided)
- `return_params=False` (toggle for retrieving the learned dict)
- The standard dextra flags: `show`, `plot`, `return_df`, `return_fig`,
  `decimals`, `df_name`

### 4.3 Universal outputs

Every Phase 4 function must produce:
- A new DataFrame (immutable; original untouched)
- A `params` dict carrying everything needed to reproduce the transform
- A printed before/after table
- A multi-panel figure (distribution before/after, or schema diff)
- A one-line `Decision:` sentence
- An append to `df.attrs['dextra_audit']`

### 4.4 The params dict shape

```python
{
    "function": "scale_numeric",
    "method": "standard",
    "version": "0.1.0",
    "fit_at": "2026-05-18T20:31:55Z",
    "columns": {
        "price": {"mean": 100.5, "std": 14.3},
        "age":   {"mean": 35.2,  "std": 8.7},
    },
    "metadata": {...},  # function-specific extras
}
```

The same dict is JSON-serialisable so it can be saved with `json.dump` and
applied on a different machine / different day.

### 4.5 Idempotency under apply mode

```python
dx.f(df_test, params=p)   # transforms
dx.f(_, params=p)         # applying again must be a no-op or identical
```

This is verified for every function in its test script.

---

## 5. Anti-patterns we will refuse

| Anti-pattern | Why it is fatal |
|---|---|
| `scaler.fit_transform(full_data)` then split  | Train statistics polluted by test data. |
| Target encoding without held-out folds        | Direct target leak; model overfits to training rows. |
| `pd.get_dummies(X_train); pd.get_dummies(X_test)` | Column sets may differ; production fails silently. |
| One-Hot on cardinality > 50 without warning   | Curse of dimensionality; memory blow-up. |
| Binning thresholds chosen on full dataset     | Bin edges shift between train and test. |
| `log(x)` on a column containing zeros / negatives | Silent NaN / Inf; downstream model degenerates. |
| Lag feature without an `as_of` timestamp      | Temporal leak; model "knows the future". |

---

## 6. The professional pipeline architecture

```
Cleaned Data (Phase 3 output)
        |
        v
   Train / Test split
        |
   +────┴────+
   |         |
   X_train   X_test
        |
   FIT all transformations
        |
   ┌────┴────┐ params (saved to disk)
   |
   v
   X_train_fe ──→ TRANSFORM (same params) ──→ X_test_fe
        |                                            |
        +────────────────┬───────────────────────────+
                         v
                  Feature Store (versioned)
                         |
                         v
                  Model training (Phase 6)
                         |
                         v
                  Production: load params → transform → predict
```

In dextra, the params dict plays the role of the saved transformer; the user
chooses how to persist it (json, pickle, parquet).

---

## 7. Reference tools we draw from

| Tool                  | Idea we adopt                                                      |
|-----------------------|---------------------------------------------------------------------|
| **scikit-learn**      | Fit / transform separation; ColumnTransformer composition           |
| **feature-engine**    | DataFrame-in / DataFrame-out semantics; named columns preserved     |
| **category-encoders** | The full encoding taxonomy (Target, WoE, Frequency, Hash)           |
| **Feature Store** (Tecton/Feast) | Params are versioned artifacts; the single source of truth |
| **AutoFeat / FeatureTools** | Aggregation primitives; cross-table feature primitives        |
| **scikit-lego**       | The "pipeline as code" philosophy                                   |

dextra Phase 4 does not try to replace any of them. It picks the most-used
piece each one gets right and ships it as a one-line call.

---

## 8. Phase 4 staged delivery plan

Three staged deliveries, each tested before the next begins:

| Stage | Functions                                       | DAMA / category    |
|-------|-------------------------------------------------|---------------------|
| 4.1   | `transform_numeric`, `scale_numeric`            | Numerical transforms + Scaling |
| 4.2   | `bin_numeric`, `encode_categorical`             | Binning + Encoding |
| 4.3   | `datetime_features`, `interaction_features`, `aggregate_features` | Temporal + Crosses + Aggregations |

After Stage 4.3: a `feature_pipeline` convenience wrapper that chains the
seven feature functions -- and, since M-5, optionally the two leakage-prone
cleaning steps `handle_missing` / `clip_outliers` -- and saves/loads the
combined params dict.

---

## 9. The golden test for any Phase 4 function

A function passes the philosophy when:

1. It has both `fit_transform` and `apply` modes.
2. Its params dict is JSON-serialisable.
3. Applying with saved params reproduces the same transform on different data.
4. The original DataFrame is never mutated.
5. Its decision sentence explicitly states what kind of feature was produced
   and how many columns / rows it affected.
6. Idempotency holds under apply mode.
7. It refuses or warns when it detects a known leakage shape (e.g. target
   encoding without `y` or without folds).

Any function violating any of these is rewritten before shipping.

---

## Update history

| Date       | Note                                                              |
|------------|-------------------------------------------------------------------|
| 2026-05-18 | Document created as Phase 4 blueprint. 7 functions confirmed.    |
| 2026-05-18 | Fit/transform contract locked. Staged delivery plan set.         |
