# dextra Phase 3 — Data Cleaning Philosophy & Framework

> **Purpose.** This document defines the philosophy, the globally-recognised
> framework, and the design constraints for every cleaning function we add to
> dextra. It is the blueprint. Any cleaning function we ship must satisfy this
> reference; if it does not, we either fix the function or extend this
> document — never silently diverge.

---

## 1. The non-negotiable principles

Data cleaning is **not cosmetic work**. It is the **act of recovering the
truth from corrupted data** without inventing facts. Every line of cleaning
code answers one of these questions: *Is the value correct? Is it present?
Is it consistent? Is it within the allowed range? Is the same entity
represented only once? Is the data current?*

### 1.1 The six DAMA-DMBOK data quality dimensions (industry canonical)

| Dimension     | Question it answers                                             |
|---------------|-----------------------------------------------------------------|
| Accuracy      | Does the value match reality?                                   |
| Completeness  | Is everything that should be there, there?                      |
| Consistency   | Is the same fact represented the same way across sources?       |
| Validity      | Does the value conform to its defined rule / format / range?    |
| Uniqueness    | Does each real-world entity appear exactly once?                |
| Timeliness    | Is the data fresh enough for the decision it informs?           |

Every dextra cleaning function must serve at least one of these. If it does
not, it is not a cleaning function.

### 1.2 Tidy Data (Hadley Wickham — the structural standard)

1. Each variable forms one column.
2. Each observation forms one row.
3. Each type of observational unit forms one table.

Any dataset that breaks these rules must be *reshaped before* cleaning, not
during.

### 1.3 The ten golden rules of professional cleaning

1. **Never touch raw data.** A pristine copy is always retained.
2. **Idempotent.** Running the pipeline twice produces the same output.
3. **Auditable.** Every change carries a log entry and a reason.
4. **Reversible.** Every step can be undone or replayed.
5. **Context-aware.** No generic rule applies without domain validation.
6. **Schema-first.** Every dataset has a target schema; deviations are flagged.
7. **Defensive.** Encoding drift, locale changes, schema changes are expected.
8. **Assertions, not assumptions.** Each step ends in a verifiable check.
9. **Do not over-clean.** Removed values are removed signal.
10. **Profile before and after.** Quality must be measured, not asserted.

### 1.4 Anti-patterns to avoid

- Dropping rows because they look noisy.
- Filling missing values with the column mean by default.
- Treating outliers as errors without inspection.
- Cleaning at analysis time (cleaning belongs to its own stage).
- Manual cleaning in spreadsheets (loses reproducibility).
- Confusing cleaning with feature engineering. Scaling, binning, encoding =
  feature engineering, **not** cleaning.

---

## 2. The unified 8-stage framework

Most published lists vary between 10 and 12 stages because they fold either
"business understanding" (a non-technical activity) or "feature engineering"
(a downstream activity) into cleaning. The canonical breakdown actually
practised at Snowflake / Databricks / Meta scale is **8 stages**:

| #  | Stage                  | What it does                                            | DAMA dimension(s) served            |
|----|------------------------|---------------------------------------------------------|--------------------------------------|
| 0  | Profile & Audit        | Diagnose without modifying.                             | Completeness, Validity, Uniqueness   |
| 1  | Structural Cleanup     | Column names, whitespace, encoding, case.               | Consistency                          |
| 2  | Type Coercion          | Cast each column to its correct dtype.                  | Validity                             |
| 3  | Missing Values         | Resolve NA per a documented strategy.                   | Completeness                         |
| 4  | Duplicate Resolution   | Exact + fuzzy duplicates, with record-linkage where fit. | Uniqueness                           |
| 5  | Outlier Treatment      | Detect + clip/drop/transform with a documented rule.    | Accuracy                             |
| 6  | Consistency Rules      | Cross-field, business-rule, and referential validation. | Consistency, Validity                |
| 7  | Final Validation       | Re-profile and diff against stage 0.                    | All six                              |

### Strict scope boundaries

- **In scope:** stages 0-7 above.
- **Out of scope (other phases of dextra):**
  - **Feature engineering** (scaling, encoding, binning, datetime features)
    → Phase 4: `features.py`.
  - **Class imbalance handling** (SMOTE, weighted loss)
    → already provided diagnostically by `class_imbalance` in Phase 2; the
    *fix* belongs to Phase 6: `modeling.py`.
  - **NLP text normalisation** (tokenisation, stemming, lemmatisation)
    → separate domain, deferred until a dedicated text module if/when needed.
  - **Time series specific operations** (resampling, gap filling)
    → Phase 8: `timeseries.py`.

---

## 3. The professional cleaning pipeline (how it flows)

```
Raw Data Store (immutable)
        |
        v
   [0] Profile  -> pre-cleaning report
        |
        v
   [1-6] Clean (idempotent steps, each with logging + diff)
        |
        v
   [7] Validate (assertions: schema match, ranges, uniqueness)
        |
        v
   Clean Data Store (versioned)
        |
        v
   Feature Engineering   (Phase 4)
        |
        v
   Feature Store
        |
        v
   Analytics / ML
```

Every step yields three artefacts, not one:
1. The transformed data.
2. A cleaning report (the diff: what changed, how many rows/cells affected).
3. Quality assertions (which passed, which failed, with thresholds).

This is the model Great Expectations, Pandera, and Soda implement.

---

## 4. Mapping to dextra Phase 3 (7 functions)

| Stage | dextra function                       | Alias       | DAMA dimension(s)              |
|-------|---------------------------------------|-------------|---------------------------------|
| 0     | `clean_report(df)`                    | `cleanrep`  | Completeness, Validity, Uniqueness |
| 1     | `standardize_columns(df)`             | `stdcols`   | Consistency                     |
| 2     | `cast_types(df, schema=None)`         | `cast`      | Validity                        |
| 3     | `handle_missing(df, strategy='auto')` | `fillna_smart` | Completeness                 |
| 4     | `dedupe(df, subset=None)`             | `dedup`     | Uniqueness                      |
| 5     | `clip_outliers(df, method='iqr')`     | `clipout`   | Accuracy                        |
| 6     | `validate_rules(df, rules)`           | `vrules`    | Consistency, Validity           |
| 7     | (re-call `clean_report`)              | -           | All six                         |

Stage 7 is intentionally **not** a new function; the contract is that calling
`clean_report` on the output of the pipeline yields a diff vs the input
report, satisfying stage 7.

---

## 5. Universal dextra contract for every cleaning function

Every function in `cleaning.py` must:

1. Accept `df: pd.DataFrame` and return a **new** DataFrame (immutability).
   The original `df` is never mutated.
2. Print a *before* and an *after* tabular summary with the relevant counts.
3. Render a multi-panel figure that visualises the change (bar chart, heatmap,
   or scatter — whatever fits).
4. Print a one-line **Decision sentence** in English describing what was done
   and to how many rows / columns.
5. Provide the standard flags: `show`, `plot`, `return_df`, `return_fig`,
   `decimals`, `inplace=False` (default), and an `audit=True` toggle that
   attaches a `.audit_log` attribute to the returned DataFrame.
6. Have a short alias (the `dextra` tradition).
7. Be idempotent: `f(f(df)) == f(df)`. We have a test for this.

### The `audit_log` convention

When `audit=True`, the returned DataFrame carries an attribute
`df.attrs['dextra_audit']` which is a list of dictionaries, one per cleaning
operation, with:

```python
{
    "stage": "missing_values",        # one of the 8 stages
    "function": "handle_missing",
    "timestamp": "2026-05-17T20:31:55Z",
    "before": {"n_rows": 500, "n_missing": 73, ...},
    "after":  {"n_rows": 500, "n_missing": 0,  ...},
    "params": {"strategy": "auto"},
    "decision": "Imputed 73 cells across 4 columns (median for numeric, mode for categorical).",
}
```

`clean_report(df)` is aware of this attribute and renders the full audit
trail at the bottom of its output.

---

## 6. Reference list — what professional tools do that we draw from

| Tool                    | Idea we adopt                                                            |
|-------------------------|--------------------------------------------------------------------------|
| **Great Expectations**  | Assertions on each cleaning step, not just visual inspection.            |
| **Pandera**             | Schema-as-code with explicit validation.                                 |
| **pyjanitor**           | The "one method per cleaning intent" API; we match its discoverability.  |
| **dbt**                 | Each step is reproducible and unit-tested.                               |
| **OpenRefine**          | Column-wise, GUI-style profiling that surfaces patterns.                 |
| **DAMA-DMBOK 2.0**      | The six quality dimensions are our north star.                           |
| **Hadley Wickham 2014** | Tidy data principles for structure.                                      |
| **Soda Core**           | Quality SLAs and thresholds, not just pass/fail.                         |

dextra Phase 3 does not try to replicate any of them in full. It picks the
one decision each tool gets right and ships it as a one-line call.

---

## 7. The golden test

A cleaning function in dextra passes the philosophy when:

1. Its decision sentence answers **which DAMA dimension** it served.
2. Its visual shows the **before vs after** change in one glance.
3. Its tabular output is **dense** (multiple metrics per call, not a single
   scalar).
4. Re-applying it leaves the data unchanged (idempotent).
5. It refuses to silently lose data (e.g. dropping rows without reporting how
   many and why).

Any function violating these is rewritten before shipping.

---

## Update history

| Date       | Note                                                              |
|------------|-------------------------------------------------------------------|
| 2026-05-17 | Document created as Phase 3 blueprint. 7 functions confirmed.    |
