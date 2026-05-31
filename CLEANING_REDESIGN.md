# dextra Phase 3 — Redesign: from Prescriptive to Consultative

> **Companion to** `CLEANING_PHILOSOPHY.md`. That document defined WHAT
> cleaning means (DAMA dimensions, the 8 stages). This document redefines
> HOW dextra exposes those stages: as a **diagnose-then-act** API rather
> than the original do-everything-automatically API.

---

## 1. The problem with the v1 Phase 3 API

The first iteration of `cleaning.py` packed inspection and modification into
single calls:

| v1 function    | Behaviour                                              |
|----------------|---------------------------------------------------------|
| `handle_missing(df)` | Imputes immediately. Only 8 strategies. No "show only" mode. |
| `dedupe(df)`         | Drops immediately. User cannot see the duplicate groups before deciding. |
| `clip_outliers(df)`  | Clips immediately. No dry-run.                          |

This conflates two distinct cognitive steps that any real data scientist
performs in sequence:

1. **Diagnose** — look at the problem, understand its shape and severity.
2. **Decide** — choose a strategy, often after discussion with stakeholders.
3. **Act** — apply exactly the chosen strategy, no surprises.

v1 collapsed (1), (2), and (3) into a single call with hard-coded defaults.

---

## 2. The new principle: Diagnose → Decide → Act

```
INSPECT (read-only)          USER DECIDES         FIX (executes)
shows the problem      →     out of code     →    applies the chosen action
returns a DataFrame                                modifies / returns new df
```

Industry alignment:
- **Pandera** — `.validate()` inspects, `.coerce()` modifies.
- **Great Expectations** — Expectations are diagnostic; Actions modify.
- **OpenRefine** — Facets reveal, Operations apply.
- **dbt tests** — Tests detect; transformations are handwritten.

---

## 3. The v2 architecture — 10 functions in 3 classes

### Class A: Inspectors (read-only diagnostic functions)

| Function                | Status | What it shows                                                   |
|-------------------------|--------|------------------------------------------------------------------|
| `clean_report(df)`      | kept   | Master audit across all 6 DAMA dimensions.                       |
| `validate_rules(df, …)` | kept   | Rule violations as a diagnostic table.                            |
| `na_show(df)`   | **new** | Rows containing NaN, which columns missing, per-column statistics. |
| `dup_show(df, subset)` | **new** | Duplicate groups laid out side-by-side, with diff-column flags. |
| `out_show(df, cols, method)` | **new** | Outlying rows with severity score, sorted by extremity. |

Universal contract for inspectors:
1. **Read-only.** Never mutates the input DataFrame, never appends to its
   `attrs['dextra_audit']`. A diagnostic function is pure.
2. **Returns a DataFrame** that the user can filter, sort, export, or
   inspect further with native pandas.
3. **Prints an English Decision sentence** that *names the next function to
   call*, e.g. "Call `dx.dedupe(drop_indices=[…])` once you have decided".
4. **Idempotent.** Calling twice produces identical output.
5. **Visualises problem size**, never proposes a solution graphically.
6. **Aliases:** `na_show`, `dup_show`, `out_show`.

### Class B: Actors (functions that modify data)

| Function                       | Status | Change                                  |
|--------------------------------|--------|------------------------------------------|
| `standardize_columns(df)`      | kept   | Add `dry_run=True` toggle.               |
| `cast_types(df, schema)`       | kept   | No change (already explicit).             |
| `handle_missing(df, strategy)` | **expanded** | Strategy options grow from 8 to 14. |
| `dedupe(df, …)`                | **expanded** | New `drop_indices=` and `dry_run=` parameters. |
| `clip_outliers(df, action)`    | kept   | Add `dry_run=True` toggle.                |

All Actor contracts from v1 still hold (immutability, audit log, decision
sentence, visual, idempotency).

---

## 4. Full strategy catalogue for `handle_missing` (14 options)

### 4.1 Single-value imputation
| Strategy   | Description                                | Best for                       |
|------------|--------------------------------------------|--------------------------------|
| `mean`     | column mean                                | symmetric numeric              |
| `median`   | column median                              | skewed / outlier-prone numeric |
| `mode`     | most frequent value                        | categorical / discrete          |
| `constant` | user-supplied via `fill_value=`            | sentinels like `"UNKNOWN"`     |

### 4.2 Random / distribution-preserving imputation
| Strategy         | Description                                                  |
|------------------|--------------------------------------------------------------|
| `random_uniform` | uniform in `[col.min(), col.max()]`                          |
| `random_normal`  | Gaussian with column's empirical mean and std                |
| `random_sample`  | sample with replacement from observed non-null values        |

### 4.3 Sequential imputation (for ordered data)
| Strategy      | Description                                |
|---------------|--------------------------------------------|
| `ffill`       | forward fill then back fill as fallback    |
| `bfill`       | back fill then forward fill as fallback    |
| `interpolate` | linear interpolation between known values  |

### 4.4 Structural choices
| Strategy     | Description                                              |
|--------------|----------------------------------------------------------|
| `drop_rows`  | drop rows containing ANY NaN                             |
| `drop_cols`  | drop columns where `pct_missing > drop_threshold`        |
| `keep`       | explicit no-op — leave the NaN in place                  |
| dict         | per-column strategy, e.g. `{'price': 'median', ...}`     |

### 4.5 Random-imputation safety
When using `random_uniform` / `random_normal` / `random_sample`, the
function **must** accept a `random_state=` argument so the result is
reproducible. Decision sentence mentions the seed used.

### 4.6 Out of scope here (Phase 4 territory)
- `knn` — KNN imputation (requires sklearn)
- `mice` — multivariate imputation by chained equations
- `regression` — model-based imputation

These belong to feature engineering / advanced imputation, not basic
cleaning, and arrive in Phase 4 or a future advanced module.

---

## 5. Workflow examples — three classic situations

### 5.1 Missing values

```python
# Step 1 - inspect
view = dx.na_show(df)
# Output:
#   - per-column profile (% missing, which dtype, suggested action)
#   - rows-with-missing table with `which_cols_missing` flag
#   - Decision sentence: "53 rows have NaNs across 4 columns. Choose a
#     strategy and call dx.handle_missing(df, strategy=...)."

# Step 2 - decide (outside the code; could be a stakeholder discussion)

# Step 3 - act with the chosen strategy
df_clean = dx.handle_missing(df, strategy={
    'price':   'median',          # robust against price outliers
    'age':     'random_normal',   # preserves the empirical distribution
    'segment': 'mode',            # categorical
    'notes':   'keep',            # do nothing; downstream tolerates NaN
}, random_state=42)
```

### 5.2 Duplicates

```python
# Step 1 - inspect
view = dx.dup_show(df, subset=['customer_id'])
# Output:
#   group_id | row_idx | customer_id | name        | last_login | ...
#   1        | 3       | C001        | Ahmed       | 2024-01-01 | ...
#   1        | 47      | C001        | Ahmed Ali   | 2024-06-30 | ...
#   2        | 12      | C002        | Sara        | 2024-03-15 | ...
#   2        | 89      | C002        | Sara        | 2024-04-22 | ...
# Decision sentence: "5 rows in 2 duplicate groups. Either:
#   (a) call dedupe(drop_indices=[...]) with rows you chose, or
#   (b) call dedupe(subset=['customer_id'], keep='last') for bulk action."

# Step 2 - decide
to_drop = [3, 12]   # keep the most recent record for each customer

# Step 3 - act
df_clean = dx.dedupe(df, drop_indices=to_drop)
# OR bulk:
df_clean = dx.dedupe(df, subset=['customer_id'], keep='last')
```

### 5.3 Outliers

```python
# Step 1 - inspect
view = dx.out_show(df, cols=['price', 'age'])
# Output:
#   row_idx | price | age  | outlier_in    | severity_z | severity_iqr
#   12      | 9999  | 35   | price         | 12.3       | 21.7
#   47      | 100   | 200  | age           | 8.7        | 9.5
# Decision sentence: "3 outlier rows detected across 2 columns.  Options:
#   action='clip' (winsorise), 'drop' (remove rows), or 'keep' (no change)."

# Step 2 - decide

# Step 3 - act
df_clean = dx.clip_outliers(df, cols=['price', 'age'], action='clip')
```

---

## 6. Per-function specification

### 6.1 `na_show(df)`
**Returns:** DataFrame of rows that contain at least one NaN, with two extra
columns: `which_cols_missing` (comma-separated list) and `n_missing_in_row`.
Plus prints a **per-column summary** (count, pct, recommended strategy).
**Visual:** bar chart of pct_missing per column + missingness heatmap.
**Decision:** *"N rows × M columns have missing values. Recommended
strategies per column: …. Decide and call `dx.handle_missing(df, strategy=…)`."*

### 6.2 `dup_show(df, subset=None)`
**Returns:** DataFrame containing only the duplicate rows, with extra columns
`duplicate_group_id`, `group_size`, `is_first_in_group`. Sorted by
`duplicate_group_id` so each cluster is visible.
**Visual:** histogram of group sizes + bar chart of duplicate counts per
column (when subset omitted).
**Decision:** *"K duplicate groups containing N rows. Call `dx.dedupe(...)`
with either `drop_indices=[…]` or `keep='first'|'last'`."*

### 6.3 `out_show(df, cols=None, method='iqr', k=1.5, z_threshold=3.0)`
**Returns:** DataFrame of outlier rows with columns `outlier_in_columns`,
`severity_z` (worst |z|), `severity_iqr` (worst distance beyond fence,
normalised by IQR), sorted by severity descending.
**Visual:** strip plot per column showing outliers + a "most extreme rows"
table.
**Decision:** *"N outliers identified. Call `dx.clip_outliers(...)` with
`action='clip'`, `'drop'`, or `'keep'`."*

### 6.4 `handle_missing(df, strategy, fill_value=None, random_state=None, ...)`
Strategy may be one of the 14 options above or a dict for per-column
choices. The Decision sentence states the strategy chosen *and* the seed
used (when applicable), to make the operation auditable.

### 6.5 `dedupe(df, subset=None, keep='first', drop_indices=None, dry_run=False)`
- `drop_indices=[…]` overrides `keep=` and drops exactly the listed rows.
- `dry_run=True` skips the drop and returns the original DataFrame
  unchanged, but still prints what would have happened. Useful for
  scripting.

### 6.6 `clip_outliers(df, cols, method, action, dry_run=False)`
Same dry-run semantics as dedupe. Decision sentence includes the bounds
used per column.

---

## 7. Backward compatibility

Every v1 call continues to work without modification:

```python
# v1 style still produces the same result
df_clean = dx.handle_missing(df)
df_clean = dx.dedupe(df)
df_clean = dx.clip_outliers(df)
```

The redesign **adds new options**, it does not remove any. v1 users see no
regressions. New users get the inspect-first workflow.

---

## 8. Staged delivery plan

| Stage | Work                                                         | Outcome                          |
|-------|--------------------------------------------------------------|----------------------------------|
| 3.4   | Add `na_show`, `dup_show`, `out_show`. | 3 inspectors, full diagnostic surface. |
| 3.5   | Expand `handle_missing` from 8 to 14 strategies (random_uniform, random_normal, random_sample, interpolate, keep, plus existing). | Imputation flexibility complete. |
| 3.6   | Add `drop_indices=` to `dedupe`, `dry_run=` to dedupe / clip_outliers / standardize_columns. | All actors gain flexibility. |
| 3.7   | Integration test: round-trip inspect → decide → fix on a deliberately messy fixture. | Pipeline validated end-to-end. |

Each stage shipped, tested, then the next begins — same workflow as
Phases 2 and 3 originally.

---

## 9. The golden test for the v2 API

A redesigned function passes the philosophy when:

1. Inspectors return data, never mutate it.
2. Actors accept user decisions explicitly (no surprise defaults).
3. Every Decision sentence names the *next function to call*, by name.
4. `dry_run=True` (where applicable) is reliable and audited.
5. Backward compatibility holds: existing v1 examples in tests still pass.
6. Documentation in `QUICKSTART.md` is updated to lead with the
   inspect-first pattern.

---

## Update history

| Date       | Note                                                          |
|------------|----------------------------------------------------------------|
| 2026-05-18 | Document created. Approved scope: 3 inspectors + expanded handle_missing + flexible dedupe & clip_outliers. |
| 2026-05-18 | Staged delivery 3.4 / 3.5 / 3.6 / 3.7 set.                     |


---

## 10. Naming convention (added per user request 2026-05-18)

All new functions follow the `<concern>_<verb>` pattern, with 6-10 character
total length:

| Concern        | Abbrev | Inspector (show) | Actor (fix)    |
|----------------|--------|-------------------|-----------------|
| Missing values | `na`   | `na_show`         | `na_fix`        |
| Duplicates     | `dup`  | `dup_show`        | `dup_fix`       |
| Outliers       | `out`  | `out_show`        | `out_fix`       |
| Rules          | `rule` | `rule_check`      | (no actor)      |
| Types          | `type` | (use clean_rep)   | `type_fix`      |
| Columns        | `col`  | (use clean_rep)   | `col_clean`     |
| Master audit   | `clean`| `clean_rep`       | -               |

### Backward compatibility

The v1 long names remain as **aliases** so existing code keeps working:

| v1 name                | v2 short name | Status                          |
|------------------------|----------------|---------------------------------|
| `handle_missing`       | `na_fix`       | both exposed, identical object  |
| `dedupe`               | `dup_fix`      | both exposed                    |
| `clip_outliers`        | `out_fix`      | both exposed                    |
| `validate_rules`       | `rule_check`   | both exposed                    |
| `cast_types`           | `type_fix`     | both exposed                    |
| `standardize_columns`  | `col_clean`    | both exposed                    |
| `clean_report`         | `clean_rep`    | both exposed                    |

### Discoverability mnemonic

```text
na   = missing values   (Not Available)
dup  = duplicates
out  = outliers
rule = business rules
type = dtypes
col  = column names
```

```text
_show  = inspect, read-only, returns a DataFrame
_fix   = act, modifies, returns a NEW DataFrame
_check = inspect for rules / validation
_clean = normalize / standardize structure
_rep   = comprehensive multi-dimension report
```
