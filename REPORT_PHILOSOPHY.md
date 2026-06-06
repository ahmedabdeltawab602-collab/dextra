# dextra Phase 9 — Report Philosophy & Framework

> **Purpose.** This document defines the philosophy, the framework, and the
> design constraints for the one-call report in dextra. It is the blueprint for
> Phase 9. Any report function we ship must satisfy this reference; if it does
> not, we either fix the function or extend this document. It is the Phase-9
> sibling of the earlier philosophy files and inherits the same one-line spirit
> (`MODELING_PHILOSOPHY.md` §4.7).

---

## 1. Definition and boundary

**Report = aggregating dextra's per-phase one-liners into a single,
self-contained artifact with one call.** The report *computes nothing new*: it
orchestrates the already-tested functions of Phases 1–8, captures their tables,
figures and `Decision:` sentences, and lays them out as one document.

Boundaries with adjacent phases:
- **Phases 1–8 — done.** Each ships a one-line analysis (table + figure +
  decision).
- **Phase 9 (this) — report.** Composes those analyses into one HTML document.
  It is an *orchestrator*, not a new analysis.
- **Phase 10 (dashboard) — final.** Turns the same composition into an
  interactive Streamlit app.

The report is **composition-first**: its quality is the quality of the functions
it calls. It must never re-implement a statistic, a plot, or a model — it calls
the canonical dextra function and embeds the result.

---

## 2. The non-negotiable principles

### 2.1 One call → one self-contained file
`dx.edareport(df)` writes a single `report.html` that opens anywhere with no
external assets: every figure is embedded as a base64 PNG and every table as
inline HTML. No CSS/JS/image files are written beside it; the document is
portable by construction.

### 2.2 No new heavy dependency
The HTML report is built with the existing stack only (matplotlib → PNG, pandas
`to_html`, the standard library's `base64`). No new dependency is introduced.
PDF export is deferred to an optional engine extra (`report`) and is **not**
required for the core report.

### 2.3 Reuse verbatim — never re-implement
Each section calls the canonical dextra function with `show=False` (so nothing
prints) and `return_df=True` / `return_fig=True` (so the table and figure are
captured), then embeds exactly what that function produced. The function's
`Decision:` sentence is captured and shown as the section caption. scikit-learn
(for the optional model section) is imported lazily, exactly as in Phases 5–7;
if absent, that section is skipped, not crashed.

### 2.4 Section isolation — a report never crashes
Every section is built inside its own guard. If a section cannot run (no numeric
columns for correlation, no categorical columns for frequencies, scikit-learn
missing for the model section, a degenerate column), it is **skipped with a
recorded reason** and the rest of the report is still produced. One bad column
never denies the user the whole document.

### 2.5 Immutability
The input DataFrame is never mutated. The optional model section operates on
copies and an internal train/test split; nothing is written back to `df`. The
report appends its audit entry to a *copy*'s `df.attrs['dextra_audit']`.

### 2.6 Honest, not magic
The report states what it did and what it skipped. The model section, when
present, labels train vs. test and never presents a baseline as a tuned final
model. The report makes no claim the underlying functions would not make.

---

## 3. The sections (modular, ordered)

| # | Section | Built from | Skipped when |
|---|---------|------------|--------------|
| 1 | Overview | shape, dtypes, memory, per-column non-null / unique (pure pandas) | never |
| 2 | Data quality | `missing_report`, duplicate count, `outliers_report` | no rows |
| 3 | Univariate | `describe_numeric` + `plot_histograms`; `frequency_table` for the top categoricals | no numeric / no categorical (each independently) |
| 4 | Bivariate | `correlation_matrix`; `class_imbalance` when a categorical `target` is given | < 2 numeric columns |
| 5 | Model (optional) | `regress` / `classify` on a split + `residual_analysis` / `confusion_report` | `include_model=False`, no `target`, or scikit-learn absent |

Section 5 is opt-in via `include_model=True` together with `target=`. The task is
inferred from the target dtype (numeric → `regress` + `residual_analysis`;
categorical → `classify` + `confusion_report`), trained on a train split and
evaluated on the held-out test split — the Phase 6/7 contract, unchanged.

---

## 4. The dextra contract for the report

### 4.1 One-line invocation
```python
dx.edareport(df)                                   # -> writes report.html
dx.edareport(df, out='eda.html', title='Sales EDA')
dx.edareport(df, target='churn', include_model=True)   # adds a model section
manifest = dx.edareport(df, return_params=True)        # the build manifest
```

### 4.2 Flags
`out` (path), `target` (optional, for the bivariate/model sections), `title`,
`sections` (None = all, or a subset of section keys), `include_model`,
`max_hist` (cap on histogram columns), `top_cat` (categoricals to tabulate),
`theme` (`light`), `return_params` (return the manifest), `show`, `df_name`.

### 4.3 Outputs
- A **self-contained HTML file** written to `out`.
- A printed one-line **`Decision:`** sentence (path + sections built / skipped).
- An append to a copy's **`df.attrs['dextra_audit']`**.
- The return value: the output **path** by default, or — when
  `return_params=True` — a JSON-safe **manifest** descriptor: `function`,
  `out`, `title`, the per-section status (`built` / `skipped` + reason), the
  captured decisions, `metadata` (`n_rows`, `n_cols`, `target`, `include_model`),
  `version`, `generated_at`.

### 4.4 Determinism
Re-running on the same DataFrame produces the same sections and decisions (any
randomness in the optional model section is seeded, inheriting Phase 6/7
reproducibility).

---

## 5. Anti-patterns we will refuse

| Anti-pattern | Why it is fatal |
|---|---|
| Re-implementing a statistic / plot inside the report | Duplicates logic the phases already own and test; the report must call them. |
| Writing sidecar CSS / JS / image files | Breaks "one portable file"; everything is embedded. |
| Letting one bad column abort the whole report | Section isolation exists precisely so the document always renders. |
| Adding a heavy dependency for HTML | The base stack already produces HTML; PDF stays an optional extra. |
| Mutating `df` (e.g. appending predictions) in the model section | The report is read-only; the model section works on a copy/split. |
| Presenting the baseline model as a final tuned model | Dishonest; the section labels it a baseline on a test split. |

---

## 6. Where the report sits in the pipeline
```
Cleaned (P3) -> Engineered (P4) -> Selected (P5) -> Modeled (P6) -> Evaluated (P7)
                                  \
                                   +-- Time series (P8)
                                          |
   ONE CALL: edareport(df[, target, include_model])                      <- Phase 9
          composes the above into a single self-contained HTML
                                          |
   Interactive Streamlit dashboard                                        <- Phase 10
```

---

## 7. Reference tools we draw from

| Tool | Idea we adopt |
|------|----------------|
| **pandas-profiling / ydata-profiling** | The one-call full-EDA report idea — reproduced as a thin orchestrator over dextra's own functions, dependency-free. |
| **sweetviz** | Compact, sectioned visual EDA; section-per-theme layout. |
| **Jupyter nbconvert** | Self-contained HTML with embedded images as the portable deliverable. |

---

## 8. Phase 9 staged delivery plan

| Stage | Scope | Status |
|-------|-------|--------|
| 9.1 | HTML framework + section isolation + Overview + Data-quality | planned |
| 9.2 | Univariate + Bivariate sections | planned |
| 9.3 | Optional target-aware Model/Evaluation section (lazy sklearn) | planned |

The public name is underscore-free (`edareport`, alias `edarep`), consistent
with the Phase 8 naming. Coverage stays above the 68% gate; `run_validation.ps1`
runs green before every commit. PDF export (an optional `report` extra) is
deferred until the HTML report is complete.

---

## 9. The golden test for the report
The report passes the philosophy when:
1. One call writes a single self-contained HTML file (no sidecar assets).
2. It introduces no new required dependency.
3. Every section is produced by calling a canonical dextra function, not by
   re-implementing it.
4. A failing section is skipped with a recorded reason; the report still renders.
5. The input DataFrame is never mutated.
6. `return_params=True` returns a JSON-safe manifest (sections, decisions,
   metadata); the default return is the output path.
7. The optional model section is opt-in, lazy about scikit-learn, and labels
   train vs. test honestly.
8. Re-running on the same data reproduces the same report.

Any violation is fixed before shipping.

---

## Update history
| Date       | Note                                                          |
|------------|----------------------------------------------------------------|
| 2026-06-06 | Document created as Phase 9 blueprint. One function confirmed: `edareport` (`edarep`). Self-contained HTML, no new dependency (PDF deferred to an optional `report` extra); section isolation; optional target-aware model section (lazy scikit-learn). Underscore-free public name. |
| 2026-06-06 | Hardening: the section builders moved to a neutral `dextra._compose` layer; `edareport` is now the HTML renderer over it, and the Phase-10 dashboard renders the same builders. No behaviour change. |
