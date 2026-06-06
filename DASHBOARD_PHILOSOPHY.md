# dextra Phase 10 — Dashboard Philosophy & Framework

> **Purpose.** This document defines the philosophy, the framework, and the
> design constraints for the interactive dashboard in dextra. It is the
> blueprint for Phase 10 — the final phase of the Roadmap. Any dashboard
> function we ship must satisfy this reference; if it does not, we either fix the
> function or extend this document. It is the interactive sibling of
> `REPORT_PHILOSOPHY.md`.

---

## 1. Definition and boundary

**Dashboard = the interactive sibling of the report.** `dx.dash(df)` *generates
a self-contained Streamlit app* that re-runs dextra's analyses live under user
controls (column pickers, target selector, toggles). Like the report, it
**computes nothing new**: it composes the already-tested functions of Phases
1-8 and renders them — the report writes one HTML file, the dashboard writes one
runnable app.

Boundaries with adjacent phases:
- **Phase 9 (report) — done.** A static, self-contained HTML composition.
- **Phase 10 (this) — dashboard.** The same composition, made interactive via
  Streamlit. The Roadmap's final goal.

The dashboard is **composition-first**, exactly as the report is. Its sections
*are* the report's section builders, rendered to Streamlit widgets instead of to
HTML. There is no second implementation of any analysis.

---

## 2. The non-negotiable principles

### 2.1 One call → one runnable app (+ its data)
`dx.dash(df)` writes a thin `dashboard_app.py` plus a sidecar data file (a
dtype-preserving pickle by default, or CSV). The app is launched with
`streamlit run dashboard_app.py`. By default `dx.dash` only **generates** and
returns the path (`launch=False`); it never silently opens a browser.

### 2.2 Reuse verbatim — the dashboard renders the report's sections
The rendering logic lives in dextra (`_build_dashboard`) and reuses the
**neutral section builders** in `dextra._compose` (the single source of truth
shared with the Phase-9 report). The dashboard no longer depends on the report
renderer; both renderers sit on top of `_compose`. Each section's table is shown
with `st.dataframe`, its figure with `st.image`, its `Decision:` line as a
caption. No analysis is re-implemented for the dashboard.

### 2.3 The new dependency is optional and lazy
Streamlit is introduced as a **new optional extra `dash`**, imported lazily and
only inside `_build_dashboard` (and the optional auto-launch). The base install
is untouched; `import dextra` still needs only the core stack. Because the
render logic takes the Streamlit module as an ordinary dependency, it is
testable with a stub — Streamlit need not be installed to validate the dashboard.

### 2.4 Section isolation — a tab never breaks the app
Every tab is rendered inside its own guard. A section that cannot run (no numeric
columns, scikit-learn missing for the model tab) shows a "section skipped"
message; the other tabs keep working. One bad column never breaks the dashboard.

### 2.5 Immutability and honesty
The input DataFrame is never mutated; the pickled sidecar is a copy. The model
tab, when enabled, trains a baseline on a split and reports test metrics — a
floor, never a tuned final model. The dashboard makes no claim the underlying
functions would not make.

---

## 3. The tabs (reusing the report's sections)

| Tab | Built from (Phase 9 builder) | Interactive control |
|-----|------------------------------|---------------------|
| Overview | `_sec_overview` | — |
| Data quality | `_sec_quality` | — |
| Univariate | `_sec_univariate` | histogram-column cap |
| Bivariate | `_sec_bivariate` | target selector |
| Model (optional) | `_sec_model` | target selector + "include model" toggle |

The sidebar exposes the target selector, the include-model toggle, and the
histogram / category caps; changing them re-runs Streamlit and rebuilds the
affected tabs from the same builders.

---

## 4. The dextra contract for the dashboard

### 4.1 One-line invocation
```python
dx.dash(df)                                   # writes dashboard_app.py (+ data)
dx.dash(df, out='app.py', target='churn', include_model=True)
dx.dash(df, launch=True)                      # also runs `streamlit run`
manifest = dx.dash(df, return_params=True)    # the build manifest
```

### 4.2 Flags
`out` (app name), `output_dir` (collect every output in one folder), `target`,
`include_model`, `launch` (default False), `data_format`
(`"pickle"` | `"csv"` | `"parquet"`), `max_hist`, `top_cat`, `theme`,
`return_params`, `show`, `df_name`.

### 4.3 Outputs
- A runnable **`dashboard_app.py`** (a thin shim that loads the data and calls
  `dextra.dashboard._build_dashboard`).
- A **sidecar data file** next to it (pickle / CSV / parquet), resolved relative
  to the app's own location so it runs from anywhere.
- A **`*_meta.json` reproducibility manifest** (dextra / Python / pandas
  versions, generation time, the settings used).
- The generated app **checks its dependencies and data file up front** with a
  clear, actionable message before rendering.
- A printed one-line **`Decision:`** sentence (paths + how to run).
- An append to a copy's **`df.attrs['dextra_audit']`**.
- The return value: the app **path**, or — when `return_params=True` — a
  JSON-safe **manifest**: `function`, `out`, `data_path`, `data_format`,
  `target`, `include_model`, the tabs that will appear, `metadata`, `version`,
  `generated_at`.

### 4.4 Determinism
Re-running `dx.dash` on the same DataFrame writes the same app and data. The
rendered analyses inherit the determinism of the Phase 1-8 functions they call.

---

## 5. Anti-patterns we will refuse

| Anti-pattern | Why it is fatal |
|---|---|
| Re-implementing an analysis for the dashboard | The dashboard renders the report's builders; there is one implementation. |
| Making Streamlit a required dependency | It is heavy and optional; the base install and CI must not need it. |
| Auto-launching a browser by default | Surprising and untestable; `launch` is opt-in. |
| Letting one tab's failure blank the whole app | Tab isolation exists precisely so the dashboard stays usable. |
| Mutating `df` or pickling a mutated frame | The dashboard is read-only; the sidecar is a faithful copy. |
| Embedding a giant base64 blob instead of a sidecar | A dashboard naturally ships app + data; the sidecar keeps the app readable. |

---

## 6. Where the dashboard sits in the pipeline
```
Cleaned -> Engineered -> Selected -> Modeled -> Evaluated -> Time series
        \________________________  composition  ________________________/
                         |                              |
            edareport(df) -> one HTML file   dx.dash(df) -> one Streamlit app
              (Phase 9, static)                 (Phase 10, interactive)   <- final
```

---

## 7. Reference tools we draw from

| Tool | Idea we adopt |
|------|----------------|
| **Streamlit** | The one-file Python app with sidebar controls and tabs; `st.pyplot` / `st.dataframe` rendering. |
| **ydata-profiling / sweetviz** | The auto-generated, sectioned EDA surface — here made live. |
| **dtale / pandasgui** | The "point dx at a DataFrame and explore it" experience, generated from one call. |

---

## 8. Phase 10 staged delivery plan

| Stage | Scope | Status |
|-------|-------|--------|
| 10.1 | App scaffold + data handoff + Overview / Data-quality tabs | planned |
| 10.2 | Univariate / Bivariate tabs | planned |
| 10.3 | Optional Model tab + `launch` auto-run | planned |

The public name is `dash` (alias `dashapp`), underscore-free, consistent with
the Phase 8/9 naming. Streamlit is the optional `dash` extra; the base install
and CI are unaffected (the renderer is tested with a Streamlit stub). Coverage
stays above the 68% gate; `run_validation.ps1` runs green before every commit.

---

## 9. The golden test for the dashboard
The dashboard passes the philosophy when:
1. One call writes a runnable Streamlit app plus its sidecar data file.
2. It introduces no new *required* dependency; Streamlit is an optional, lazy
   extra and the renderer is testable with a stub.
3. Every tab is produced by the Phase 9 section builders, not re-implemented.
4. A failing tab is isolated; the rest of the dashboard still works.
5. The input DataFrame is never mutated; the sidecar is a faithful copy.
6. `return_params=True` returns a JSON-safe manifest; the default return is the
   app path; `launch` is opt-in.
7. Re-running on the same data writes the same app and data.

Any violation is fixed before shipping.

---

## Update history
| Date       | Note                                                          |
|------------|----------------------------------------------------------------|
| 2026-06-06 | Document created as Phase 10 blueprint (the Roadmap's final phase). One function confirmed: `dash` (`dashapp`). Generates a self-contained Streamlit app that reuses the Phase 9 report builders; Streamlit is an optional, lazy `dash` extra; `launch=False` by default. Underscore-free public name. |
| 2026-06-06 | Hardening: section builders extracted to a neutral `_compose` layer (dashboard no longer depends on `report`); added `output_dir`, a `parquet` data format (lazy engine), a `*_meta.json` reproducibility manifest, and up-front dependency / data-file checks in the generated app. |
