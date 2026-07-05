# dextra

**One line of code → a rich table, a multi-panel plot, and a decision sentence.**

`dextra` is a lightweight toolkit layered on top of pandas and scikit-learn. It
turns the repetitive first steps of any data analysis — exploration, cleaning,
feature engineering, feature selection and baseline modeling — into single,
self-explaining calls. It ships a vocabulary of **63 public functions** plus 5
scikit-learn-compatible wrapper classes — **68 distinct public callables in
all** (see the [API consistency table](api-consistency.md)) — organised as the
`load` entry layer, nine analytical modules, and the `compat` wrappers.

## The design contract

- **One-line invocation** — `dx.something(df)` is enough to get a useful result.
- **Rich tabular output** — a densely packed metrics table, not a single number.
- **Comprehensive visual** — a multi-panel figure beside the numbers.
- **Decision sentence** — a short `Decision:` line summarising the result.
- **Choices, not decisions** — `method="compare"` ranks candidates without
  silently picking one (in the feature-engineering, selection and
  modeling families).
- **Leakage-safe fit/apply** — learn on train only, replay verbatim on new data
  via a JSON-serialisable `params` artifact. This is honoured by the
  feature-engineering, selection and modeling families and by the two
  leakage-prone cleaning steps `handle_missing` and `clip_outliers`; the
  remaining cleaning helpers are deterministic, stateless one-shot operations.
- **Immutability + audit trail** — the input frame is never mutated; every
  operation is logged to `df.attrs["dextra_audit"]`.

## The phases

| Phase | Module | What it does |
|-------|--------|--------------|
| 11 | `_loader` | `load` / `peek` — raw, messy source → typed DataFrame + full disclosure (the entry layer, "Phase 0" of the pipeline) |
| 1 | `stats`, `plots` | Rich numeric EDA + better default plots |
| 2 | `stats_advanced` | 22 statistical helpers (inference, hypothesis tests, diagnostics) |
| 3 | `cleaning` | 10 cleaning helpers across the DAMA-DMBOK stages; leakage-safe fit/apply on `handle_missing`/`clip_outliers` |
| 4 | `features` | 8 leakage-safe feature-engineering functions |
| 5 | `selection` | 5 feature-selection functions (Filter / Embedded / Wrapper) |
| 6 | `modeling` | `regress`, `classify`, `cluster` — instant baselines |
| 7 | `evaluation` | `confusion_report`, `roc_pr`, `residual_analysis`, `learning_curves` — deep model evaluation |
| 8 | `timeseries` | `tsdecomp`, `tsstat`, `tsfcast` — decomposition, stationarity, baseline forecasts |
| 9 | `report` | `edareport` — one-call self-contained HTML EDA report |
| 10 | `dashboard` | `dash` — generated interactive Streamlit app |
| 6.5 | `compat` | scikit-learn-compatible wrappers |

See [Getting started](getting-started.md) and the [API reference](api.md).
