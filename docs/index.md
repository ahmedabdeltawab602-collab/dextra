# dextra

**One line of code → a rich table, a multi-panel plot, and a decision sentence.**

`dextra` is a lightweight toolkit layered on top of pandas and scikit-learn. It
turns the repetitive first steps of any data analysis — exploration, cleaning,
feature engineering, feature selection and baseline modeling — into single,
self-explaining calls.

## The design contract (every function honours it)

- **One-line invocation** — `dx.something(df)` is enough to get a useful result.
- **Rich tabular output** — a densely packed metrics table, not a single number.
- **Comprehensive visual** — a multi-panel figure beside the numbers.
- **Decision sentence** — a short `Decision:` line summarising the result.
- **Choices, not decisions** — `method="compare"` ranks candidates without
  silently picking one.
- **Leakage-safe fit/apply** — learn on train only, replay verbatim on new data
  via a JSON-serialisable `params` artifact.
- **Immutability + audit trail** — the input frame is never mutated; every
  operation is logged to `df.attrs["dextra_audit"]`.

## The phases

| Phase | Module | What it does |
|-------|--------|--------------|
| 11 | `_loader` | `load` / `peek` — raw, messy source → typed DataFrame + full disclosure (the entry layer, "Phase 0" of the pipeline) |
| 1 | `stats`, `plots` | Rich numeric EDA + better default plots |
| 2 | `stats_advanced` | 22 statistical helpers (inference, hypothesis tests, diagnostics) |
| 3 | `cleaning` | 10 cleaning helpers across the DAMA-DMBOK stages |
| 4 | `features` | 8 leakage-safe feature-engineering functions |
| 5 | `selection` | 5 feature-selection functions (Filter / Embedded / Wrapper) |
| 6 | `modeling` | `regress`, `classify`, `cluster` — instant baselines |
| 6.5 | `compat` | scikit-learn-compatible wrappers |

See [Getting started](getting-started.md) and the [API reference](api.md).
