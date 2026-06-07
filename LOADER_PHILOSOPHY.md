# dextra Phase 11 — Loader Philosophy & Framework

> **Purpose.** This document defines the philosophy, the framework, and the
> design constraints for the smart data-loading layer in dextra. It is the
> blueprint for Phase 11. Any loading function we ship must satisfy this
> reference; if it does not, we either fix the function or extend this document.
> It is written in the same one-line spirit as the earlier philosophy files
> (`MODELING_PHILOSOPHY.md` §4.7) and **adopts the unified contract in full from
> day one** — so it becomes the *gold standard* for that contract, not an
> exception to it.

> **Status of this document.** Design blueprint only. No code has been written.
> All design decisions below were settled deliberately and are recorded in the
> Update history.

---

## 0. Why this phase exists (and why it is "Phase 0" of the pipeline)

The 10-phase Roadmap begins where most pain has already happened: it assumes a
clean, correctly-typed `DataFrame` already exists. In real analyst work, **half
the time is spent before that point** — fighting messy CSVs, mystery encodings,
multi-sheet Excel files, and ad-hoc SQL. dextra never touched that step.

Phase 11 closes the loop. Chronologically it is the eleventh phase; **in the
pipeline it is the entry layer ("Phase 0")**: raw source → faithful, typed,
*documented* DataFrame. With it, the whole story becomes two honest lines:

```python
df = dx.load('sales.xlsx')     # raw, messy file -> typed DataFrame + full disclosure
dx.edareport(df)               # -> a self-contained report
```

This phase is the chosen wedge for dextra's positioning: **a safe, auditable
data-analysis tool for education and fast reports.** Loading is the most
teachable moment and the most security-sensitive entry point, so it carries the
brand more directly than any other phase.

---

## 1. Definition and boundary

**Loading = turning an external source (file / connection) into a faithful,
correctly-typed pandas DataFrame, while disclosing exactly how it was parsed and
what was uncertain — and emitting a replayable record of every decision.**

The loader is **disclosure-first**. Its value is not "we parse better than
pandas" (we largely cannot) — it is **"we parse transparently, safely, and
reproducibly, with a safety net."**

### 1.1 The boundary with cleaning (Phase 3) — settled, non-negotiable

This boundary is the single most important design decision in this phase. A
fuzzy line here produces duplication and scope creep.

| | **Loader (Phase 11)** | **Cleaning (Phase 3)** |
|---|---|---|
| Job | Get bytes into a faithful, typed frame | Repair corrupted *values* |
| Operations | **Read-time, safe, reversible** coercions only: encoding, delimiter/dialect, real header row, dtype inference, locale numerics, dates | Missing-value strategy, outliers, dedupe, standardisation, rule validation |
| On problems | **Detect, report, flag** — minimal automatic fixing (parsing-level only) | Actively transform values |
| Posture | Never silently loses data; failed coercions surface as NaN **and** are counted in the report | Decides and applies value-level fixes |

The loader **ends with a bridge, never an overlap**: its `Decision:` sentence
points to cleaning (e.g. "…run `dx.clean_rep(df)` to inspect quality"). The
loader does *parsing*; cleaning does *repair*.

### 1.2 Reuse, don't reinvent

Encoding and dialect detection are deep, well-studied problems. The loader
**wraps mature libraries lazily** rather than re-implementing them:
`charset-normalizer` (encoding), `clevercsv` (dialect/delimiter), `pandas` /
`openpyxl` (reading), `SQLAlchemy` (connections), `pyarrow` (parquet). dextra's
value-add sits *on top*: the unified contract, the disclosure report, categorical
confidence with reasons, the replayable plan, and security-by-default.

---

## 2. The non-negotiable principles

### 2.1 Transparency scales with uncertainty
The governing rule of the whole phase. The loader is **adaptively transparent**:

- **High confidence → load in one line**, but disclose fully and store the plan.
- **Ambiguity → stop guessing silently**: surface the plan, flag the ambiguous
  decisions, and ask the user to confirm/replay with an explicit plan.

Transparency is not constant friction; friction appears **only where genuine
doubt exists.** This single principle reconciles "education / audit" with "quick
reports" without two separate functions.

### 2.2 "Ask for confirmation" is propose→inspect→apply — never a blocking prompt
A blocking interactive prompt (`input()`) would break dextra: its functions run
in scripts, notebooks, reports, dashboards and CI, where nothing may block.
"Confirmation" is therefore implemented as a **non-blocking, plan-first** flow:

- `peek(source)` proposes a **load plan** (+ preview) and commits nothing.
- `load(source)` applies its best plan, prints the full report, stores the plan
  on `df.attrs`, and **shouts about any ambiguity** ("⚠ 3 columns ambiguous —
  review and re-run with `params=`").
- `load(source, params=plan)` replays a confirmed/edited plan deterministically.

A truly interactive prompt exists **only** behind an explicit `interactive=True`,
reserved for live classroom demos. It is never the default and never reached
during composition.

### 2.3 The load plan is a reproducible, auditable artifact
The loader's reproducible record **is the `params` artifact of the unified
contract** (the "load plan" is simply the human-facing name for it). It is:

- **human-readable** (printed as a table; explains each decision in words),
- **JSON-serialisable** (savable, diffable, version-controllable),
- **hand-editable** (override an encoding, force a dtype, pick the header row),
- **source-stamped** (carries a hash of the source so replay can warn when the
  file changed), and
- **deterministic on replay**: `load(source, params=plan)` reproduces the frame
  exactly.

This turns the least reproducible step in analysis into a documented, replayable
one — the literal meaning of "auditable".

### 2.4 Honest, not magic — categorical confidence with reasons
The loader never reports a fake-precise probability (e.g. "confidence 0.92"); we
have no calibration that makes such a number meaningful, and false precision
would itself violate "honest, not magic". Instead every decision carries a
**coarse categorical confidence plus a concrete reason**:

- `confirmed` — load silently, log it.
- `ambiguous` — show, flag, ask (e.g. "2 conflicting parses: 78% dates, 22% text").
- `ambiguous-high-risk` — ambiguity on a column likely to be a key/target/id.

Confidence is shown as a **category + evidence**, never a manufactured float.

### 2.5 Security by default (this is what "safe" means at the entry point)
A loader branded "safe" must earn it where data enters:

- **No automatic `pickle` loading.** Pickle is code execution; if a source is a
  pickle, refuse-by-default with a clear, explicit opt-in and warning.
- **SQL is parametrised only** — never string-interpolated; a read-only option;
  a row-limit guard.
- **Excel reads values, not formulas**; no macro execution.
- **Path, size, and resource guards**; encoding-bomb / pathological-input aware.
- **CSV-injection awareness** is carried into any later export.

### 2.6 Immutability and faithfulness
The source is never modified. The returned frame is a faithful representation of
the parse described by the plan; nothing is invented. Failed coercions become
NaN **and** are counted — data is never silently dropped.

### 2.7 No new *required* dependency
The core install is untouched. New backends are **optional, lazy extras**:
`io` (charset-normalizer, clevercsv, openpyxl), `sql` (SQLAlchemy); parquet
reuses the existing `perf` extra (pyarrow). `import dextra` still needs only the
core stack; each backend is imported lazily, only when its source type is used.

---

## 3. The functions

Public names are underscore-free and short, consistent with Phases 8–10.

| Function (alias) | Role |
|---|---|
| `load` (`dload`) | The universal smart loader. Auto-detects source type (csv/tsv, xlsx/xls, parquet, json/ndjson, sql), applies the best plan, discloses fully, stores a replayable plan, and adapts transparency to uncertainty. |
| `peek` (`dpeek`) | **Inspect without committing.** Proposes the load plan + a small preview + the disclosure report, for huge files and for teaching ("look before you load"). Returns the plan; loads nothing. |

Two functions are enough. Resist adding more — the plan/policy flags cover the
variation.

---

## 4. The dextra contract for the loader

### 4.1 One-line invocation
```python
df = dx.load('messy.csv')                       # one line: load + full disclosure
df = dx.load('book.xlsx', sheet='Q1')           # pick a sheet
df, plan = dx.load('messy.csv', return_params=True)   # capture the replayable plan
df = dx.load('messy.csv', params=plan)          # deterministic replay
plan = dx.peek('huge.csv')                      # inspect only; load nothing
df = dx.load('db', sql='SELECT * FROM t WHERE d = :d', sql_params={'d': '2026'})
```

### 4.2 Flags
Universal (unified contract): `params=None`, `return_params=False`, `show=True`,
`decimals`, `df_name`. Loader-specific:
`source`, `kind` (auto-detected; override allowed), `sheet`,
`sql` / `sql_params`, `on_ambiguous` (`"warn"` default | `"raise"` | `"plan"`),
`interactive=False`, `max_rows` / size guards, `allow_pickle=False`.

`plot`/`return_fig`/`return_df` are inherited where meaningful (the disclosure
report is primarily a table; a figure is optional, e.g. a per-column parse-rate
bar). The loader otherwise honours the same flag vocabulary as the fit/apply
families — making it the contract's reference implementation.

### 4.3 Outputs
- The parsed **DataFrame** (faithful to the plan; original source untouched).
- A printed **disclosure report**: per column — inferred dtype, % parsed OK,
  % null, n distinct, problem detected, action taken, **confidence category +
  reason**.
- A one-line **`Decision:`** sentence, e.g.
  *"Loaded 10,234 rows × 12 cols from messy.csv; detected cp1256 encoding,
  ';' delimiter, header at row 3; coerced 4 columns to numeric (0.2% cells
  failed → NaN); 1 column parsed as datetime; 3 columns ambiguous — re-run with
  `params=` to confirm. Next: dx.clean_rep(df)."*
- An append to `df.attrs['dextra_audit']` (returned on the frame, observably).
- The return value: the **DataFrame** by default; the **(DataFrame, plan)** pair
  when `return_params=True`; for `peek`, the **plan** alone.

### 4.4 The load plan / `params` schema (fixed key set)
```text
function       "load"
source         {path|url|conn, kind, sha256, size, mtime}
parse          {encoding, delimiter, quotechar, header_row, skiprows,
                sheet, na_values, thousands, decimal, date_formats}
columns        {col: {dtype, coerced_from, parse_rate, n_failed,
                      confidence: "confirmed|ambiguous|ambiguous-high-risk",
                      reason}}
problems       [ {scope, kind, detail, action} ]
policy         {on_ambiguous, allow_pickle, max_rows}
metadata       {n_rows, n_cols, n_ambiguous}
version        dextra version
generated_at   ISO-8601 UTC
```
The plan is the JSON-safe descriptor **and** the replay key. Replaying asserts
the source hash; a mismatch is reported, never silently ignored.

### 4.5 The `on_ambiguous` policy (one flag, three environments)
- `"warn"` (default — notebooks / quick reports): load, disclose loudly, flag.
- `"raise"` (audit pipelines / CI): raise on any ambiguity — no silent pass of
  an uncertain decision into a trusted result.
- `"plan"` (teaching / review): do not load; return the plan for inspection.

This is "ask for confirmation" expressed as a non-blocking, automatable policy.

### 4.6 Determinism
`load(source, params=plan)` reproduces the same frame from the same source.
Detection (encoding/dialect/dtype) is deterministic given identical bytes; the
source hash guarantees replay validity.

### 4.7 Composition (how report / dashboard / cleaning use it)
`edareport`, `dash` and the cleaning flow are **non-interactive**. When they call
the loader they must run with `on_ambiguous="warn"` or a fixed, pre-confirmed
plan — **the loader never asks for confirmation during composition**; disclosure
is logged to the audit trail, the build never blocks.

---

## 5. Anti-patterns we will refuse

| Anti-pattern | Why it is fatal |
|---|---|
| A blocking interactive prompt by default | Breaks scripts/notebooks/reports/CI; violates the one-line spirit. |
| Silently coercing/dropping data | Destroys trust and auditability; failed coercions must surface and be counted. |
| Fake-precise confidence numbers | "Honest, not magic" — uncalibrated floats are dishonest; use categories + reasons. |
| Auto-loading pickle | Code execution at the entry point; the opposite of "safe". |
| String-interpolated SQL | Injection; SQL must be parametrised. |
| Re-implementing encoding/dialect detection | Wasteful and worse than mature libraries; wrap them lazily. |
| Doing value-level repair in the loader | That is cleaning (Phase 3); the loader only parses and discloses. |
| A new *required* dependency | Backends are optional, lazy extras; the core install stays light. |
| Asking for confirmation inside `edareport`/`dash` | Composition must never block; disclosure is logged, not prompted. |

---

## 6. Where the loader sits in the pipeline
```
RAW SOURCE (csv / xlsx / parquet / json / sql)
        |
   dx.load(source)   <-- Phase 11 (the entry layer / "Phase 0")
        |   faithful typed DataFrame + disclosure report + replayable plan
        v
   Clean (P3) -> Features (P4) -> Select (P5) -> Model (P6) -> Evaluate (P7)
                                  \-> Time series (P8)
        |
   edareport (P9)  /  dash (P10)
        |
   PROVENANCE CHAIN: load plan  ->  dextra_audit  ->  report
   (raw file to conclusion, every step documented and replayable)
```

The provenance chain is the defensible niche: **auditable analysis**, valued in
education and regulated/professional contexts — a moat that function strength
alone does not provide.

---

## 7. Reference tools we draw from

| Tool | Idea we adopt |
|------|----------------|
| **charset-normalizer** | Robust encoding detection (wrapped, lazy). |
| **clevercsv** | CSV dialect/delimiter detection that beats naive sniffing. |
| **pandas `read_*` / openpyxl** | The actual readers; we orchestrate, not replace. |
| **SQLAlchemy** | Safe, parametrised, engine-agnostic SQL connections. |
| **Frictionless / Great Expectations / pandera** | The idea of an explicit, inspectable data contract/manifest — here as the replayable load plan. |
| **pandas-profiling / ydata-profiling** | The disclosure-report mindset — here at load time, with confidence and reasons. |

---

## 8. Staged delivery plan

Each stage passes `run_validation.ps1` (with deliberately messy fixtures) before
the next begins. Public names underscore-free; coverage stays above the gate.

| Stage | Scope | Status |
|-------|-------|--------|
| 11.1 | `load` for **messy CSV/TSV**: encoding + dialect detection, real header row, locale-aware numeric/date inference, categorical confidence + reasons, the replayable plan, `on_ambiguous` policy, full disclosure report, audit trail. (≈80% of the pain.) | planned |
| 11.2 | **Excel**: sheet listing/selection, data-block detection within a sheet, multi-row headers, values-not-formulas, date serials. | done |
| 11.3 | **Safe SQL** (parametrised, read-only option, row guard) + **`peek`** (inspect-without-load) + parquet/json convenience. | done |

Fixtures of deliberately broken inputs (wrong encoding, junk pre-header rows,
ragged rows, mixed-type columns, Excel layout traps) are themselves a
brand-building asset and the backbone of the test suite.

---

## 9. The golden test for the loader
The loader passes the philosophy when:
1. One call turns a messy source into a faithful, typed DataFrame **and** prints
   a full disclosure report + a `Decision:` sentence.
2. Transparency scales with uncertainty: confident parses load in one line;
   ambiguous ones are flagged and never guessed silently.
3. "Confirmation" is non-blocking (propose→inspect→apply); a real prompt exists
   only behind `interactive=True` and never during composition.
4. `return_params=True` yields a human-readable, JSON-safe, editable load plan
   with a source hash; `load(source, params=plan)` replays the frame exactly.
5. Confidence is categorical with a concrete reason — never a fake float.
6. Security defaults hold: no auto-pickle, parametrised SQL, values-not-formulas,
   size/path guards.
7. It introduces no new *required* dependency; backends are optional, lazy
   extras; `import dextra` is unaffected.
8. It only parses and discloses; it never does value-level repair (that is P3),
   and it bridges to cleaning in its decision sentence.
9. It adopts the unified contract in full (`params`/`return_params`/`df_name`/
   `show`/audit), serving as the contract's reference implementation.
10. The original source is never modified; failed coercions surface as NaN and
    are counted, never silently dropped.

Any function violating any of these is rewritten before shipping.

---

## Update history
| Date | Note |
|------|------|
| 2026-06-06 | Document created as the Phase 11 blueprint (the loader / "Phase 0" entry layer). Settled by design discussion: positioning = safe + auditable, for education and fast reports; **adaptive transparency** ("transparency scales with uncertainty"); "ask for confirmation" = non-blocking propose→inspect→apply (real prompt only behind `interactive=True`); default policy `on_ambiguous="warn"`, with `"plan"`/`"raise"` for teaching/audit; **categorical confidence + reason** (no fake floats); the **load plan = the unified-contract `params` artifact** (readable, JSON, editable, source-hashed, replayable); Phase 11 chronologically but documented as the pipeline's entry layer; the loader **adopts the unified contract in full**, becoming its gold standard and addressing the audit's contract-consistency gap (#4); reuse-don't-reinvent (lazy-wrap charset-normalizer / clevercsv / SQLAlchemy / openpyxl / pyarrow); new optional extras `io` and `sql`, reusing `perf` for parquet; strict boundary with cleaning (parse vs repair). Two public functions confirmed: `load` (`dload`) and `peek` (`dpeek`). No code written. |
