# API reference

The full public API, generated from the in-source NumPy-style docstrings, is at
the bottom of this page. The hand-written sections below cover the parts of the
contract that the auto-generated reference cannot show at a glance — most
importantly **Phase 11, the loading layer**, which is the pipeline's entry
point.

## Phase 11 — the loader (entry layer)

`dextra` begins one step earlier than most analysis toolkits: at the raw,
untrusted file. `load` turns an external source into a faithful, correctly-typed
`DataFrame` **and discloses exactly how it was parsed and what was uncertain**,
emitting a JSON-safe record that reproduces the frame on demand. It is
disclosure-first: the value is not "we parse better than pandas" — it is "we
parse transparently, safely, and reproducibly, with a safety net."

```python
import dextra as dx

df = dx.load('sales.csv')          # raw, messy file -> typed DataFrame + full disclosure
plan = dx.peek('sales.csv')        # look before you load: plan + small preview, no commit
```

Short aliases `dload` / `dpeek` are identical to `load` / `peek`.

### Five source families

`load` auto-detects the source family from the extension (override with
`kind=`), and each family is parsed and disclosed under the same contract:

| Family | Extensions | What is detected / disclosed |
|--------|-----------|------------------------------|
| Delimited text | `.csv`, `.tsv`, and unknown text | Encoding, delimiter/dialect, the real header row (junk preambles skipped), per-column dtype, locale numerics and dates — quote-aware and preamble-safe |
| Excel | `.xlsx`, `.xlsm` | Sheet selection, the data block within a sheet, single- and multi-row headers (combined into `top_bottom` names), native cell types (values, never formulas). Legacy `.xls` is refused with guidance |
| Parquet | `.parquet`, `.pq` | Typed pass-through; needs a lazy engine such as `pyarrow` |
| JSON | `.json`, `.jsonl`, `.ndjson` | Records array or one object per line; nested values are serialised and the nesting is disclosed; bad NDJSON lines are skipped and flagged |
| SQL | `.db`, `.sqlite`, `.sqlite3`, or an open DB-API connection | One parametrised `SELECT` via `sql=` + `sql_params=`; SQLite files are opened **read-only**; results are row-capped and truncation disclosed |

### `on_ambiguous` — transparency scales with uncertainty

Confident parses load in a single line. Ambiguous decisions (an uncertain header
row, several Excel sheets, an ID-like column that looks numeric, a single JSON
object instead of an array) are surfaced according to one policy:

| Value | Behaviour |
|-------|-----------|
| `"warn"` (default for `load`) | Load the frame and emit a `DextraLoaderWarning` for each ambiguous decision |
| `"raise"` | Raise `LoaderAmbiguityError` instead of guessing |
| `"plan"` (default for `peek`) | Return the load plan **without loading** the full frame |

Every load also prints a disclosure report and a one-line `Decision:` sentence
unless `show=False`.

### The replayable load plan

When `return_params=True`, `load` also returns a JSON-safe **load plan** — the
complete, serialisable record of every decision it made. Passing it back
reproduces the frame deterministically:

```python
df, plan = dx.load('messy.csv', return_params=True)
import json; json.dumps(plan)          # the plan is always JSON-serialisable
df2 = dx.load('messy.csv', params=plan)  # exact, decision-for-decision replay
```

The plan stores the source `sha256`. If the source has changed since the plan
was created, replay still proceeds but the mismatch is flagged (and raised under
`on_ambiguous='raise'`), so a silently-edited file can never masquerade as the
original.

### Security model

The loader is the most security-sensitive entry point in the library, so it is
conservative by default:

- **No code execution on load.** Pickle sources are refused unless
  `allow_pickle=True` is passed explicitly.
- **SQL is parametrised only.** Exactly one statement is allowed; stacked
  statements are refused, and values bind through `sql_params=` — never string
  formatting — so injection is structurally impossible. SQLite files open
  read-only; for a live connection read-only cannot be enforced and that fact is
  disclosed in the plan.
- **Bounded reads.** `max_rows` (and a default 1,000,000-row SQL guard) cap how
  much is read; truncation is disclosed and flagged.
- **Immutability.** The source is never modified, and an in-memory `DataFrame`
  passed to `load` is returned as a typed copy, never mutated in place.

### Audit trail

Like every dextra function, `load` records its decision to
`df.attrs["dextra_audit"]`, including the load plan, so the provenance of a
frame travels with it through the rest of the pipeline.

---

## Full generated reference

::: dextra
    options:
      show_root_heading: false
      members: true
