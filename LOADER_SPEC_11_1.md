# dextra Phase 11.1 — Loader: detailed implementation spec (CSV/TSV)

> Implementation-ready specification for Stage 11.1, derived from
> `LOADER_PHILOSOPHY.md`. Scope: **delimited text only** (csv/tsv). Excel and SQL
> are 11.2 / 11.3. This document fixes signatures, the plan schema, detection
> algorithms (with graceful fallbacks when optional libs are absent), the
> disclosure report, the `Decision:` sentence, error handling, security, and the
> test matrix. Written to be coded verbatim on the owner's machine / CI.

---

## 1. Module & public surface

- New private module: `src/dextra/_loader.py` (all logic).
- Public re-exports in `__init__.py`: `load`, `dload`, `peek`, `dpeek`
  (aliases are the *same object*, per the alias-identity contract).
- New phase label in `_PHASE_LABELS`: `"dextra._loader": "Phase 11 - loader"`,
  inserted **before** Phase 1 in `_PHASE_ORDER` (it is the entry layer).
- Add to `__all__`.

### Why a single module (not a package yet)
11.1 is one cohesive concern. If 11.2/11.3 grow it past ~800 lines, split into
`_loader_csv.py` / `_loader_excel.py` / `_loader_sql.py` behind a `loader.py`
façade — mirroring the Phase-4 `_features_*` precedent. Not now.

---

## 2. Signatures

```python
def load(
    source,                       # str | os.PathLike | file-like | pandas/polars/pyarrow frame
    *,
    kind: str = "auto",           # "auto" | "csv" | "tsv"  (excel/sql in 11.2/11.3)
    params: dict | None = None,   # a load plan to REPLAY (deterministic)
    on_ambiguous: str = "warn",   # "warn" | "raise" | "plan"
    encoding: str | None = None,  # force; None => detect
    sep: str | None = None,       # force delimiter; None => detect
    header_row: int | None = None,# force 0-based header row; None => detect
    parse_dates: bool = True,     # attempt safe datetime inference
    decimal: str | None = None,   # force locale decimal; None => detect (".")
    thousands: str | None = None, # force locale thousands; None => detect
    na_values=None,               # extra NA tokens (added to the default set)
    max_rows: int | None = None,  # safety guard (None => no cap)
    sample_bytes: int = 262144,   # bytes read for detection (256 KiB)
    allow_pickle: bool = False,   # reserved; .pkl refused unless True (security)
    return_params: bool = False,  # also return the load plan
    show: bool = True,            # print the disclosure report + Decision
    decimals: int = 4,            # numeric formatting in the printed report
    df_name: str | None = None,   # name used in title/audit (inferred when omitted)
    interactive: bool = False,    # opt-in blocking prompt (classroom only)
):
    ...

def peek(source, *, kind="auto", on_ambiguous="plan", show=True,
         n_preview: int = 10, **load_kwargs):
    """Propose a plan + preview WITHOUT committing a full load.
    Returns the load plan (dict). Loads at most `n_preview` rows for the sample."""
```

`dload = load`, `dpeek = peek`.

### Return shape (unified contract)
- `load(...)` → `df` by default; `(df, plan)` when `return_params=True`.
- `load(..., on_ambiguous="plan")` → returns the **plan** (dict), loads nothing,
  prints the report. (Mirrors `peek`; "plan" means "don't commit".)
- `peek(...)` → the **plan** (dict) always.
- `interactive=True`: after printing the plan, prompt
  `Apply this plan? [y]/edit/abort`. `y` → load; `abort` → raise `LoaderAbort`;
  edit is out of 11.1 scope (documented as 11.x). Never reached when `show=False`
  or during composition.

---

## 3. Detection pipeline (order is fixed)

All detection runs on the first `sample_bytes` of the raw stream, decoded once.
Every step records `{value, confidence, reason}` into the plan.

1. **Source kind.** From extension (`.csv`→csv, `.tsv`/`.tab`→tsv); `.pkl`/`.pickle`
   → refuse unless `allow_pickle=True` (raise `LoaderSecurityError`). Unknown
   extension on text → treat as csv with detection. `kind=` overrides.

2. **Encoding.** If `encoding=` given → `confirmed`. Else:
   - try `charset_normalizer.from_bytes(sample).best()` (lazy `io` extra);
   - fallback chain when absent: strip/detect BOM → try `utf-8` (strict) → try
     `cp1256` (Arabic) → try `latin-1` (always succeeds).
   - confidence: `confirmed` if BOM or utf-8 strict-decodes the whole sample;
     `ambiguous` if a lossy fallback was used; reason names the method
     (e.g. `"charset-normalizer: cp1256, 0 undecodable bytes"`).

3. **Header rows / junk preamble.** Detect the real header by scanning the first
   ~20 logical lines for the first row whose field count equals the modal field
   count of the following rows **and** whose cells are mostly non-numeric
   (header-like). `header_row=` overrides. Rows above it → `skiprows`.
   confidence `confirmed` if a single clear candidate; `ambiguous` if the modal
   field count is unstable.

4. **Delimiter / dialect.** If `sep=` given → `confirmed`. Else
   `clevercsv.Sniffer().sniff(sample)` (lazy `io` extra); fallback to
   `csv.Sniffer().sniff(sample, delimiters=",;\t|")`; final fallback `,`.
   confidence `confirmed` if the sniffers agree and field count is stable across
   sampled rows; else `ambiguous` (reason: e.g. `"',' vs ';' both plausible"`).

5. **Read.** `pandas.read_csv` with the resolved `encoding, sep, skiprows,
   header, na_values, decimal, thousands, nrows=max_rows, dtype=object`
   (read everything as text first, then infer types ourselves so we can MEASURE
   parse rates). Ragged rows: `on_bad_lines="warn"` and count them into
   `problems`.

6. **Per-column type inference (the measured part).** For each object column,
   in order, on the non-null cells:
   - **datetime** (only if `parse_dates`): try a small set of explicit formats
     first (`%Y-%m-%d`, `%d/%m/%Y`, `%Y-%m-%d %H:%M:%S`, ISO), then
     `pd.to_datetime(errors="coerce")`; accept if `parse_rate >= 0.95`.
   - **numeric**: strip `thousands`, normalise `decimal`, strip a single leading
     currency symbol and a trailing `%`; `pd.to_numeric(errors="coerce")`;
     accept if `parse_rate >= 0.95`. (If `%` stripped, record action; value is
     the number as-written, no division — that is a *cleaning* choice, not ours.)
   - **boolean**: map a fixed token set
     (`true/false/yes/no/y/n/1/0/t/f`, case-insensitive) iff the column's unique
     non-null values are a subset; else skip.
   - **categorical**: if `nunique/len <= 0.5` and dtype stayed object, leave as
     object in 11.1 (no auto-`category`; record a *suggestion* only).
   - else **keep as object (text)**.
   - **parse_rate** = parsed_ok / non_null. `n_failed` = non_null − parsed_ok.
     Failed cells become NaN in the typed column **and** are counted.

### 3.1 Confidence rules (categorical — never a float)
- `confirmed`: forced by the user, or detector unambiguous, or `parse_rate==1.0`.
- `ambiguous`: lossy fallback used, sniffers disagreed, header unstable, or a
  type accepted with `0.95 <= parse_rate < 1.0` (some cells failed).
- `ambiguous-high-risk`: ambiguity on a column whose name matches a key/id/target
  heuristic (`id`, `_id`, `key`, `target`, `label`, `y`, case-insensitive) — or
  a column that becomes **all-NaN** after coercion.

`n_ambiguous` in metadata = count of columns whose confidence != `confirmed`,
plus 1 for each non-`confirmed` parse-level decision (encoding/header/delimiter).

---

## 4. The load plan schema (exact)

```python
plan = {
  "function": "load",
  "source": {"name": str, "kind": "csv"|"tsv", "sha256": str|None,
             "size": int|None, "mtime": float|None},
  "parse": {"encoding": str, "delimiter": str, "quotechar": str,
            "header_row": int, "skiprows": list[int], "decimal": str,
            "thousands": str|None, "na_values": list[str]},
  "columns": {  # ordered, one entry per output column
     col: {"dtype": str, "coerced_from": "object",
           "parse_rate": float, "n_failed": int,
           "confidence": "confirmed"|"ambiguous"|"ambiguous-high-risk",
           "reason": str,
           "suggest": str|None}     # e.g. "category", "% stripped"
  },
  "problems": [ {"scope": str, "kind": str, "detail": str, "action": str} ],
  "decisions": {  # parse-level, each {value, confidence, reason}
     "encoding": {...}, "delimiter": {...}, "header": {...}, "decimal": {...}
  },
  "policy": {"on_ambiguous": str, "allow_pickle": bool, "max_rows": int|None},
  "metadata": {"n_rows": int, "n_cols": int, "n_ambiguous": int},
  "version": __version__,
  "generated_at": "<ISO-8601 UTC>",
}
```

- JSON-safe: floats rounded, no numpy scalars (reuse a `_json_safe` helper).
- `sha256/size/mtime` are `None` for non-path sources (file-like, in-memory).

### 4.1 Replay (`params=plan`)
When `params` is supplied, **all detection is skipped**; the parse + per-column
coercions in the plan are applied verbatim. If `source` is a path and its
`sha256` differs from `plan["source"]["sha256"]`, append a `problems` entry
(`kind="source_changed"`) and, under `on_ambiguous="raise"`, raise
`LoaderReplayError`. Replaying yields an identical frame on identical bytes.

---

## 5. Disclosure report & Decision

### 5.1 Report (printed when `show=True`)
A single pandas table, one row per column, columns:

```
column | dtype | parsed_% | null_% | n_distinct | problem | action | confidence
```

Preceded by a one-line parse banner:
`"source=messy.csv | encoding=cp1256 (ambiguous) | sep=';' | header=row 3 | 10,234x12"`.
Ambiguous rows are marked (e.g. a leading `⚠`). Rendering is plain `print` of a
formatted DataFrame (no new dependency); decimals honour `decimals`.

### 5.2 Decision sentence (format)
```
Decision: Loaded {n:,} rows x {m} cols from '{name}' [encoding={enc}, sep={sep!r},
header=row {h}]; coerced {k} column(s) ({types}); {f} cell(s) failed -> NaN;
{a} ambiguous decision(s){replay_hint}. Next: dx.clean_rep(df).
```
`replay_hint` = `" - re-run with params= to confirm"` when `a > 0`, else "".

---

## 6. Behaviour by policy (`on_ambiguous`)

| policy | ambiguity present | no ambiguity |
|---|---|---|
| `"warn"` (default) | load; print report; emit `warnings.warn(DextraLoaderWarning)` listing ambiguous items | load; print report |
| `"raise"` | raise `LoaderAmbiguityError` (message lists ambiguous items + how to override) | load; print report |
| `"plan"` | print report; **return the plan**, load nothing | print report; return the plan, load nothing |

`interactive=True` overlays `"warn"` with a prompt (see §2).

---

## 7. Errors, immutability, composition

- New exception hierarchy in `_loader.py`:
  `class DextraLoaderError(Exception)` → `LoaderSecurityError`,
  `LoaderAmbiguityError`, `LoaderReplayError`, `LoaderAbort`.
  `class DextraLoaderWarning(UserWarning)`.
- **Immutability:** the source is never written. If `source` is already a
  DataFrame (or polars/pyarrow via `_ensure_pandas`), `load` becomes a *typed
  pass-through*: it re-infers types on object columns, reports, and **returns a
  copy** with `attrs` carried — original untouched.
- **Audit:** append to the returned frame's `df.attrs['dextra_audit']` (reuse the
  shared helper — see §8). The audit IS observable here because `load` returns
  the frame (unlike report/dash). Entry: `{stage:"loader", function:"load",
  timestamp, params: <plan>, decision: <sentence>}`.
- **Composition:** when called by `edareport`/`dash`/cleaning, callers pass
  `on_ambiguous="warn"` (or a fixed `params=`) and `interactive=False`; the
  loader never prompts during composition.

---

## 8. Contract hygiene (do it right, set the gold standard)

Per audit findings #4 and #5, **do not re-define `_now_iso`/`_append_audit`
locally** in `_loader.py`. Instead, as part of this slice, add the shared
helpers to `_utils.py` (`now_iso`, `append_audit`, `json_safe`) and import them
here. (Migrating the other modules to them is a separate slice; this module
starts clean.) `load` exposes the full unified flag set, becoming the contract's
reference implementation.

---

## 9. Dependencies

New optional extra in `pyproject.toml`:
```toml
io = ["charset-normalizer>=3.0", "clevercsv>=0.8"]
```
Both imported lazily inside detection with the documented pure-stdlib fallbacks,
so 11.1 works (with reduced detection quality) on the base install. No change to
core dependencies.

---

## 10. Test matrix (`tests/test_phase11.py`)

Fixtures written to `tmp_path` (deliberately broken inputs). No optional libs
required for the core cases (fallbacks must hold); a few cases
`importorskip("charset_normalizer")` / `clevercsv` for the high-quality path.

**Happy / detection**
1. clean utf-8 comma csv → all `confirmed`, no warnings, dtypes correct.
2. semicolon delimiter → `sep=';'` detected (or via clevercsv).
3. cp1256-encoded Arabic headers → encoding detected/fallback; no mojibake.
4. junk preamble (3 title rows before header) → `header_row==3`, skiprows set.
5. locale numerics (`1.234,56`, thousands `.`, decimal `,`) → numeric, correct.
6. currency/percent strings (`$1,200`, `45%`) → numeric + `suggest`/action recorded.
7. ISO + `dd/mm/yyyy` date columns → datetime, `parse_rate>=0.95`.

**Ambiguity / policy**
8. mixed column (70% dates / 30% text) → column `ambiguous`; `warn` warns,
   `raise` raises `LoaderAmbiguityError`, `plan` returns plan + loads nothing.
9. ambiguous delimiter (',' vs ';') → parse-level `ambiguous`.
10. `id` column with failed coercions → `ambiguous-high-risk`.
11. column all-NaN after coercion → `ambiguous-high-risk` + problem entry.

**Contract / replay / security**
12. `return_params=True` → `(df, plan)`; plan is JSON-serialisable (`json.dumps`).
13. `load(src, params=plan)` reproduces an identical frame (`assert_frame_equal`).
14. replay after editing the source file → `source_changed` problem;
    `raise` → `LoaderReplayError`.
15. `.pkl` source with `allow_pickle=False` → `LoaderSecurityError`.
16. immutability: passing a DataFrame returns a copy; original `attrs`/values
    unchanged; audit entry present on the result.
17. alias identity: `dx.dload is dx.load`, `dx.dpeek is dx.peek`.
18. `peek` loads ≤ `n_preview` rows and returns a plan; commits no full frame.
19. ragged rows counted in `problems`; load still succeeds under `warn`.
20. `show=False` prints nothing and never prompts even with `interactive=True`.

Coverage target: keep total ≥ 68% gate; aim `_loader.py` ≥ 85%.

---

## 11. Definition of done (11.1)
- `run_validation.ps1` green (ruff + pytest + coverage) on the owner's machine.
- `load`/`peek` honour every §2 flag; `load` passes the §8 contract surface.
- All §10 cases pass; fallbacks verified with the `io` extra absent.
- `ROADMAP.md` updated (Phase 11 row + "entry layer" note); `CHANGELOG.md` entry.
- No new *required* dependency; `import dextra` unaffected.
