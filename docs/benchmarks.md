# Benchmarks

Two complementary suites track dextra's performance:

- **Macro** (`benchmarks/macro_bench.py`) -- the documented one-liner
  workflow on a **~1M-row** synthetic frame: load, the full cleaning chain
  and the feature pipeline. Run manually; measured results below.
- **Micro** (`benchmarks/test_bench.py`) -- pytest-benchmark timings of hot
  paths at 5k rows, tracked in CI by the non-blocking `benchmarks` job
  (`pytest benchmarks/ --benchmark-only`).

## Macro: ~1M rows end-to-end

The data is synthetic but realistically dirty: messy column names, padded
and case-varying categories, dates and yes/no flags stored as text, ~2%
missing values in two columns, ~0.1% extreme outliers and ~1% duplicated
rows -- 1,010,000 rows x 7 columns, a 49 MB CSV. Every step runs in silent
mode (`show=False, plot=False`), one cold pass per step, wall-clock timed
with `time.perf_counter`, on:

```
dextra 0.4.0 | python 3.13.13 | pandas 3.0.3 | numpy 2.4.5 | Windows | 24 logical CPUs
```

| step | seconds | rows in | rows/s |
|---|---:|---:|---:|
| load (csv, measured inference) | 54.75 | 1,010,000 | 18,449 |
| audit (clean_report) | 0.97 | 1,010,000 | 1,042,590 |
| tidycols (standardize_columns) | 0.14 | 1,010,000 | 7,454,067 |
| recast (cast_types, auto) | 1.44 | 1,010,000 | 702,122 |
| impute (handle_missing, auto) | 0.09 | 1,010,000 | 11,432,452 |
| dedup (dedupe) | 0.69 | 1,010,000 | 1,463,564 |
| winsor (clip_outliers, income) | 0.04 | 1,000,000 | 27,559,825 |
| verify (validate_rules, 2 rules) | 0.03 | 1,000,000 | 30,694,902 |
| featpipe (scale + one-hot) | 0.37 | 1,000,000 | 2,678,974 |

### Reading the numbers

- **The cleaning and feature chain is pandas-fast.** After loading, the full
  chain -- `audit` → `tidycols` → `recast` → `impute` → `dedup` → `winsor` →
  `verify` → `featpipe` -- completes in **≈ 3.8 s combined** for a million
  rows. The summary-table / decision-sentence / audit-trail layer adds no
  meaningful overhead on top of the underlying pandas operations.
- **`load` dominates, by design.** Its ~55 s is the price of the loader's
  contract on a million rows of raw text: parsing plus *measured* per-column
  type inference with full disclosure, paid once at ingestion. Already-typed
  formats avoid the text-parsing part by design (the parquet path confirms
  typed columns instead of re-parsing them); `dx.peek` previews a source
  without committing to a full load.
- Numbers are a single cold pass on the maintainer's machine and will vary
  with hardware; treat them as orders of magnitude, not guarantees.

### Reproducing

```bash
python benchmarks/macro_bench.py                 # 1,000,000 rows (~1 min + setup)
python benchmarks/macro_bench.py --rows 100000   # quicker sanity run
```

The script writes a temporary CSV (removed afterwards), prints the
environment line and a ready-to-paste markdown table, and never fails the
run: a step that raises is reported as `FAILED` and timing continues.

## Memory: the loader's transient peak

`load` spends memory to keep ingestion transparent. Measured on the same
1.01M-row, 48.6 MB CSV used above (`python 3.13`, `pandas 3.0.3`):

| path | transient peak | x final frame | process RSS delta |
|---|---:|---:|---:|
| default | 1,322 MB | 13.7x | +313 MB |
| `low_memory=True` | 869 MB | 9.0x | +249 MB |

The loaded frame is only ~97 MB (about 2x the CSV). The transient **peak**
is higher because measured per-column inference reads the whole source
*twice* in the default path: once as a Python list-of-lists (the `csv` parse
used to locate the header row and count ragged rows) and once as an
object-dtype DataFrame (the raw values that inference then types). Both
full-width copies are alive while the typed columns are built, so the
high-water mark lands near 13-14x the final frame for a plain CSV. It is
transient -- memory falls back to the frame's own footprint once `load`
returns -- but it is the loader's real ceiling on a large file.

### `low_memory=True`

Passing `low_memory=True` streams the source **once** instead of
materialising every cell as a Python list-of-lists: it keeps only a bounded
header sample plus a per-row field-count tally (integers), which is all that
header-row and ragged-row detection actually require. The larger of the two
transient copies is never built, so the peak drops by about a third (1,322
-> 869 MB here; 13.7x -> 9.0x the frame) and the process RSS delta falls
with it (+313 -> +249 MB). The returned frame **and** the load plan are
identical to the default path -- per-column measured inference is unchanged
-- so the flag is a pure space/time trade with no effect on the result. It
applies to delimited text (CSV/TSV); already-typed sources (Excel / parquet
/ JSON / SQL) ignore it.

Measure it yourself with `tracemalloc` around a load:

```python
import tracemalloc, dextra as dx
tracemalloc.start()
df = dx.load("big.csv", show=False)        # add low_memory=True to compare
_, peak = tracemalloc.get_traced_memory()
print(round(peak / 1e6), "MB transient peak")
```
