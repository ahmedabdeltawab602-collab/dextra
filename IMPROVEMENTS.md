# dextra — Suggested improvements to existing functions

> Status: **suggestions only — not applied**. The original prompt asked us not
> to modify existing functions directly. Each item below explains the issue,
> the impact, and the suggested change. Each can be picked up individually
> later without disturbing the rest of the library.

---

## 1. `describe_numeric.modes` becomes unreadable on continuous data

**Observation.** When every value in a column is unique (common for floats),
`describe_numeric` reports every observation as a "mode". For a 500-row column
this produces a multi-thousand-character row that destroys the table layout in
the terminal (seen during the very first verification run, where the `modes`
row stretched across the screen).

**Suggested fix (any one of these is enough):**

1. Cap the mode list at, say, 5 values and append `… (N more)`.
2. Skip the mode entirely (display `"-"`) when
   `n_unique == n_rows` — every value is its own "mode", so the result is
   meaningless.
3. Detect "all unique" and show `n_modes = n` with no enumeration.

Code sketch (drop-in for `stats.py`):

```python
modes_raw[c] = values
n_unique = s.nunique(dropna=True)
if not values or n_unique == len(s):
    modes_str[c] = "-"            # every value is unique
elif len(values) > 5:
    head = " | ".join(f"{x:,.{decimals}f}" for x in values[:5])
    modes_str[c] = f"{head} | … ({len(values) - 5} more)"
else:
    modes_str[c] = " | ".join(f"{x:,.{decimals}f}" for x in values)
```

---

## 2. `describe_numeric` printed layout breaks in plain terminals

**Observation.** Outside Jupyter (`_ipy_display` is None), the function falls
back to `print(frame.to_string())`. With wide columns (especially the bloated
`modes` column above) the output wraps weirdly and the headers separate from
the data.

**Suggested fix.** Two layers:

1. Fix #1 above shrinks the table dramatically by itself.
2. Set `pd.set_option('display.max_colwidth', 60)` inside the fallback path,
   then restore the previous value with a `try / finally`. That keeps any
   single column from blowing up the layout.

---

## 3. `plot_histograms` default `bins=20` is fixed

**Observation.** `bins=20` is fine for many small samples but is a poor choice
for skewed or large samples. The current API forces users to think about bins
on every call.

**Suggested fix.** Accept a string and forward to numpy / seaborn:

```python
def plot_histograms(df, cols=None, bins='auto', ...):
    ...
    sns.histplot(x=series, bins=bins, ...)   # numpy supports 'auto', 'fd',
                                             # 'sturges', 'sqrt', 'rice'
```

`bins='auto'` lets numpy pick between Freedman-Diaconis and Sturges based on
the data. The default integer behavior would still work because numpy accepts
both.

---

## 4. `plot_boxplots` side-panel omits the median in the annotation

**Observation.** Looking at the annotation rendered by `plot_boxplots`, the
side panel shows `mean`, `Q1`, `Q3`, `LB`, `UB`, `min`, `max`, `outliers`. The
**median** is plotted on the box but is **not in the text panel**, even though
it is one of the most common readings users compare against the mean.

**Suggested fix.** Add a line in the `txt = ...` block in `plots.py`:

```python
txt = "<br>".join([
    f"mean = {format_value(mean_val, 'num', decimals)}",
    f"median = {format_value(median_val, 'num', decimals)}",   # NEW
    f"Q1 = ...",
    ...
])
```

`median_val` is already computed a few lines above.

---

## 5. `describe_numeric` returns sample variance / sample std with no flag

**Observation.** `pandas.Series.var()` and `.std()` use `ddof=1` (sample). The
course distinguishes **population** σ² (denominator N, F-M03-L04-01) from
**sample** s² (denominator n−1, F-M03-L04-02). The current API only exposes
the sample formula. Most users want sample (Bessel's correction), but the
distinction matters for instruction.

**Suggested fix.** Add `ddof: int = 1` as a parameter and pass it through:

```python
def describe_numeric(df, ..., ddof: int = 1):
    ...
    std_ = num_col.std(ddof=ddof)
    var_ = num_col.var(ddof=ddof)
```

Default `ddof=1` keeps backward compatibility; `ddof=0` is for population data.

---

## 6. `numdesc`, `hister`, `boxpl` aliases are functional but not discoverable

**Observation.** The short aliases are perfectly maintained, but a new user
reading `__all__` doesn't see why two names exist or what's deprecated.

**Suggested fix.** No code change — add one line to each alias's docstring:

```python
numdesc = describe_numeric   # short alias kept for backward compatibility
```

This is purely documentation hygiene. The current setup is correct.

---

## 7. `format_value` fails on `numpy.bool_` (pandas 3.x default)

**Observation.** Discovered during Stage 3 testing: in pandas 3.x, columns that
hold mixed Python objects can take dtype `str` (rather than `object`), and
non-numeric values hit the numeric branch of `format_value`. We already fixed
this for `_format_summary` in `stats_advanced.py` by catching
`TypeError`/`ValueError` and falling back to `str(v)`. The same change should
be applied to `format_value` in `_utils.py` to harden the legacy formatter.

**Suggested fix:** mirror the `_format_summary._fmt` logic from
`stats_advanced.py` line 69 into `_utils.format_value`.

---

## 8. Add `dx.help()` or `dx.functions()` for discoverability

**Observation.** The library now ships 22+ public functions and 22+ aliases.
A new user has no way to print "what does this library offer?" short of
reading `__all__` or this Quick Start.

**Suggested fix.** A small helper:

```python
def help_index():
    """Print every public function in dextra with its one-line summary."""
    import dextra as dx
    for name in dx.__all__:
        obj = getattr(dx, name)
        if callable(obj):
            doc = (obj.__doc__ or "").strip().split("\n")[0]
            print(f"{name:<28} {doc}")
```

Adds zero dependencies, useful for both new users and demos.

---

## Priority ranking (if/when picked up)

1. **High:** #1 (modes display), #7 (format_value robustness).
   Both currently cause visible glitches.
2. **Medium:** #4 (median in boxplot annotation), #2 (terminal layout).
   Quality-of-life issues.
3. **Nice to have:** #3 (auto bins), #5 (ddof flag), #6 (alias docs),
   #8 (help_index).

None are blockers for the 22 functions just delivered.
