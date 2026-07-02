# dextra Phase 8 — Time-Series Philosophy & Framework

> **Purpose.** This document defines the philosophy, the globally-recognised
> framework, and the design constraints for every time-series function we add to
> dextra. It is the blueprint for Phase 8. Any time-series function we ship must
> satisfy this reference; if it does not, we either fix the function or extend
> this document. It is the Phase-8 sibling of `MODELING_PHILOSOPHY.md` and
> `EVALUATION_PHILOSOPHY.md`, and inherits the same return contract (§4.7 of the
> modeling document).

---

## 1. Definition and boundary

**Time-series analysis = understanding the structure of a series observed over
time (trend, seasonality, stationarity) and producing an instant,
validated baseline forecast with a single call.**

dextra time-series is **baseline-first**, exactly as Phase 6 modeling is. Its
job is to expose the structure of one series and to establish *where the
forecasting floor is* — not to be a full forecasting framework. Sophisticated
models (ARIMA, ETS, Prophet, state-space) are deliberately deferred; Phase 8
ships the three diagnostics that every time-series workflow opens with.

Boundaries with adjacent phases:
- **Phase 6 (modeling) — done.** Learns a cross-sectional baseline (rows are
  i.i.d. samples).
- **Phase 7 (evaluation) — done.** Judges a trained model in depth.
- **Phase 8 (this) — time series.** Treats rows as *ordered observations of one
  series* and asks: what is its trend / seasonality, is it stationary, and what
  is the naive forecasting floor?
- **Phase 9 (report) — later.** Aggregates these into a one-call report.

The unit of analysis changes in Phase 8. Phases 6–7 assume **independent rows**;
Phase 8 assumes **one ordered series** where order carries information and the
future must never inform the past.

---

## 2. The non-negotiable principles

### 2.1 Two input modes (the Phase-8 adaptation of fit/apply)
Every time-series function accepts its data in one of two ways:

- **Series mode.** `df + value` (a column name or array-like for the observed
  series), plus an optional `time` (a datetime column / index used for the
  x-axis and for inferring the seasonal period). A pure, self-contained
  analysis of one series.
- **Artifact mode.** `df + params`, where `params` is a descriptor produced by a
  previous Phase-8 call (`return_params=True`). It carries the resolved
  `value`/`time` columns, the inferred `period`, the chosen `model`/`method`,
  and — for `tsfcast` — the fitted baseline constants, so the analysis
  is reproduced on new data **without re-deciding** anything.

### 2.2 No look-ahead — the time-series leak we refuse
The cardinal sin of time-series work is letting the future leak into the past.
dextra refuses it structurally:

- **Validation is always on a held-out *tail* window.** `tsfcast`
  trains each baseline only on observations *before* the last `valid` points and
  measures error on that untouched tail. No metric is ever computed on data the
  baseline saw.
- **No centred statistic is used as a causal feature.** Where a centred moving
  average is shown (trend in `tsdecomp`), it is a *descriptive* decomposition
  of the whole series, never a predictor fed forward.
- **Differencing is reported, not silently applied to forecasts.**
  `tsstat` *suggests* a differencing order; it never mutates the series
  the user passed.

### 2.3 Honesty about validation vs. forward
When a function both validates on history and projects forward, it reports the
two distinctly. `tsfcast` measures error on the held-out window and
then **re-fits the chosen baseline on the full series** for the forward forecast;
the `Decision:` sentence names which window each metric came from.

### 2.4 Immutability
The input DataFrame is never mutated. Time-series functions append nothing to
`df`'s columns; they return a freshly-built result frame (the decomposition
components, the stationarity table, or the forecast frame) and only write to a
*copy*'s `df.attrs['dextra_audit']`.

### 2.5 Don't decide for the user — surface the structure
Time-series functions never declare a series "stationary enough", never pick the
"true" period, never anoint a single forecast as correct. They expose the full
decomposition / the test statistics / the candidate-baseline leaderboard and a
single honest `Decision:` sentence; the user owns the verdict.

### 2.6 Order and index integrity
Before any computation the series is ordered by `time` when given (ties and
gaps surfaced, never silently reindexed). Missing observations are reported, not
imputed away; a function that needs an evenly-spaced series says so rather than
fabricating one.

---

## 3. The three functions and their scope

| Family | dextra function | Task | What it needs | Multi-panel figure |
|--------|--------------------------|------|---------------|--------------------|
| Decomposition | `tsdecomp` | split a series into trend / seasonal / residual | `value` (+ `time`, `period`) | observed + trend + seasonal + residual (4 stacked panels) |
| Stationarity | `tsstat` | test for a unit root / stationarity and suggest differencing | `value` (+ `time`) | series + rolling mean/std + autocorrelation (ACF) |
| Forecast | `tsfcast` | baseline forecast validated on a held-out tail | `value` (+ `time`, `horizon`, `valid`, `period`) | history + validation (actual vs predicted) + forward forecast |

### 3.1 Methods and the optional dependency
- `tsdecomp` defaults to **classical decomposition** (a centred moving-average
  trend, period-averaged seasonal component, additive *or* multiplicative),
  implemented with **numpy / pandas only**. `method="stl"` enables an STL
  decomposition via a **lazy** `statsmodels` import.
- `tsstat` runs the **ADF** (Augmented Dickey–Fuller) and **KPSS** tests
  via a **lazy** `statsmodels` import — the canonical, battle-tested
  implementations. These are imported only inside the function, exactly as
  scikit-learn is in Phases 5–7. If `statsmodels` is not installed the function
  raises a clear, actionable error (`pip install "pydextra[ts]"`).
- `tsfcast` ships four dependency-free baselines — **naive**
  (last value), **snaive** (last season), **drift** (last value + average slope),
  **mean** — plus `method="compare"`, ranked on the held-out window. **No new
  dependency.**

`statsmodels` is therefore introduced as a **new optional extra `ts`**,
lazy-imported, never required by the base install. This preserves the roadmap
rule "no new heavy dependencies without explicit approval" (approved
2026-06-06) and keeps `import dextra` working with numpy / pandas / matplotlib /
scipy alone.

---

## 4. The dextra contract — every Phase 8 function must satisfy this

### 4.1 Dual input mode
```python
# Series mode: just the column (and, ideally, a time column)
dx.tsdecomp(df, value='sales', time='month', period=12)

# Artifact mode: replay resolved settings on new data
res, p = dx.tsdecomp(df_2024, value='sales', time='month',
                         return_params=True)
dx.tsdecomp(df_2025, params=p)        # same value/time/period/model

# Forecast: validate on the tail, then project forward
fc, p = dx.tsfcast(df, value='sales', time='month',
                             horizon=12, method='compare', return_params=True)
```

### 4.2 Universal flags
The standard dextra flags, identical to Phases 6–7:
`params=None`, `return_params=False`, `show`, `plot`, `return_df`, `return_fig`,
`decimals`, `df_name`, plus `fig_width`, `fig_height`, `dpi`.
Per-function inputs: `value`/`time` (all); `period`/`model`/`method`
(`tsdecomp`); `max_diff`/`alpha`/`regression` (`tsstat`);
`horizon`/`valid`/`period`/`method` (`tsfcast`).

### 4.3 Universal outputs (the locked return shape)
- A **result DataFrame** densely packed with the relevant output: the
  component series (`observed`/`trend`/`seasonal`/`resid`) for `tsdecomp`;
  the test-statistic block (ADF stat/p, KPSS stat/p, verdict, suggested `d`) for
  `tsstat`; the per-method validation metrics and the forward forecast
  for `tsfcast`.
- A **multi-panel matplotlib figure** (the table in §3 names the panels).
- A printed one-line **`Decision:`** sentence naming the headline result and the
  window / mode it was measured on.
- An append to a copy's **`df.attrs['dextra_audit']`**.
- When `return_params=True`, a **JSON-safe descriptor** (NOT an estimator):
  `function`, `task`, the resolved `value`/`time`/`period`, the computed
  `metrics`, `metadata` (`n`, `input_mode`, freq, …), `version`,
  `analyzed_at`, and — for `tsfcast` — the chosen `method` and its
  fitted baseline constants so a later artifact-mode call reproduces the forecast.

### 4.4 Return packing order
Identical to Phases 6–7 via the shared `_ret_pack`: `df_or_report`, `params`,
`fig`. The leading element is the **result frame** (components / tests /
forecast); `return_df=True` yields it.

### 4.5 Idempotency
Analysing the same series twice yields an identical result and figure.
`tsfcast` is fully deterministic (its baselines carry no randomness);
artifact mode reproduces the forward forecast exactly.

---

## 5. Anti-patterns we will refuse

| Anti-pattern | Why it is fatal |
|---|---|
| Computing forecast error on data the baseline was fit on | In-sample "accuracy" is not generalisation; the held-out tail exists for exactly this. |
| Using a centred moving average as a forward predictor | A centred window peeks at future points — textbook look-ahead leakage. |
| Silently differencing the user's series | Differencing changes the data's meaning; we *suggest* `d`, never mutate. |
| Declaring a series "stationary" from one test | ADF and KPSS test complementary nulls; we report both and the conflict cases. |
| Imputing gaps to fake an evenly-spaced series | Hides data-quality problems; we surface gaps and let the user fix them. |
| Inventing a seasonal period without justification | The period is inferred from the time index / ACF and always reported, never hard-coded silently. |
| Mutating `df` (e.g. adding a `trend` column to the input) | Time-series analysis is read-only; it returns a new frame. |

---

## 6. Where time series sits in the professional pipeline
```
Cleaned (P3) -> Engineered (P4)
        |
   One ordered series (value over time)
        |
   DECOMPOSE        tsdecomp     -> trend / seasonal / residual structure   <- Phase 8
        |
   DIAGNOSE         tsstat  -> ADF / KPSS + suggested differencing d
        |
   Hold out the tail window
        |
   BASELINE FORECAST tsfcast -> naive / snaive / drift / mean, ranked
        |
   (later) ARIMA / ETS / Prophet, then deep evaluation (P7) on the forecast
        |
   One-call report (HTML/PDF)                                                   <- Phase 9
```

---

## 7. Reference tools we draw from

| Tool | Idea we adopt |
|------|----------------|
| **statsmodels** | `seasonal_decompose` / `STL`, and the `adfuller` / `kpss` unit-root tests — consumed lazily via the `ts` extra, never hidden. |
| **sktime / Darts** | The baseline-forecaster mindset (naive, seasonal-naive, drift, mean) and held-out backtesting on a tail window. |
| **pandas** | Datetime index, `asfreq`/period inference, rolling windows, autocorrelation. |
| **Hyndman, *Forecasting: Principles and Practice*** | MASE as the scale-free accuracy metric; seasonal-naive as the honest benchmark every model must beat. |

---

## 8. Phase 8 staged delivery plan
Three staged deliveries, each verified before the next begins:

| Stage | Function | Family | Dependency |
|-------|----------|--------|------------|
| 8.1 | `tsdecomp` | decomposition | none (classical) / lazy statsmodels (STL) |
| 8.2 | `tsstat` | stationarity | lazy statsmodels (ADF / KPSS) |
| 8.3 | `tsfcast` | baseline forecast | none |

Every public name is short and underscore-free (`tsdecomp` / `tsstat` /
`tsfcast`), consistent with the short-name spirit of Phases 2–7. Coverage
stays above the 68% gate; `run_validation.ps1` runs green before every commit.

---

## 9. The golden test for any Phase 8 function
A function passes the philosophy when:
1. It accepts BOTH series mode (`value`/`time`) AND artifact mode (`params`),
   and resolves the series correctly from each.
2. It never lets the future inform the past: validation is on a held-out tail,
   and no centred statistic is used as a forward predictor.
3. It reports its result densely as a DataFrame AND as a multi-panel figure.
4. The original DataFrame is never mutated; differencing is suggested, not applied.
5. Validation-window and forward signals, when both exist, are labelled distinctly.
6. The decision sentence names the headline result and the window / mode.
7. `return_params=True` returns a JSON-safe descriptor (no estimator); artifact
   mode reproduces the analysis identically.
8. Any new dependency (`statsmodels`) is optional, lazy-imported, and degrades to
   a clear, actionable error when absent.

Any function violating any of these is rewritten before shipping.

---

## Update history
| Date       | Note                                                          |
|------------|----------------------------------------------------------------|
| 2026-06-06 | Document created as Phase 8 blueprint. 3 functions confirmed: tsdecomp / tsstat / tsfcast. Two-input-mode contract (series / artifact) locked; return shape inherits §4.7. `statsmodels` approved as an optional, lazy `ts` extra (ADF/KPSS + STL); classical decomposition and all forecast baselines stay dependency-free. |
| 2026-06-06 | Phase 8 delivered: `tsdecomp` (classical / lazy-STL decomposition), `tsstat` (ADF/KPSS + iterative suggested differencing), `tsfcast` (naive/snaive/drift/mean baselines + compare, tail-validated, no look-ahead). Underscore-free names; 60 tests. |
