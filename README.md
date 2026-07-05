# dextra

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/ahmedabdeltawab602-collab/dextra/actions/workflows/tests.yml/badge.svg)](https://github.com/ahmedabdeltawab602-collab/dextra/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/pydextra.svg)](https://pypi.org/project/pydextra/)
[![Docs](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://ahmedabdeltawab602-collab.github.io/dextra/)

**Data functions that explain themselves.** One line in → a rich metrics
table, a multi-panel figure, and a one-sentence `Decision:` stating what was
done and why. Leakage-sensitive steps also return a replayable `params`
artifact — fit on train, replay verbatim on test — so train/test leakage
becomes hard to commit by accident.

One real call, verbatim from the
[leakage-safe pipeline notebook](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/02-leakage-safe-pipeline.ipynb):

```python
train_fe, params = dx.featpipe(train, steps=steps, return_params=True)
```

```text
Decision: Fitted a 3-step featpipe pipeline (handle_missing -> encode -> scale);
33 new column(s) produced; combined params is a versioned, JSON-serialisable
artifact. Apply to held-out data with featpipe(df_test, params=...).
```

```bash
pip install pydextra
```

> **Two names, one library:** the name `dextra` was unavailable on PyPI, so
> you **install `pydextra`** and **import `dextra`**. Repo, docs and import
> are all `dextra`.

[**Documentation**](https://ahmedabdeltawab602-collab.github.io/dextra/) ·
[**Open in Colab**](https://colab.research.google.com/github/ahmedabdeltawab602-collab/dextra/blob/main/notebooks/00-leakage-in-5-minutes.ipynb) ·
[**🇪🇬 العربية**](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/README.ar.md)

`dextra` is a choice-first toolkit for the whole exploratory workflow — a
vocabulary of **63 public functions** plus 5 scikit-learn-compatible wrapper
classes, **68 distinct public callables in all** (long names also ship short
aliases; see the
[API consistency table](https://ahmedabdeltawab602-collab.github.io/dextra/api-consistency/)):
the `load` entry layer at the front, nine analytical modules (EDA, statistics,
cleaning, features, selection, modeling, evaluation, time series and
reporting), and the `compat` wrappers:

| Module | Functions | What it covers |
|---|---|---|
| `_loader` (entry layer) | `load`, `peek` (aliases `dload`, `dpeek`) | Raw, messy CSV / TSV / Excel / JSON / Parquet → a typed DataFrame with full, replayable disclosure. |
| `stats` / `plots` | `describe_numeric`, `plot_histograms`, `plot_boxplots` | Foundational EDA: rich summaries and better default plots. |
| `stats_advanced` | 22 functions (z-scores, skewness, correlation, SLR, CIs, t-tests, ANOVA, chi-square, VIF, class imbalance, ...) | Descriptive, bivariate, inference, hypothesis tests and ML diagnostics. |
| `cleaning` | 10 functions (`clean_report`, `handle_missing`/`impute`, `dedupe`, `clip_outliers`/`winsor`, ...) | Data-quality auditing and cleaning across the DAMA-DMBOK stages; leakage-safe fit/apply on `handle_missing` and `clip_outliers`. |
| `features` | 8 functions (`transform`, `scale`, `bin`, `encode`, `dtfeats`, `cross`, `aggfeat`, `featpipe`) | Leakage-safe feature engineering with a fit/apply contract. |
| `selection` | 5 functions (`redundancy`, `relevance`, `importance`, `rfe`, `selectpipe`) | Filter / Embedded / Wrapper feature selection, leakage-safe. |
| `modeling` | 3 functions (`regress`, `classify`, `cluster`) | Instant baselines: fit / apply / compare with a hybrid (JSON + fitted estimator) artifact. |
| `evaluation` | `confusion_report`, `roc_pr`, `residual_analysis`, `learning_curves` | Deep multi-metric model evaluation (label or saved-artifact mode). |
| `timeseries` | `tsdecomp`, `tsstat`, `tsfcast` | Decomposition, stationarity tests, baseline forecasting. |
| `report` | `edareport` | One-call, self-contained HTML EDA report (composes Phases 1-8). |
| `dashboard` | `dash` | Generates a self-contained, interactive Streamlit app. |
| `compat` | `DextraFeaturePipeline` / `DextraSelectPipeline`, `DextraRegressor` / `DextraClassifier` / `DextraClusterer` | scikit-learn-compatible wrappers that drop into `Pipeline` / `GridSearchCV`. |

Run `dx.functions()` to print the whole public API with one-line summaries.
Most functions expose a `method='compare'` mode that ranks every option and
**decides nothing for you**, and a `fit/apply` mode (via a `params` dict) that
learns on training data and replays verbatim on held-out data -- the safeguard
against data leakage.

---

## Project status

**Feature-complete and API-stable since 0.6.0 — frozen by policy, not abandoned.**

- The public API is locked: no new features, no signature changes. What you
  evaluate today is what you run next year.
- **Bug reports are read and answered.** Confirmed defects are fixed
  red-test-first and released as patch versions — open one via the
  [issue templates](https://github.com/ahmedabdeltawab602-collab/dextra/issues/new/choose).
- Before 0.6.0 the library went through an adversarial external audit:
  eleven evidence-backed defects, each closed red→green with a named
  regression test — see
  [`AUDIT_REPORT.md`](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/AUDIT_REPORT.md)
  and the [changelog](https://github.com/ahmedabdeltawab602-collab/dextra/blob/main/CHANGELOG.md).
- Known, deliberately declared debt is public:
  [#1 — relevance scoring over complete rows](https://github.com/ahmedabdeltawab602-collab/dextra/issues/1).

---

## Installation

From PyPI:

```bash
pip install pydextra   # import name is unchanged: import dextra as dx
```

Directly from GitHub:

```bash
pip install git+https://github.com/ahmedabdeltawab602-collab/dextra.git
```

From source, in development mode:

```bash
git clone https://github.com/ahmedabdeltawab602-collab/dextra.git
cd dextra
pip install -e ".[dev]"
```

### Optional extras

The core install is lightweight (numpy, pandas, matplotlib, seaborn, scipy).
Enable the rest as needed:

```bash
pip install "pydextra[io]"       # charset-normalizer + clevercsv + openpyxl: best loader detection + Excel
pip install "pydextra[ml]"       # scikit-learn: regress / classify / cluster, model-based selectors, dextra.compat
pip install "pydextra[viz]"      # plotly: interactive plot_boxplots
pip install "pydextra[ts]"       # statsmodels: time-series STL + ADF / KPSS
pip install "pydextra[dash]"     # streamlit: the generated interactive dashboard
pip install "pydextra[perf]"     # polars + pyarrow: alternative DataFrame backends
pip install "pydextra[notebook]" # jupyter + ipykernel
pip install "pydextra[docs]"     # mkdocs-material site
```

### scikit-learn interoperability

dextra's pipelines and models also expose the standard scikit-learn API via
`dextra.compat` (`DextraFeaturePipeline`, `DextraSelectPipeline`,
`DextraRegressor`, `DextraClassifier`, `DextraClusterer`), so they drop
directly into `sklearn.pipeline.Pipeline` and `GridSearchCV`.

---

## Quick start

Most real workflows begin at the **loader** -- it turns a messy file into a
typed, fully documented DataFrame:

```python
import dextra as dx

df = dx.load("your_data.csv")   # encoding + delimiter + per-column type inference, replayable
```

The rest of this quick start uses a small in-memory frame so it runs as-is:

```python
import pandas as pd
import numpy as np
import dextra as dx

rng = np.random.default_rng(42)
df = pd.DataFrame({
    "price":    rng.normal(100, 15, 500),
    "quantity": rng.integers(1, 20, 500),
    "score":    rng.beta(2, 5, 500) * 100,
})

# 1) Rich numeric summary
dx.describe_numeric(df)

# 2) Histograms with side-by-side statistics
dx.plot_histograms(df, bins=30)

# 3) Interactive Plotly box-plots  -- needs the viz extra:  pip install "pydextra[viz]"
dx.plot_boxplots(df)
```

### Returning data instead of rendering

Every function has `show=` (controls rendering) and `return_*=` flags so
you can feed the results into another step:

```python
summary = dx.describe_numeric(df, return_df=True, raw=True, show=False)
fig, stats = dx.plot_boxplots(df, return_fig=True, return_df=True, show=False)  # needs pydextra[viz]
```

`raw=True` returns un-formatted `float64` values — use that when you plan
to export to Excel, CSV, or do further math.

---

## API reference

### `describe_numeric(df, cols=None, decimals=2, df_name=None, iqr_multiplier=1.5, ddof=1, metrics_as_rows=True, show=True, return_df=False, raw=False)`

Return a rich numeric summary. 21 metrics per column: mean, std, var,
coefficient of variation, min, Q1, median, mean-vs-median gap, Q3, max,
IQR, Tukey lower/upper bounds, outlier count & %, count, missing, unique,
skewness, kurtosis, modes.

### `plot_histograms(df, cols=None, bins='auto', decimals=2, iqr_multiplier=1.5, fig_width=17.0, fig_row_height=4.8, width_ratios=(3, 1), dpi=120, hist_color='skyblue', hist_edgecolor='black', alpha=0.85, kde=True, kde_color='blue', kde_linewidth=2.2, title=..., save=False, output_dir='plots', filename='histograms_with_summary.png', show=True, return_fig=False, return_df=False)`

Matplotlib histograms with KDE overlay and a monospace stats panel next to
each plot. Mean (red dashed) and median (green dash-dot) are highlighted.

### `plot_boxplots(df, cols=None, decimals=2, iqr_multiplier=1.5, width=1400, row_height=350, opacity=0.7, line_color='orange', template='plotly_white', show_grid=True, title='Boxplots', colors=None, show=True, return_fig=False, return_df=False)`

Interactive Plotly box-plots, one row per column, horizontal orientation,
with dashed lines at the Tukey lower/upper bounds and a per-row annotation
summarising the distribution.

### Discoverability and aliases

Call `dx.functions()` to list every public function with its one-line summary.
Long descriptive names each have a short alias (e.g. `numdesc`, `zsc`, `corrmat`,
`impute`, `redun`, `selpipe`); the pre-0.1.0 names still forward to the new ones.

For the full per-module API, see the philosophy documents in the repository
(`FEATURES_PHILOSOPHY.md`, `SELECTION_PHILOSOPHY.md`, `CLEANING_PHILOSOPHY.md`)
and the three end-to-end notebooks indexed in
[`notebooks/README.md`](notebooks/README.md) -- messy-CSV rescue,
leakage-safe pipelines, and a full Arabic EDA on Egyptian food prices.

---

## Development

```bash
git clone https://github.com/ahmedabdeltawab602-collab/dextra.git
cd dextra
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

---

## Contributing

Pull requests welcome. Please run `pytest` and `ruff check .` before
opening one.

---

## License

[MIT](LICENSE) © 2026 Ahmed Abd El Tawab
