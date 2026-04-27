# dextra

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://github.com/ahmedabdeltawab602-collab/dextra/actions/workflows/tests.yml/badge.svg)](https://github.com/ahmedabdeltawab602-collab/dextra/actions/workflows/tests.yml)

> *Lightweight exploratory-data-analysis helpers built on top of pandas,
> seaborn and plotly.*

**🇦🇷 العربية:** [README.ar.md](README.ar.md) — هذا الملف بالعربية.

`dextra` gives you three small, focused helpers for the first ten minutes
of any data-analysis workflow:

| Function | What it does |
|---|---|
| `describe_numeric` | A richer version of `df.describe()` with IQR, outlier counts, skew/kurtosis, and formatted output. |
| `plot_histograms` | One matplotlib figure per DataFrame, with histogram + KDE on the left and the full stats panel on the right. |
| `plot_boxplots` | Stacked horizontal Plotly box-plots with annotated statistics — interactive, good for reports. |

---

## Installation

From PyPI (once published):

```bash
pip install dextra
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

---

## Quick start

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

# 3) Interactive Plotly box-plots
dx.plot_boxplots(df)
```

### Returning data instead of rendering

Every function has `show=` (controls rendering) and `return_*=` flags so
you can feed the results into another step:

```python
summary = dx.describe_numeric(df, return_df=True, raw=True, show=False)
fig, stats = dx.plot_boxplots(df, return_fig=True, return_df=True, show=False)
```

`raw=True` returns un-formatted `float64` values — use that when you plan
to export to Excel, CSV, or do further math.

---

## API reference

### `describe_numeric(df, cols=None, decimals=2, df_name=None, iqr_multiplier=1.5, metrics_as_rows=True, show=True, return_df=False, raw=False)`

Return a rich numeric summary. 21 metrics per column: mean, std, var,
coefficient of variation, min, Q1, median, mean-vs-median gap, Q3, max,
IQR, Tukey lower/upper bounds, outlier count & %, count, missing, unique,
skewness, kurtosis, modes.

### `plot_histograms(df, cols=None, bins=20, decimals=2, iqr_multiplier=1.5, fig_width=17.0, fig_row_height=4.8, width_ratios=(3, 1), dpi=120, hist_color='skyblue', hist_edgecolor='black', alpha=0.85, kde=True, kde_color='blue', kde_linewidth=2.2, title=..., save=False, output_dir='plots', filename='histograms_with_summary.png', show=True, return_fig=False, return_df=False)`

Matplotlib histograms with KDE overlay and a monospace stats panel next to
each plot. Mean (red dashed) and median (green dash-dot) are highlighted.

### `plot_boxplots(df, cols=None, decimals=2, iqr_multiplier=1.5, width=1400, row_height=350, opacity=0.7, line_color='orange', template='plotly_white', show_grid=True, title='Boxplots', colors=None, show=True, return_fig=False, return_df=False)`

Interactive Plotly box-plots, one row per column, horizontal orientation,
with dashed lines at the Tukey lower/upper bounds and a per-row annotation
summarising the distribution.

### Backwards-compatible aliases

`numdesc`, `hister`, and `boxpl` from the pre-0.1.0 API still work and
simply forward to the new names.

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
