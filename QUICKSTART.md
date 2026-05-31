# dextra — Quick Start (Phase 2)

Every function follows the same pattern:

```python
import dextra as dx
result = dx.<function>(df, ...)   # tabular result + visual + decision sentence
```

Common flags across the library: `show=True`, `plot=True`, `return_df=False`,
`return_fig=False`, `decimals=4`, `alpha=0.05`. Short aliases exist for every
function (shown in the second column).

---

## Stage 1 — Descriptive extensions

### 1. `z_scores` (alias: `zsc`)

Standardised distances from the mean with per-sigma counts.

```python
dx.z_scores(df)
dx.z_scores(df, cols=['price', 'qty'], threshold=2.5)
zs_summary = dx.z_scores(df, return_df=True, show=False, plot=False)
all_zs = dx.z_scores(df, return_zscores=True, show=False, plot=False)
```

### 2. `pearson_skewness` (alias: `pskew`)

Karl Pearson's coefficient `3·(mean − median)/σ`, plus Fisher's for comparison.

```python
dx.pearson_skewness(df)
```

### 3. `empirical_rule_check` (alias: `emprule`)

Actual vs theoretical 68 / 95 / 99.7 coverage per column.

```python
dx.empirical_rule_check(df)
dx.empirical_rule_check(df, tolerance=2.0)   # stricter normality check
```

### 4. `outliers_report` (alias: `outrep`)

Row-level outliers via IQR fence or Z-score.

```python
dx.outliers_report(df)                              # IQR fence, k=1.5
dx.outliers_report(df, method='zscore', z_threshold=3.0)
flagged_rows = dx.outliers_report(df, return_rows=True, show=False, plot=False)
```

---

## Stage 2 — Bivariate analysis

### 5. `correlation_matrix` (alias: `corrmat`)

Pearson / Spearman / Kendall correlation + p-values + significance marks +
heatmap.

```python
dx.correlation_matrix(df)
dx.correlation_matrix(df, method='spearman')
r, p = dx.correlation_matrix(df, return_p=True, show=False, plot=False)
```

### 6. `simple_linear_regression` (alias: `slr`)

Y = mX + b with full diagnostics (residual normality, CI band, R²).

```python
dx.simple_linear_regression(df, x='hours', y='score')
summary, residuals = dx.simple_linear_regression(
    df, x='hours', y='score',
    return_df=True, return_residuals=True, show=False, plot=False,
)
```

---

## Stage 3 — EDA market tools

### 7. `missing_report` (alias: `missrep`)

Per-column missing counts, percentages, dtype, sample value, and a heuristic
recommendation (`OK` / `impute_mean` / `impute_median` / `impute_mode` /
`review` / `consider_drop` / `drop_column`).

```python
dx.missing_report(df)
```

### 8. `frequency_table` (alias: `freqtab`)

Counts + percentages + cumulative percentages for a categorical column, with a
Pareto chart.

```python
dx.frequency_table(df, 'category')
dx.frequency_table(df, 'category', top_n=10)   # collapse the rest into <other>
```

### 9. `cross_tab` (alias: `xtab`)

Contingency table + chi-square test + Cramér's V, with observed-vs-expected
heatmaps.

```python
dx.cross_tab(df, row='gender', col='product')
table, test = dx.cross_tab(df, row='gender', col='product',
                            show=False, plot=False, return_test=True)
```

### 10. `group_compare` (alias: `gcmp`)

Per-group descriptives + violin/box plots overlaid.

```python
dx.group_compare(df, group_col='region', value_cols=['sales', 'margin'])
```

---

## Stage 4 — Inference (CI + sample size)

### 11. `confidence_interval_mean` (alias: `cim`)

t-based CI for the mean. Forest plot.

```python
dx.confidence_interval_mean(df['weight'])
dx.confidence_interval_mean([8, 12, 10, 9, 11], confidence=0.99)
```

### 12. `confidence_interval_proportion` (alias: `cip`)

Wilson (default) or Wald CI for a proportion.

```python
dx.confidence_interval_proportion(35, 100)                  # Wilson
dx.confidence_interval_proportion(35, 100, method='wald')
```

### 13. `sample_size_mean` (alias: `ssm`)

`n = (z · σ / E)²` with an n-vs-E curve.

```python
dx.sample_size_mean(margin_error=2.0, std=10.0, confidence=0.95)
```

### 14. `sample_size_proportion` (alias: `ssp`)

`n = z² · p · (1−p) / E²` with n-vs-E and n-vs-p panels.

```python
dx.sample_size_proportion(margin_error=0.05)               # worst-case p=0.5
dx.sample_size_proportion(margin_error=0.05, p=0.2)
```

---

## Stage 5 — Hypothesis tests

### 15. `normality_test` (alias: `normtest`)

Shapiro-Wilk (n < 5000) or D'Agostino-Pearson. Q-Q + histogram with Normal fit.

```python
dx.normality_test(df['returns'])
dx.normality_test(df['returns'], method='normaltest')
```

### 16. `t_test_one_sample` (alias: `t1`)

H0: mean = popmean.

```python
dx.t_test_one_sample(df['height'], popmean=170)
dx.t_test_one_sample(df['height'], popmean=170, alternative='greater')
```

### 17. `t_test_two_sample` (alias: `t2`)

Welch's t-test by default (`equal_var=False`).

```python
dx.t_test_two_sample(df[df.group == 'A']['score'],
                     df[df.group == 'B']['score'])
```

### 18. `t_test_paired` (alias: `tpair`)

Before/after for matched pairs.

```python
dx.t_test_paired(df['pre'], df['post'])
```

### 19. `anova_oneway` (alias: `aov1`)

F-statistic + eta² for groups.

```python
dx.anova_oneway(df, group_col='diet', value_col='weight_loss')
```

### 20. `chi_square_independence` (alias: `chi2ind`)

Chi-square distribution + Cramér's V + observed/expected counts.

```python
dx.chi_square_independence(df, row='region', col='product')
```

---

## Stage 6 — ML diagnostic tools

### 21. `vif_scores` (alias: `vif`)

Multicollinearity check (`VIF = 1 / (1 − R²)`) with traffic-light coloring.

```python
dx.vif_scores(df[['x1', 'x2', 'x3', 'x4']])
dx.vif_scores(df, threshold=5)
```

### 22. `class_imbalance` (alias: `imbalance`)

Class balance + severity tag + recommendation.

```python
dx.class_imbalance(df['target'])
```

---

## A 60-second tour

```python
import dextra as dx
import pandas as pd

df = pd.read_csv("your_data.csv")

# 1) Quality + descriptive
dx.missing_report(df)
dx.describe_numeric(df)
dx.outliers_report(df)

# 2) Relationships
dx.correlation_matrix(df)
dx.simple_linear_regression(df, x='age', y='income')

# 3) Test a claim
dx.t_test_one_sample(df['score'], popmean=70)
dx.anova_oneway(df, group_col='region', value_col='sales')

# 4) Before modeling
dx.vif_scores(df[['x1', 'x2', 'x3']])
dx.class_imbalance(df['target'])
```

Each call gives you: a printed summary table, a multi-panel figure, and a
one-line decision sentence — without you writing more code.
