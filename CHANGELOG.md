# Changelog

All notable changes to `dextra` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- `missing_report` - visualize and summarise missing values.
- `correlation_report` - correlation matrix with heatmap and ranked pairs.
- `describe_categorical` - summaries for object/category columns.
- Interactive Plotly version of `plot_histograms`.
- `quick_report` - one-liner that runs every analysis and renders an HTML file.

## [0.1.0] - 2026-04-24

### Added
- Initial public release.
- `describe_numeric` - rich numeric summary with 21 metrics per column
  (mean, std, var, coefficient of variation, IQR, Tukey bounds, outlier
  counts, skewness, kurtosis, modes, and more).
- `plot_histograms` - matplotlib histograms with KDE overlay and a
  monospace statistics panel beside each plot.
- `plot_boxplots` - interactive Plotly box-plots with annotated Tukey
  bounds and per-row statistics.
- Backwards-compatible aliases: `numdesc`, `hister`, `boxpl`.
- `DEFAULT_BOX_COLORS` palette exported at top level.
- Flags on every function: `show=`, `return_df=`, `return_fig=`, `raw=`.

### Dependencies
- Python 3.9+
- numpy >= 1.23
- pandas >= 1.5
- matplotlib >= 3.6
- seaborn >= 0.12
- plotly >= 5.10

[Unreleased]: https://github.com/ahmedabdeltawab602-collab/dextra/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ahmedabdeltawab602-collab/dextra/releases/tag/v0.1.0