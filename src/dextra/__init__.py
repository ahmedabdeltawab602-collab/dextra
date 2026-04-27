"""dextra — lightweight exploratory-data-analysis helpers.

A small, opinionated toolkit that wraps ``pandas`` / ``seaborn`` / ``plotly``
to give you a richer numeric summary and nicer default plots than the
ones shipped with the libraries themselves.

Quick start
-----------
>>> import dextra as dx
>>> import pandas as pd
>>> df = pd.read_csv("some_data.csv")
>>> dx.describe_numeric(df)
>>> dx.plot_histograms(df)
>>> dx.plot_boxplots(df)

The legacy short aliases ``numdesc`` / ``hister`` / ``boxpl`` are
still exposed so existing notebooks keep working.
"""

from ._version import __version__
from ._utils import DEFAULT_BOX_COLORS
from .stats import describe_numeric, numdesc
from .plots import plot_histograms, plot_boxplots, hister, boxpl

__all__ = [
    "__version__",
    "DEFAULT_BOX_COLORS",
    # New, descriptive names
    "describe_numeric",
    "plot_histograms",
    "plot_boxplots",
    # Backward-compatible aliases
    "numdesc",
    "hister",
    "boxpl",
]
