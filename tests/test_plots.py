from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import pytest

from dextra import plot_boxplots, plot_histograms


def test_plot_histograms_returns_figure(numeric_df):
    fig = plot_histograms(numeric_df, show=False, return_fig=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_histograms_returns_summary(numeric_df):
    summary = plot_histograms(numeric_df, show=False, return_df=True)
    assert isinstance(summary, pd.DataFrame)
    assert set(numeric_df.columns).issubset(summary.index)


def test_plot_histograms_returns_both(numeric_df):
    fig, summary = plot_histograms(
        numeric_df, show=False, return_fig=True, return_df=True
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(summary, pd.DataFrame)
    plt.close(fig)


def test_plot_histograms_bins_validated(numeric_df):
    with pytest.raises(ValueError):
        plot_histograms(numeric_df, bins=0, show=False)


def test_plot_boxplots_returns_figure(numeric_df):
    fig = plot_boxplots(numeric_df, show=False, return_fig=True)
    assert isinstance(fig, go.Figure)


def test_plot_boxplots_returns_summary(numeric_df):
    summary = plot_boxplots(numeric_df, show=False, return_df=True)
    assert isinstance(summary, pd.DataFrame)
    for expected in ("mean", "median", "q1", "q3", "outliers_count"):
        assert expected in summary.columns


def test_plot_boxplots_custom_colors_list(numeric_df):
    fig = plot_boxplots(
        numeric_df,
        colors=["#111111", "#222222", "#333333"],
        show=False,
        return_fig=True,
    )
    assert isinstance(fig, go.Figure)


def test_plot_boxplots_custom_colors_mapping(numeric_df):
    fig = plot_boxplots(
        numeric_df,
        colors={"a": "#111111", "b": "#222222"},  # missing 'c' falls back
        show=False,
        return_fig=True,
    )
    assert isinstance(fig, go.Figure)
