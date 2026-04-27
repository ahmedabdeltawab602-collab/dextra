from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dextra import describe_numeric


def test_returns_none_by_default(numeric_df):
    assert describe_numeric(numeric_df, show=False) is None


def test_returns_dataframe_when_requested(numeric_df):
    out = describe_numeric(numeric_df, show=False, return_df=True, raw=True)
    assert isinstance(out, pd.DataFrame)
    # Every requested column is present.
    assert set(numeric_df.columns).issubset(out.columns)
    # Core metrics present.
    for metric in ("mean", "std", "median", "q1", "q3", "IQR"):
        assert metric in out.index


def test_raw_vs_formatted(numeric_df):
    formatted = describe_numeric(numeric_df, show=False, return_df=True, raw=False)
    raw = describe_numeric(numeric_df, show=False, return_df=True, raw=True)

    # Formatted output is string-typed.
    assert all(isinstance(v, str) for v in formatted["a"].tolist())
    # Raw output preserves numeric dtype for mean row.
    assert isinstance(raw.loc["mean", "a"], (float, np.floating))


def test_metrics_as_rows_toggle(numeric_df):
    rows = describe_numeric(
        numeric_df, show=False, return_df=True, raw=True, metrics_as_rows=True
    )
    cols = describe_numeric(
        numeric_df, show=False, return_df=True, raw=True, metrics_as_rows=False
    )
    assert rows.shape == (cols.shape[1], cols.shape[0])


def test_missing_columns_raise_keyerror(numeric_df):
    with pytest.raises(KeyError):
        describe_numeric(numeric_df, cols=["a", "does_not_exist"], show=False)


def test_non_dataframe_raises_typeerror():
    with pytest.raises(TypeError):
        describe_numeric([1, 2, 3], show=False)  # type: ignore[arg-type]


def test_no_numeric_columns_raises_valueerror():
    df = pd.DataFrame({"a": ["x", "y", "z"], "b": ["p", "q", "r"]})
    with pytest.raises(ValueError):
        describe_numeric(df, show=False)


def test_negative_decimals_raises():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        describe_numeric(df, decimals=-1, show=False)


def test_iqr_multiplier_affects_outlier_bounds(numeric_df):
    narrow = describe_numeric(
        numeric_df, iqr_multiplier=0.5, show=False, return_df=True, raw=True
    )
    wide = describe_numeric(
        numeric_df, iqr_multiplier=3.0, show=False, return_df=True, raw=True
    )
    # Wider bounds → more lenient → fewer outliers per column.
    for col in numeric_df.columns:
        assert wide.loc["outliers_count", col] <= narrow.loc["outliers_count", col]


def test_safe_against_zero_mean():
    # A series with mean exactly 0 would have raised or produced inf before.
    df = pd.DataFrame({"a": [-1.0, 0.0, 1.0]})
    out = describe_numeric(df, show=False, return_df=True, raw=True)
    cv = out.loc["cv_%", "a"]
    assert pd.isna(cv) or np.isfinite(cv)
