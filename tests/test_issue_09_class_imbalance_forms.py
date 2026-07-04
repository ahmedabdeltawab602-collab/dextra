"""Regression for issue #9.

``class_imbalance`` should accept both the Series form and the
``(df, col)`` form (mirroring ``frequency_table``), instead of silently
mapping the column name onto ``decimals`` and raising.
"""
import pandas as pd
import pytest

import dextra as dx


def _df():
    return pd.DataFrame(
        {
            "churn": ["yes", "no", "no", "no", "yes", "no", "no", "no"],
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )


def _frame(ret):
    return ret[0] if isinstance(ret, tuple) else ret


def test_issue_09_series_form():
    df = _df()
    r = _frame(dx.class_imbalance(df["churn"], show=False, plot=False,
                                  return_df=True))
    assert isinstance(r, pd.DataFrame)
    assert set(r.index) == {"yes", "no"}


def test_issue_09_df_col_form_matches_series():
    df = _df()
    a = _frame(dx.class_imbalance(df["churn"], show=False, plot=False,
                                  return_df=True))
    b = _frame(dx.class_imbalance(df, "churn", show=False, plot=False,
                                  return_df=True))
    pd.testing.assert_frame_equal(a, b)


def test_issue_09_df_without_col_errors():
    df = _df()
    with pytest.raises((ValueError, KeyError)):
        dx.class_imbalance(df, show=False, plot=False, return_df=True)
