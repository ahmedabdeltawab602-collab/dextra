"""Regression for issue #7.

``dtfeats(features=[...])`` must honour the calendar selection literally.
Passing ``features=['year', 'month']`` (default method='both') previously
leaked the default cyclical sin/cos pairs -- eight new columns instead of
the two requested.
"""
import pandas as pd
import pytest

import dextra as dx


def _df():
    return pd.DataFrame(
        {
            "d": pd.to_datetime(
                ["2020-01-15", "2020-06-20", "2021-03-10",
                 "2021-11-05", "2022-07-30", "2022-12-25"]
            )
        }
    )


def _frame(ret):
    return ret[0] if isinstance(ret, tuple) else ret


def _new_cols(out, df):
    return [c for c in out.columns if c not in df.columns]


def test_issue_07_features_literal_no_cyclical_leak():
    df = _df()
    out = _frame(
        dx.dtfeats(df, cols=["d"], features=["year", "month"],
                   show=False, plot=False, return_df=True)
    )
    new = _new_cols(out, df)
    assert set(new) == {"d_year", "d_month"}, new
    assert not [c for c in new if c.endswith("_sin") or c.endswith("_cos")]


def test_issue_07_default_still_emits_cyclical():
    df = _df()
    out = _frame(dx.dtfeats(df, cols=["d"], show=False, plot=False,
                            return_df=True))
    new = _new_cols(out, df)
    assert any(c.endswith("_sin") for c in new), new


def test_issue_07_features_with_cyclical_method_errors():
    df = _df()
    with pytest.raises(ValueError):
        dx.dtfeats(df, cols=["d"], features=["year", "month"],
                   method="cyclical", show=False, plot=False, return_df=True)
