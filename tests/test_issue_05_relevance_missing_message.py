"""Regression for issue #5 (delegated decision 5).

On a wide-missing frame, ``relevance`` (and the selectors sharing its
feature-matrix cleaner) must fail with a message that names the literal
remedy -- impute first via ``handle_missing`` -- instead of a bare
"not enough complete rows".
"""
import numpy as np
import pandas as pd
import pytest

import dextra as dx


def _wide_missing():
    # every row has at least one NaN across the candidates -> 0 complete rows
    n = 24
    df = pd.DataFrame(
        {
            "a": [np.nan if i % 2 == 0 else float(i) for i in range(n)],
            "b": [np.nan if i % 2 == 1 else float(i) for i in range(n)],
            "target": [i % 2 for i in range(n)],
        }
    )
    return df


def test_issue_05_relevance_names_the_remedy():
    df = _wide_missing()
    with pytest.raises(ValueError) as ei:
        dx.relevance(df, y="target", keep=1, show=False, plot=False)
    msg = str(ei.value)
    assert "handle_missing" in msg, msg
    # the count diagnosis is part of the guidance
    assert "complete" in msg


def test_issue_05_message_counts_are_correct():
    df = _wide_missing()
    with pytest.raises(ValueError) as ei:
        dx.relevance(df, y="target", keep=1, show=False, plot=False)
    msg = str(ei.value)
    # 0 complete rows out of 24
    assert "0" in msg and "24" in msg, msg


def test_issue_05_remedy_is_literally_executable():
    df = _wide_missing()
    fixed = dx.handle_missing(df, show=False, plot=False, return_df=True)
    out = dx.relevance(fixed, y="target", keep=1, show=False, plot=False,
                       return_df=True)
    if isinstance(out, tuple):
        out = out[0]
    assert isinstance(out, pd.DataFrame)
