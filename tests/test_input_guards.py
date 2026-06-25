"""m-4 / m-5 input guards.

Frame-first functions reject malformed inputs with a clear dextra-style
message (pointing at ``tidycols``) instead of a raw AttributeError:

* duplicate column labels  -> ValueError suggesting tidycols (m-4)
* empty frame (no rows/cols) -> ValueError in clean_report (m-5)
"""
import pandas as pd
import pytest

import dextra as dx


def test_describe_numeric_rejects_duplicate_columns():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="duplicate column"):
        dx.describe_numeric(df, show=False)


def test_clean_report_rejects_duplicate_columns():
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "a", "b"])
    with pytest.raises(ValueError, match="tidycols"):
        dx.clean_report(df, show=False, plot=False)


def test_clean_report_rejects_empty_frame():
    with pytest.raises(ValueError, match="nothing to audit"):
        dx.clean_report(pd.DataFrame(), show=False, plot=False)


def test_input_guards_do_not_fire_on_valid_frame():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10, 20, 30, 40]})
    dx.describe_numeric(df, show=False)          # must not raise
    dx.clean_report(df, show=False, plot=False)  # must not raise
