"""Regression for issue #6.

``handle_missing`` must not emit numpy RuntimeWarnings ("Mean of empty
slice" / "All-NaN slice encountered") when a frame contains all-NaN
columns: per-column statistics are computed on non-empty data only.
"""
import warnings

import numpy as np
import pandas as pd

import dextra as dx


def _all_nan_frame():
    return pd.DataFrame(
        {
            "num_allnan_a": pd.Series([np.nan] * 6, dtype="float64"),
            "num_allnan_b": pd.Series([np.nan] * 6, dtype="float64"),
            "obj_allnan": pd.Series([None] * 6, dtype="object"),
            "dt_allnan": pd.Series([pd.NaT] * 6),
            "num_partial": pd.Series([1.0, 2.0, np.nan, 4.0, np.nan, 6.0]),
            "obj_partial": pd.Series(["x", None, "y", None, "x", None]),
        }
    )


def _runtime_warnings(recorded):
    return [w for w in recorded if issubclass(w.category, RuntimeWarning)]


def test_issue_06_no_runtime_warning_auto_on_all_nan():
    df = _all_nan_frame()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = dx.handle_missing(
            df, strategy="auto", show=False, plot=False, return_df=True
        )
    msgs = [str(w.message) for w in _runtime_warnings(rec)]
    assert not msgs, msgs
    assert out["num_partial"].isna().sum() == 0
    assert out["num_allnan_a"].isna().all()


def test_issue_06_no_runtime_warning_explicit_median_all_nan():
    df = _all_nan_frame()[["num_allnan_a", "num_partial"]]
    for strat in ("mean", "median"):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            out = dx.handle_missing(
                df, strategy=strat, show=False, plot=False, return_df=True
            )
        msgs = [str(w.message) for w in _runtime_warnings(rec)]
        assert not msgs, (strat, msgs)
        assert out["num_partial"].isna().sum() == 0
        assert out["num_allnan_a"].isna().all()
