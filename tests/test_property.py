"""Phase 6.5 -- property-based tests (Hypothesis).

Invariants that must hold for arbitrary finite numeric input, complementing the
example-based suites. Skipped if Hypothesis is absent.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)
hyp = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import dextra as dx  # noqa: E402

KW = dict(show=False, plot=False)

_floats = st.lists(
    st.floats(allow_nan=False, allow_infinity=False,
              min_value=-1e6, max_value=1e6),
    min_size=12, max_size=200)


@settings(max_examples=40, deadline=None)
@given(_floats)
def test_describe_count_matches_n(values):
    df = pd.DataFrame({"x": values})
    out = dx.describe_numeric(df, return_df=True, show=False)
    # the count metric must equal the number of (finite) observations
    assert int(float(out.loc["count", "x"])) == len(values)


@settings(max_examples=40, deadline=None)
@given(_floats)
def test_zscore_extreme_count_bounded(values):
    df = pd.DataFrame({"x": values})
    res = dx.z_scores(df, cols=["x"], return_df=True, **KW)
    # count of points beyond 2 sigma can never exceed the sample size
    n_above = int(float(res.loc["x", "n_above_2sigma"]))
    assert 0 <= n_above <= len(values)


@settings(max_examples=40, deadline=None)
@given(_floats)
def test_standard_scale_centers_data(values):
    arr = np.asarray(values, dtype=float)
    assume(np.std(arr) > 1e-6)                 # scaling needs non-zero spread
    df = pd.DataFrame({"x": values})
    out = dx.scale(df, cols=["x"], method="standard", return_df=True, **KW)
    scaled = pd.to_numeric(out.iloc[:, -1], errors="coerce").to_numpy()
    assert abs(float(np.nanmean(scaled))) < 1e-6
