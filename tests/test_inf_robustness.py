"""Slice 5 / n-2: non-finite (inf) inputs must not leak raw numpy
RuntimeWarnings. dextra surfaces a single named UserWarning instead and
suppresses numpy's ``invalid value`` / ``overflow`` noise. The reported
statistics and clipping results are unchanged -- this is a disclosure-only
behaviour, not a computational one.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from dextra import clip_outliers, describe_numeric

INF = pd.DataFrame({"v": [1.0, np.inf, -np.inf, 1e308, 5.0, 2.0]})
CLEAN = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6], "b": [10.0, 9, 8, 7, 6, 5]})


def _record(fn, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = fn(**kw)
    return result, w


def _no_runtimewarning(w):
    return not any(issubclass(rec.category, RuntimeWarning) for rec in w)


def _user_msgs(w):
    return [str(rec.message) for rec in w if issubclass(rec.category, UserWarning)]


def test_describe_numeric_inf_emits_named_warning_not_runtimewarning():
    out, w = _record(describe_numeric, df=INF, return_df=True, raw=True,
                     show=False, plot=False)
    assert _no_runtimewarning(w), [r.category.__name__ for r in w]
    msgs = _user_msgs(w)
    assert any("describe_numeric" in m and "non-finite" in m for m in msgs), msgs
    # Values are unchanged: the inf survives into min/max verbatim.
    assert out.loc["max", "v"] == np.inf
    assert out.loc["min", "v"] == -np.inf


def test_clip_outliers_inf_emits_named_warning_not_runtimewarning():
    out, w = _record(clip_outliers, df=INF, return_df=True, show=False, plot=False)
    assert _no_runtimewarning(w), [r.category.__name__ for r in w]
    msgs = _user_msgs(w)
    assert any("clip_outliers" in m and "non-finite" in m for m in msgs), msgs
    res = out["v"].tolist()
    # Clipping behaviour is byte-for-byte what it was before the fix:
    assert res[1] == np.inf          # +inf left untouched (upper bound is inf)
    assert res[3] == 1e308           # within bounds -> untouched
    assert res[0] == 1.0 and res[4] == 5.0 and res[5] == 2.0  # inliers intact
    assert np.isfinite(res[2]) and res[2] < 0  # -inf clipped to finite lower bound


def test_clean_data_emits_no_warning():
    _, w1 = _record(describe_numeric, df=CLEAN, return_df=True, show=False, plot=False)
    _, w2 = _record(clip_outliers, df=CLEAN, return_df=True, show=False, plot=False)
    assert list(w1) == [], [r.category.__name__ for r in w1]
    assert list(w2) == [], [r.category.__name__ for r in w2]
