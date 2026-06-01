"""Hardening sprint -- polars / pyarrow input acceptance via _ensure_pandas.

dextra's data-processing and modeling entry points accept any table exposing
``.to_pandas()`` (polars DataFrame, pyarrow Table) and convert it to pandas at
the boundary. pandas frames pass through untouched. No real polars/pyarrow
dependency is needed for these tests -- a stand-in object exercises the path.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import dextra as dx  # noqa: E402
from dextra._utils import _ensure_pandas  # noqa: E402


class _TableLike:
    """Minimal stand-in for a polars DataFrame / pyarrow Table."""

    def __init__(self, pdf):
        self._pdf = pdf

    def to_pandas(self):
        return self._pdf


def test_pandas_passes_through_unchanged():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert _ensure_pandas(df) is df               # identity: zero-copy


def test_table_like_is_converted():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = _ensure_pandas(_TableLike(df))
    assert isinstance(out, pd.DataFrame) and out.equals(df)


def test_unsupported_type_raises():
    with pytest.raises(TypeError):
        _ensure_pandas([1, 2, 3])


def test_entry_point_accepts_table_like():
    df = pd.DataFrame({"x": np.arange(1.0, 21.0), "y": np.arange(20.0)})
    out = dx.transform(_TableLike(df), cols=["x"], method="log1p",
                       return_df=True, show=False, plot=False)
    assert isinstance(out, pd.DataFrame) and "x_log1p" in out.columns
