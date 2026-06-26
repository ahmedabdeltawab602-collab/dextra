"""Slice 5: exercise the two defensive branches added with n-2 / n-3 so the
new code is fully covered -- the parquet DextraLoaderError pass-through and the
duplicate-column guard in clip_outliers' inf scan.
"""
from __future__ import annotations

import pandas as pd
import pytest

import dextra as dx
from dextra import _loader


def test_read_parquet_frame_passes_through_dextra_error(monkeypatch):
    # A DextraLoaderError raised by the reader must propagate unchanged, not be
    # re-wrapped by the broad fallback.
    monkeypatch.setattr(_loader, "_have_parquet_engine", lambda: True)

    def _boom(*a, **k):
        raise _loader.DextraLoaderError("inner dextra error")

    monkeypatch.setattr(pd, "read_parquet", _boom)
    with pytest.raises(_loader.DextraLoaderError, match="inner dextra error"):
        _loader._read_parquet_frame(b"not parquet")


def test_clip_outliers_duplicate_column_skipped_in_inf_scan():
    # Duplicate labels make df[col] a DataFrame; the inf scan must skip it
    # rather than crash there. The original behaviour (erroring later in the
    # main loop) is preserved, so the call still raises.
    dup = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["x", "x"])
    with pytest.raises(Exception):
        dx.clip_outliers(dup, cols=["x"], show=False, plot=False)
