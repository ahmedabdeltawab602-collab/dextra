"""Slice 5 / n-3: a non-parquet payload behind a .parquet extension must fail
as a DextraLoaderError that points at an explicit kind=, not leak the engine's
raw pyarrow ArrowInvalid. Mirrors the clean wrapping the Excel / JSON paths
already provide for the same wrong-extension mistake.
"""
from __future__ import annotations

import pytest

pytest.importorskip("pyarrow")

import pandas as pd  # noqa: E402

import dextra as dx  # noqa: E402
from dextra._loader import DextraLoaderError  # noqa: E402


def test_csv_content_named_parquet_raises_dextra_error(tmp_path):
    p = tmp_path / "really_csv.parquet"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    with pytest.raises(DextraLoaderError) as ei:
        dx.load(str(p), show=False)
    msg = str(ei.value)
    assert "parquet" in msg.lower()
    assert "kind=" in msg  # actionable: tells the caller how to recover


def test_valid_parquet_still_loads(tmp_path):
    p = tmp_path / "good.parquet"
    pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}).to_parquet(p)
    out = dx.load(str(p), show=False)
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 3
