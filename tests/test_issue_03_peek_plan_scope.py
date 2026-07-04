"""Regression for issue #3 (delegated decision 4).

A plan returned by ``peek`` records ``plan_scope='preview'``. Replaying it
with ``load(params=plan)`` loads ALL rows (the preview cap is an artifact
of peeking, not part of the recipe) and emits ONE loud warning that names
the origin and recommends ``load(source, return_params=True)``. Saved
plans that predate the field behave the same way. A cap the user set
deliberately at load time (``plan_scope='load'``) is still honoured.
"""
import warnings

import pandas as pd
import pytest

import dextra as dx
from dextra._loader import DextraLoaderWarning


def _csv(tmp_path, n=50):
    p = tmp_path / "data.csv"
    lines = ["id,amount"] + [f"{i},{i * 1.5}" for i in range(n)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def _loader_warnings(rec):
    return [w for w in rec if issubclass(w.category, DextraLoaderWarning)
            and "max_rows" in str(w.message)]


def test_issue_03_peek_plan_records_preview_scope(tmp_path):
    plan = dx.peek(_csv(tmp_path), show=False)
    assert plan["plan_scope"] == "preview"


def test_issue_03_peek_plan_replays_all_rows_one_loud_warning(tmp_path):
    src = _csv(tmp_path)
    plan = dx.peek(src, show=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = dx.load(src, params=plan, show=False)
    assert len(out) == 50, len(out)
    lw = _loader_warnings(rec)
    assert len(lw) == 1, [str(w.message) for w in rec]
    msg = str(lw[0].message)
    assert "peek" in msg
    assert "return_params=True" in msg


def test_issue_03_legacy_plan_without_field_same_behaviour(tmp_path):
    src = _csv(tmp_path)
    plan = dx.peek(src, show=False)
    del plan["plan_scope"]  # simulate an artifact saved before 0.6.0
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = dx.load(src, params=plan, show=False)
    assert len(out) == 50
    assert len(_loader_warnings(rec)) == 1


def test_issue_03_deliberate_load_cap_honoured_silently(tmp_path):
    src = _csv(tmp_path)
    _df, plan = dx.load(src, max_rows=7, return_params=True, show=False)
    assert plan["plan_scope"] == "load"
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = dx.load(src, params=plan, show=False)
    assert len(out) == 7
    assert not _loader_warnings(rec)


def test_issue_03_json_peek_replay_loads_all(tmp_path):
    p = tmp_path / "data.json"
    rows = ",".join(f'{{"id": {i}, "v": {i * 2}}}' for i in range(20))
    p.write_text(f"[{rows}]", encoding="utf-8")
    plan = dx.peek(str(p), show=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = dx.load(str(p), params=plan, show=False)
    assert len(out) == 20
    assert len(_loader_warnings(rec)) == 1


def test_issue_03_excel_peek_replay_loads_all(tmp_path):
    pytest.importorskip("openpyxl")
    p = tmp_path / "data.xlsx"
    pd.DataFrame({"id": range(30), "v": range(30)}).to_excel(
        p, index=False, engine="openpyxl")
    plan = dx.peek(str(p), show=False)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = dx.load(str(p), params=plan, show=False)
    assert len(out) == 30
    assert len(_loader_warnings(rec)) == 1
