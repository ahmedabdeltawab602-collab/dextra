"""Slice 5 / n-4: dropping unparsable JSON / NDJSON records must emit a named
DextraLoaderWarning instead of vanishing silently, while keeping the existing
plan['problems'] disclosure. Same 'disclosure + warning' shape as m-6.
"""
from __future__ import annotations

import warnings

import dextra as dx
from dextra._loader import DextraLoaderWarning


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _dropped_warnings(records):
    return [str(r.message) for r in records
            if issubclass(r.category, DextraLoaderWarning)
            and "dropped" in str(r.message)]


def test_ndjson_bad_line_warns_and_keeps_disclosure(tmp_path):
    src = _write(tmp_path, "x.ndjson",
                 '{"a":1,"b":2}\n{"a":3,"b":4}\nNOT JSON\n{"a":5,"b":6}\n')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out, plan = dx.load(src, return_params=True, show=False)
    assert out.shape == (3, 2)                       # behaviour unchanged
    msgs = _dropped_warnings(w)
    assert any("1" in m for m in msgs), [(r.category.__name__, str(r.message)) for r in w]
    assert any(p["kind"] == "bad_records" for p in plan["problems"])  # still disclosed


def test_json_array_bad_items_warn(tmp_path):
    src = _write(tmp_path, "x.json", '[{"a":1},{"a":2}, 7, "nope"]')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out, plan = dx.load(src, return_params=True, show=False)
    assert _dropped_warnings(w)
    assert any(p["kind"] == "bad_records" for p in plan["problems"])


def test_clean_ndjson_is_silent(tmp_path):
    src = _write(tmp_path, "ok.ndjson", '{"a":1,"b":2}\n{"a":3,"b":4}\n')
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out, _ = dx.load(src, return_params=True, show=False)
    assert out.shape == (2, 2)
    assert _dropped_warnings(w) == []
