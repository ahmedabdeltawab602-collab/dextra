"""Phase 11.1 - smart loader (csv/tsv) tests.

Core cases use the pure-stdlib detection fallbacks, so they run on the base
install (no `io` extra). The high-quality detectors are used when present but
never required.
"""
from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

import dextra as dx
from dextra._loader import (
    DextraLoaderWarning,
    LoaderAmbiguityError,
    LoaderReplayError,
    LoaderSecurityError,
)


def _write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_text(text, encoding=encoding)
    return str(p)


# --------------------------------------------------------------------------- #
# Happy / detection
# --------------------------------------------------------------------------- #

def test_clean_csv_all_confirmed(tmp_path):
    src = _write(tmp_path, "clean.csv", "a,b,c\n1,2.5,x\n3,4.0,y\n5,6.5,z\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warnings expected on a clean file
        df = dx.load(src, show=False)
    assert list(df.columns) == ["a", "b", "c"]
    assert df["a"].dtype.kind in "if"
    assert df["b"].dtype.kind == "f"
    assert df.shape == (3, 3)


def test_semicolon_delimiter(tmp_path):
    src = _write(tmp_path, "semi.csv", "a;b;c\n1;2;3\n4;5;6\n7;8;9\n")
    df = dx.load(src, show=False)
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (3, 3)


def test_tsv_kind(tmp_path):
    src = _write(tmp_path, "data.tsv", "a\tb\n1\t2\n3\t4\n")
    df = dx.load(src, show=False)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_cp1256_arabic_headers(tmp_path):
    src = _write(tmp_path, "ar.csv", "الاسم,القيمة\nأحمد,10\nسارة,20\n",
                 encoding="cp1256")
    df, plan = dx.load(src, return_params=True, show=False)
    assert df.shape == (2, 2)
    assert plan["decisions"]["encoding"]["value"].lower() in ("cp1256", "utf-8", "latin-1")


def test_junk_preamble_header_detection(tmp_path):
    text = "Sales Report\nGenerated 2026\n\nname,amount\nA,10\nB,20\nC,30\n"
    src = _write(tmp_path, "junk.csv", text)
    df, plan = dx.load(src, return_params=True, show=False)
    assert list(df.columns) == ["name", "amount"]
    assert df.shape == (3, 2)
    assert plan["decisions"]["header"]["value"] >= 1


def test_locale_numerics(tmp_path):
    src = _write(tmp_path, "loc.csv", "v\n1.234,56\n2.000,00\n3.500,25\n")
    df = dx.load(src, decimal=",", thousands=".", show=False)
    assert df["v"].dtype.kind == "f"
    assert abs(df["v"].iloc[0] - 1234.56) < 1e-6


def test_currency_percent_semicolon(tmp_path):
    # thousands=',' implies a non-comma delimiter; use ';'
    src = _write(tmp_path, "cur.csv",
                 "price;rate\n$1,200;45%\n$2,300;50%\n$900;5%\n")
    df = dx.load(src, thousands=",", show=False)
    assert df["price"].dtype.kind in "if"
    assert df["rate"].dtype.kind in "if"
    assert list(df["price"]) == [1200, 2300, 900]


def test_iso_dates(tmp_path):
    src = _write(tmp_path, "d.csv", "day\n2026-01-01\n2026-02-15\n2026-03-30\n")
    df = dx.load(src, show=False)
    assert str(df["day"].dtype).startswith("datetime64")


# --------------------------------------------------------------------------- #
# Ambiguity / policy
# --------------------------------------------------------------------------- #

def _ambiguous_src(tmp_path):
    # 19/20 numeric -> accepted (>=0.95) but not 1.0 -> ambiguous column.
    body = "\n".join(str(i) for i in range(19)) + "\nbad"
    return _write(tmp_path, "amb.csv", "x\n" + body + "\n")


def test_plan_policy_returns_plan_without_loading(tmp_path):
    plan = dx.load(_ambiguous_src(tmp_path), on_ambiguous="plan", show=False)
    assert isinstance(plan, dict) and plan["function"] == "load"
    assert plan["metadata"]["n_ambiguous"] >= 1


def test_warn_policy_emits_warning(tmp_path):
    with pytest.warns(DextraLoaderWarning):
        dx.load(_ambiguous_src(tmp_path), on_ambiguous="warn", show=False)


def test_raise_policy_raises(tmp_path):
    with pytest.raises(LoaderAmbiguityError):
        dx.load(_ambiguous_src(tmp_path), on_ambiguous="raise", show=False)


def test_clean_file_never_raises_under_raise_policy(tmp_path):
    src = _write(tmp_path, "ok.csv", "a,b\n1,2\n3,4\n")
    df = dx.load(src, on_ambiguous="raise", show=False)
    assert df.shape == (2, 2)


def test_high_risk_confidence_for_id_like(tmp_path):
    body = "\n".join(f"{i},v{i}" for i in range(19)) + "\nbad,v"
    src = _write(tmp_path, "id.csv", "id,name\n" + body + "\n")
    _, plan = dx.load(src, return_params=True, show=False)
    assert plan["columns"]["id"]["confidence"] in ("ambiguous-high-risk", "confirmed")


# --------------------------------------------------------------------------- #
# Contract / replay / security
# --------------------------------------------------------------------------- #

def test_return_params_is_json_serialisable(tmp_path):
    src = _write(tmp_path, "j.csv", "a,b\n1,2\n3,4\n")
    df, plan = dx.load(src, return_params=True, show=False)
    json.dumps(plan)  # must not raise
    assert plan["metadata"]["n_cols"] == 2


def test_replay_reproduces_frame(tmp_path):
    src = _write(tmp_path, "r.csv", "a,b\n1,2.5\n3,4.5\n5,6.5\n")
    df1, plan = dx.load(src, return_params=True, show=False)
    df2 = dx.load(src, params=plan, show=False)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                  df2.reset_index(drop=True))


def test_replay_detects_source_change(tmp_path):
    src = _write(tmp_path, "c.csv", "a,b\n1,2\n3,4\n")
    _, plan = dx.load(src, return_params=True, show=False)
    _write(tmp_path, "c.csv", "a,b\n9,9\n8,8\n")  # mutate the source
    with pytest.raises(LoaderReplayError):
        dx.load(src, params=plan, on_ambiguous="raise", show=False)


def test_pickle_refused(tmp_path):
    p = tmp_path / "x.pkl"
    p.write_bytes(b"\x80\x04K\x01.")
    with pytest.raises(LoaderSecurityError):
        dx.load(str(p), show=False)


def test_dataframe_passthrough_immutability(tmp_path):
    raw = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
    raw.attrs["foo"] = "bar"
    before_dtype = raw["a"].dtype          # version-agnostic (object or StringDtype)
    out = dx.load(raw, show=False)
    assert out["a"].dtype.kind in "if"     # re-typed to numeric
    assert raw["a"].dtype == before_dtype  # original dtype untouched
    assert list(raw["a"]) == ["1", "2", "3"]  # original values untouched
    assert any(e["function"] == "load" for e in out.attrs.get("dextra_audit", []))


def test_alias_identity():
    assert dx.dload is dx.load
    assert dx.dpeek is dx.peek


def test_peek_returns_plan_without_full_load(tmp_path):
    src = _write(tmp_path, "p.csv", "a,b\n" + "\n".join("1,2" for _ in range(100)))
    plan = dx.peek(src, show=False, n_preview=5)
    assert isinstance(plan, dict)
    assert plan["metadata"]["n_rows"] <= 5


def test_show_false_is_silent(capsys, tmp_path):
    src = _write(tmp_path, "s.csv", "a,b\n1,2\n3,4\n")
    dx.load(src, show=False, interactive=True)  # interactive ignored when show=False
    assert capsys.readouterr().out == ""


def test_audit_entry_present(tmp_path):
    src = _write(tmp_path, "a.csv", "a,b\n1,2\n3,4\n")
    df = dx.load(src, show=False)
    audit = df.attrs.get("dextra_audit", [])
    assert audit and audit[-1]["stage"] == "loader"
