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


# --------------------------------------------------------------------------- #
# Phase 11.2 - Excel (xlsx/xlsm via openpyxl; values only, never formulas)
# --------------------------------------------------------------------------- #

def _simple_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    import datetime as dt
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "data"
    ws.append(["name", "qty", "price", "when", "ok"])
    ws.append(["a", 1, 1.5, dt.datetime(2024, 1, 2), True])
    ws.append(["b", 2, 2.5, dt.datetime(2024, 2, 3), False])
    ws.append(["c", 3, 3.5, dt.datetime(2024, 3, 4), True])
    p = str(tmp_path / "simple.xlsx")
    wb.save(p)
    return p


def _multi_xlsx(tmp_path):
    """Two sheets; Q1 has a title preamble and an offset data block."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    w1 = wb.active
    w1.title = "Q1"
    w1["C3"] = "Sales report"
    w1["C5"] = "city"
    w1["D5"] = "rev"
    w1["C6"] = "Cairo"
    w1["D6"] = 100
    w1["C7"] = "Giza"
    w1["D7"] = 200
    w2 = wb.create_sheet("Q2")
    w2["A1"] = "x"
    p = str(tmp_path / "multi.xlsx")
    wb.save(p)
    return p


def _merged_xlsx(tmp_path):
    """Two-row merged header + formula cells + an 'NA' token."""
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["region", "2024", None, "total"])
    ws.append([None, "q1", "q2", None])
    ws.merge_cells("B1:C1")
    ws.append(["north", 1, 2, "=B3+C3"])
    ws.append(["south", 3, "NA", "=B4+C4"])
    p = str(tmp_path / "merged.xlsx")
    wb.save(p)
    return p


def _is_texty(dt_):
    return pd.api.types.is_object_dtype(dt_) or pd.api.types.is_string_dtype(dt_)


def test_excel_simple_native_types(tmp_path):
    src = _simple_xlsx(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # clean file -> no warnings
        df, plan = dx.load(src, return_params=True, show=False)
    assert plan["source"]["kind"] == "excel"
    assert plan["metadata"]["n_ambiguous"] == 0
    assert df.shape == (3, 5)
    assert _is_texty(df["name"].dtype)
    assert df["qty"].dtype.kind == "i"
    assert df["price"].dtype.kind == "f"
    assert str(df["when"].dtype).startswith("datetime")
    assert str(df["ok"].dtype) == "boolean"


def test_excel_plan_json_safe_and_replayable(tmp_path):
    src = _simple_xlsx(tmp_path)
    df, plan = dx.load(src, return_params=True, show=False)
    json.dumps(plan)  # JSON-safe
    df2 = dx.load(src, params=plan, show=False)
    pd.testing.assert_frame_equal(df2, df.reset_index(drop=True))


def test_excel_multisheet_default_warns_and_lists_sheets(tmp_path):
    src = _multi_xlsx(tmp_path)
    with pytest.warns(DextraLoaderWarning, match="sheet"):
        df, plan = dx.load(src, return_params=True, show=False)
    assert plan["parse"]["sheet"] == "Q1"
    assert [m["name"] for m in plan["sheets"]] == ["Q1", "Q2"]
    assert list(df.columns) == ["city", "rev"]


def test_excel_sheet_by_name_and_index(tmp_path):
    src = _multi_xlsx(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d_name = dx.load(src, sheet="Q1", show=False)
        d_idx = dx.load(src, sheet=0, show=False)
    pd.testing.assert_frame_equal(d_name, d_idx)


def test_excel_missing_sheet_clear_error(tmp_path):
    src = _multi_xlsx(tmp_path)
    from dextra._loader import DextraLoaderError
    with pytest.raises(DextraLoaderError, match="Q1"):
        dx.load(src, sheet="Q9", show=False)
    with pytest.raises(DextraLoaderError, match="out of range"):
        dx.load(src, sheet=9, show=False)


def test_excel_offset_block_and_preamble(tmp_path):
    src = _multi_xlsx(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, plan = dx.load(src, sheet="Q1", return_params=True, show=False)
    assert list(df.columns) == ["city", "rev"]
    assert df.shape == (2, 2)
    assert df["rev"].dtype.kind in "if"
    assert any(p["kind"] == "offset_block" for p in plan["problems"])
    assert plan["parse"]["block"]["first_row"] == 2   # C3 -> 0-based row 2
    assert plan["parse"]["block"]["first_col"] == 2   # column C


def test_excel_merged_two_row_header_combined(tmp_path):
    src = _merged_xlsx(tmp_path)
    with pytest.warns(DextraLoaderWarning, match="combined"):
        df, plan = dx.load(src, na_values=["NA"], return_params=True,
                           show=False)
    assert list(df.columns) == ["region", "2024_q1", "2024_q2", "total"]
    assert plan["parse"]["header_rows"] == 2
    assert df["2024_q2"].isna().iloc[1]               # NA token honoured


def test_excel_values_not_formulas(tmp_path):
    src = _merged_xlsx(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = dx.load(src, show=False)
    flat = df.astype(str).to_numpy().ravel().tolist()
    assert not any(str(v).startswith("=") for v in flat)


def test_excel_header_rows_forced_confirmed(tmp_path):
    src = _merged_xlsx(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # forcing removes the ambiguity
        df = dx.load(src, header_row=0, header_rows=2, na_values=["NA"],
                     show=False)
    assert list(df.columns) == ["region", "2024_q1", "2024_q2", "total"]


def test_excel_replay_detects_source_change(tmp_path):
    src = _simple_xlsx(tmp_path)
    _, plan = dx.load(src, return_params=True, show=False)
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(src)
    wb["data"].append(["d", 4, 4.5, None, False])
    wb.save(src)
    with pytest.warns(DextraLoaderWarning, match="changed"):
        dx.load(src, params=plan, show=False)


def test_excel_peek_caps_rows(tmp_path):
    src = _multi_xlsx(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plan = dx.peek(src, show=False)
    assert plan["source"]["kind"] == "excel"
    assert plan["metadata"]["n_rows"] <= 10


def test_excel_on_ambiguous_raise_multisheet(tmp_path):
    src = _multi_xlsx(tmp_path)
    with pytest.raises(LoaderAmbiguityError):
        dx.load(src, on_ambiguous="raise", show=False)


def test_excel_legacy_xls_refused(tmp_path):
    from dextra._loader import DextraLoaderError
    p = tmp_path / "old.xls"
    p.write_bytes(b"\xd0\xcf\x11\xe0junk")
    with pytest.raises(DextraLoaderError, match="xlsx"):
        dx.load(str(p), show=False)


def test_excel_empty_sheet_returns_empty_frame(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    p = str(tmp_path / "empty.xlsx")
    wb.save(p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df, plan = dx.load(p, return_params=True, show=False)
    assert df.empty
    assert any(pr["kind"] == "empty" for pr in plan["problems"])
