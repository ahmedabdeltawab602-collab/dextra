"""B-1 regression -- leading-zero identifiers must never be corrupted.

EVALUATION_REPORT.md B-1 (Blocker): measured inference used to coerce
pure-digit identifier columns into datetime ("001" -> year 1) or int64
("01001" -> 1001, zeros lost) at parse_rate=1.0 and mark the decision
CONFIRMED -- so no on_ambiguous policy could ever catch the corruption.
The fix keeps such columns as text, flagged ambiguous / ambiguous-high-risk,
without over-reaching into lossless conversions.
"""
from __future__ import annotations

import warnings

import pytest

import dextra as dx
from dextra._loader import DextraLoaderWarning, LoaderAmbiguityError


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _csv(col, values):
    return (f"{col},amount\n"
            + "".join(f"{v},{10 + i}\n" for i, v in enumerate(values)))


# --------------------------------------------------------------------------- #
# Corruption guard: leading-zero columns stay text and are flagged
# --------------------------------------------------------------------------- #

def test_leading_zero_id_stays_text_and_high_risk(tmp_path):
    src = _write(tmp_path, "id.csv", _csv("id", ["001", "002", "003"]))
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        df, plan = dx.load(src, return_params=True, show=False)
    assert df["id"].dtype == object
    assert list(df["id"]) == ["001", "002", "003"]  # zeros preserved
    cp = plan["columns"]["id"]
    assert cp["confidence"] == "ambiguous-high-risk"
    assert cp["reason"] == "leading-zero identifier kept as text"
    assert any(isinstance(w.message, DextraLoaderWarning) for w in wlist)


def test_leading_zero_zipcode_stays_text_and_ambiguous(tmp_path):
    # name does not match the high-risk regex -> plain ambiguous, still caught
    src = _write(tmp_path, "zip.csv",
                 _csv("zipcode", ["01001", "02134", "10001"]))
    with pytest.warns(DextraLoaderWarning):
        df, plan = dx.load(src, return_params=True, show=False)
    assert df["zipcode"].dtype == object
    assert list(df["zipcode"]) == ["01001", "02134", "10001"]
    assert plan["columns"]["zipcode"]["confidence"] == "ambiguous"


def test_leading_zero_raises_under_raise_policy(tmp_path):
    src = _write(tmp_path, "ord.csv",
                 _csv("order_id", ["0001", "0002", "0010"]))
    with pytest.raises(LoaderAmbiguityError):
        dx.load(src, on_ambiguous="raise", show=False)


def test_leading_zero_replay_is_deterministic(tmp_path):
    src = _write(tmp_path, "rep.csv",
                 _csv("sku", ["0001234", "0005678", "0009012"]))
    with pytest.warns(DextraLoaderWarning):
        df1, plan = dx.load(src, return_params=True, show=False)
        df2 = dx.load(src, params=plan, show=False)
    assert df1.equals(df2)
    assert list(df2["sku"]) == ["0001234", "0005678", "0009012"]


# --------------------------------------------------------------------------- #
# Anti-overreach controls: lossless conversions keep working, CONFIRMED
# --------------------------------------------------------------------------- #

def test_pure_digits_without_leading_zero_still_int(tmp_path):
    src = _write(tmp_path, "n.csv", _csv("qty", ["100", "200", "300"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # clean load: no warning at all
        df, plan = dx.load(src, return_params=True, show=False)
    assert df["qty"].dtype.kind in "iu"
    assert list(df["qty"]) == [100, 200, 300]
    assert plan["columns"]["qty"]["confidence"] == "confirmed"


def test_real_dates_still_datetime(tmp_path):
    src = _write(tmp_path, "d.csv",
                 _csv("when", ["2024-01-02", "2024-01-03", "2024-01-04"]))
    df = dx.load(src, show=False)
    assert str(df["when"].dtype).startswith("datetime64")


def test_single_zero_is_a_number_not_an_identifier(tmp_path):
    src = _write(tmp_path, "z.csv", _csv("flag", ["0", "1", "0"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = dx.load(src, show=False)
    assert df["flag"].dtype.kind in "iub"  # numeric/boolean, never object


def test_dirty_id_control_unchanged(tmp_path):
    # one unparseable value -> conversion already refused before the fix;
    # the column stays text exactly as before (confirmed, no new flag)
    src = _write(tmp_path, "dirty.csv",
                 _csv("id", ["001", "002", "003", "004", "NOPE"]))
    df, plan = dx.load(src, return_params=True, show=False)
    assert df["id"].dtype == object
    assert plan["columns"]["id"]["confidence"] == "confirmed"
