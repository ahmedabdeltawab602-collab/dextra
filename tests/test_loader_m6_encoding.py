"""M-6 regression -- encoding detection must not corrupt legacy Arabic.

EVALUATION_REPORT.md M-6 (Major): charset-normalizer's first-try guess used
to be stamped CONFIRMED, so cp1256 Arabic came back as cp932/mac_latin2
mojibake and even valid utf-8 Arabic came back as gb18030 -- silently.
Detection now puts strict utf-8 first, prefers cp1256 for Arabic-looking
byte runs, reports every non-utf-8 guess as ambiguous (so on_ambiguous
warn/raise/plan surface it) and documents encoding= as the deterministic
way out. Arabic strings are built from codepoints to keep this file ASCII.
"""
from __future__ import annotations

import warnings

import pytest

import dextra as dx
from dextra._loader import DextraLoaderWarning, LoaderAmbiguityError

CAIRO = "".join(map(chr, [0x627, 0x644, 0x642, 0x627, 0x647, 0x631, 0x629]))
JEDDAH = "".join(map(chr, [0x62C, 0x62F, 0x629]))


def _body(n=12):
    cities = [CAIRO, JEDDAH]
    rows = ["name,city"] + ["user%d,%s" % (i, cities[i % 2]) for i in range(n)]
    return "\n".join(rows) + "\n"


def _write_bytes(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


# --------------------------------------------------------------------------- #
# Legacy Arabic: correct guess, honestly flagged
# --------------------------------------------------------------------------- #

def test_cp1256_auto_guesses_cp1256_ambiguous_and_round_trips(tmp_path):
    src = _write_bytes(tmp_path, "ar1256.csv", _body().encode("cp1256"))
    with pytest.warns(DextraLoaderWarning):
        df, plan = dx.load(src, return_params=True, show=False)
    enc = plan["decisions"]["encoding"]
    assert enc["value"] == "cp1256"
    assert enc["confidence"] == "ambiguous"  # a guess is never confirmed
    assert df["city"].iloc[0] == CAIRO       # text survived intact


def test_cp1256_auto_raises_under_raise_policy(tmp_path):
    src = _write_bytes(tmp_path, "ar1256r.csv", _body().encode("cp1256"))
    with pytest.raises(LoaderAmbiguityError):
        dx.load(src, on_ambiguous="raise", show=False)


def test_explicit_encoding_is_confirmed_and_silent(tmp_path):
    src = _write_bytes(tmp_path, "ar1256e.csv", _body().encode("cp1256"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the escape hatch must be silent
        df, plan = dx.load(src, encoding="cp1256", return_params=True,
                           show=False)
    assert plan["decisions"]["encoding"]["confidence"] == "confirmed"
    assert df["city"].iloc[0] == CAIRO


# --------------------------------------------------------------------------- #
# utf-8 must always beat the detectors
# --------------------------------------------------------------------------- #

def test_utf8_arabic_is_confirmed_and_silent(tmp_path):
    src = _write_bytes(tmp_path, "ar8.csv", _body().encode("utf-8"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # recommended encoding: no warning
        df, plan = dx.load(src, return_params=True, show=False)
    enc = plan["decisions"]["encoding"]
    assert enc["value"] == "utf-8"
    assert enc["confidence"] == "confirmed"
    assert df["city"].iloc[0] == CAIRO


def test_big_utf8_survives_sample_boundary_split(tmp_path):
    # file > sample_bytes (262144), normal-size rows; place the first byte
    # of an Arabic run exactly at offset 262143 so the sample cut at 262144
    # splits the 2-byte char (the old strict-decode would fail and fall to
    # a guess; the boundary tolerance must keep utf-8 confirmed)
    header, rowlen = "name,city\n", 64
    filler = "x," + "a" * (rowlen - 3) + "\n"          # exactly rowlen bytes
    n = (262143 - len(header) - 2) // rowlen
    m = 262143 - (len(header) + n * rowlen) - 2        # ASCII pad in last row
    body = header + filler * n + "y," + "a" * m + (CAIRO * 4) + "\n"
    data = body.encode("utf-8")
    assert data[262143] >= 0x80      # multi-byte char straddles the cut
    assert len(data) > 262144        # the sample really is truncated
    src = _write_bytes(tmp_path, "big8.csv", data)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df, plan = dx.load(src, return_params=True, show=False)
    enc = plan["decisions"]["encoding"]
    assert enc["value"] == "utf-8"
    assert enc["confidence"] == "confirmed"
    assert df["city"].iloc[-1].endswith(CAIRO)


# --------------------------------------------------------------------------- #
# European legacy text is not hijacked by the Arabic-first guess
# --------------------------------------------------------------------------- #

def test_european_latin1_not_claimed_as_arabic(tmp_path):
    koln = "K" + chr(0xF6) + "ln"
    creteil = "Cr" + chr(0xE9) + "teil"
    body = "name,town\nw1,%s\nw2,%s\n" % (koln, creteil)
    src = _write_bytes(tmp_path, "eur.csv", body.encode("latin-1"))
    with pytest.warns(DextraLoaderWarning):
        df, plan = dx.load(src, return_params=True, show=False)
    enc = plan["decisions"]["encoding"]
    assert enc["confidence"] == "ambiguous"  # still a guess, still flagged
    assert "legacy Arabic byte runs" not in enc["reason"]
