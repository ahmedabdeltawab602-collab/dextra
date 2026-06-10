"""dextra Phase 9 -- report (edareport).

Covers the one-call self-contained HTML report: a portable file with embedded
base64 figures and inline tables, the JSON-safe build manifest, section
isolation (a section that cannot run is skipped, the report still renders),
immutability of the input DataFrame, the sections subset, and the optional
target-aware model section (skipped when scikit-learn is absent).
"""
from __future__ import annotations

import json
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import dextra as dx


@pytest.fixture
def mixed_df():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(5000, 1500, n),
        "score": rng.normal(0, 1, n),
        "city": rng.choice(["Cairo", "Giza", "Alex"], n),
        "churn": rng.choice(["yes", "no"], n),
    })
    df.loc[:5, "income"] = np.nan
    return df


def _json_ok(obj):
    json.loads(json.dumps(obj))


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_exports_alias_and_no_underscore():
    assert callable(dx.edareport)
    assert dx.edarep is dx.edareport
    assert not hasattr(dx, "eda_report")     # underscore-free public name


# ---------------------------------------------------------------------------
# self-contained HTML
# ---------------------------------------------------------------------------

def test_writes_self_contained_html(mixed_df, tmp_path):
    out = str(tmp_path / "r.html")
    path = dx.edareport(mixed_df, out=out, show=False)
    assert path == out
    assert os.path.exists(out)
    html = open(out, encoding="utf-8").read()
    assert html.startswith("<!DOCTYPE html>")
    assert html.rstrip().endswith("</html>")
    assert "data:image/png;base64," in html        # embedded figures
    # no sidecar assets written next to the report
    siblings = [p for p in os.listdir(tmp_path) if p != "r.html"]
    assert siblings == []


def test_default_return_is_path(mixed_df, tmp_path):
    out = str(tmp_path / "r.html")
    assert dx.edareport(mixed_df, out=out, show=False) == out


# ---------------------------------------------------------------------------
# manifest
# ---------------------------------------------------------------------------

def test_manifest_is_json_safe_and_built(mixed_df, tmp_path):
    out = str(tmp_path / "r.html")
    man = dx.edareport(mixed_df, out=out, return_params=True, show=False)
    assert man["function"] == "edareport"
    _json_ok(man)
    for key in ("overview", "quality", "univariate", "bivariate"):
        assert man["sections"][key]["status"] == "built", key
    assert man["metadata"]["n_rows"] == len(mixed_df)
    assert man["metadata"]["n_cols"] == mixed_df.shape[1]


# ---------------------------------------------------------------------------
# immutability / isolation / subset
# ---------------------------------------------------------------------------

def test_does_not_mutate_input(mixed_df, tmp_path):
    before = mixed_df.copy(deep=True)
    dx.edareport(mixed_df, out=str(tmp_path / "r.html"), show=False)
    pd.testing.assert_frame_equal(mixed_df, before)
    assert "dextra_audit" not in mixed_df.attrs


def test_section_isolation_skips_but_still_writes(tmp_path):
    # only one numeric column -> bivariate (correlation) cannot run
    df = pd.DataFrame({"x": np.arange(12.0), "g": list("aabbccddeeff")})
    out = str(tmp_path / "r.html")
    man = dx.edareport(df, out=out, return_params=True, show=False)
    assert man["sections"]["bivariate"]["status"] == "skipped"
    assert "reason" in man["sections"]["bivariate"]
    assert man["sections"]["overview"]["status"] == "built"
    assert os.path.exists(out)


def test_sections_subset(mixed_df, tmp_path):
    man = dx.edareport(mixed_df, out=str(tmp_path / "r.html"),
                       sections=["overview", "quality"],
                       return_params=True, show=False)
    assert set(man["sections"]) == {"overview", "quality"}


def test_model_omitted_without_flag(mixed_df, tmp_path):
    man = dx.edareport(mixed_df, out=str(tmp_path / "r.html"),
                       return_params=True, show=False)
    assert "model" not in man["sections"]


# ---------------------------------------------------------------------------
# optional model section (needs scikit-learn)
# ---------------------------------------------------------------------------

def test_model_section_classification(mixed_df, tmp_path):
    pytest.importorskip("sklearn")
    man = dx.edareport(mixed_df, out=str(tmp_path / "r.html"), target="churn",
                       include_model=True, return_params=True, show=False)
    assert man["sections"]["model"]["status"] == "built"


def test_model_section_regression(tmp_path):
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "a": rng.normal(0, 1, n),
        "b": rng.normal(0, 1, n),
        "price": rng.normal(100, 20, n),
    })
    man = dx.edareport(df, out=str(tmp_path / "r.html"), target="price",
                       include_model=True, return_params=True, show=False)
    assert man["sections"]["model"]["status"] == "built"


def test_idempotent_report(mixed_df, tmp_path):
    a = dx.edareport(mixed_df, out=str(tmp_path / "a.html"),
                     return_params=True, show=False)
    b = dx.edareport(mixed_df, out=str(tmp_path / "b.html"),
                     return_params=True, show=False)
    assert (set(a["sections"]) == set(b["sections"]))
    for k in a["sections"]:
        assert a["sections"][k]["status"] == b["sections"][k]["status"]
# ---------------------------------------------------------------------------
# audit trail in the manifest (audit #6)
# ---------------------------------------------------------------------------

def test_manifest_includes_audit_trail(mixed_df, tmp_path):
    man = dx.edareport(mixed_df, out=str(tmp_path / "r.html"),
                       return_params=True, show=False)
    trail = man["dextra_audit"]
    assert isinstance(trail, list) and len(trail) == 1
    entry = trail[-1]
    assert entry["stage"] == "report" and entry["function"] == "edareport"
    assert entry["params"]["sections_built"]
    assert entry["decision"]
    _json_ok(man)                                    # stays JSON-safe
    assert "dextra_audit" not in mixed_df.attrs      # input never mutated


def test_manifest_audit_keeps_input_history(mixed_df, tmp_path):
    df2 = mixed_df.copy()
    df2.attrs = {"dextra_audit": [{"stage": "cleaning", "function": "fillna"}]}
    man = dx.edareport(df2, out=str(tmp_path / "r.html"),
                       return_params=True, show=False)
    trail = man["dextra_audit"]
    assert len(trail) == 2 and trail[0]["stage"] == "cleaning"
    assert trail[-1]["function"] == "edareport"
    assert len(df2.attrs["dextra_audit"]) == 1       # input list untouched
