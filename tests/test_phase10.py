"""dextra Phase 10 -- dashboard (dash).

Covers the generated Streamlit app: a runnable ``dashboard_app.py`` plus a
dtype-preserving sidecar data file, the JSON-safe build manifest, immutability of
the input DataFrame, the data round-trip, the CSV format, and the guard-error
paths. The renderer (`_build_dashboard`) is exercised with a STUB ``streamlit``
injected into ``sys.modules`` -- Streamlit need not be installed -- reusing the
Phase-9 report builders and asserting tab isolation.
"""
from __future__ import annotations

import json
import os
import py_compile
import sys

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import dextra as dx


@pytest.fixture
def mixed_df():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.normal(5000, 1500, n),
        "city": rng.choice(["Cairo", "Giza", "Alex"], n),
        "churn": rng.choice(["yes", "no"], n),
    })
    df.loc[:3, "income"] = np.nan
    return df


def _json_ok(obj):
    json.loads(json.dumps(obj))


# ---------------------------------------------------------------------------
# a stub streamlit module for the renderer
# ---------------------------------------------------------------------------

class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Sidebar:
    def header(self, *a, **k):
        pass

    def selectbox(self, label, options, index=0, **k):
        return options[index]

    def checkbox(self, label, value=False, **k):
        return value

    def slider(self, label, mn, mx, val, **k):
        return val


class _FakeST:
    def __init__(self):
        self.calls = []
        self.sidebar = _Sidebar()

    def set_page_config(self, *a, **k):
        pass

    def title(self, *a, **k):
        self.calls.append("title")

    def caption(self, *a, **k):
        pass

    def subheader(self, *a, **k):
        self.calls.append("subheader")

    def info(self, *a, **k):
        self.calls.append("info")

    def warning(self, *a, **k):
        self.calls.append(("warning", a[0] if a else ""))

    def dataframe(self, *a, **k):
        self.calls.append("dataframe")

    def image(self, data, **k):
        self.calls.append(("image", len(data)))

    def tabs(self, labels):
        self.calls.append(("tabs", tuple(labels)))
        return [_Ctx() for _ in labels]


@pytest.fixture
def fake_st():
    st = _FakeST()
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = st
    try:
        yield st
    finally:
        if saved is not None:
            sys.modules["streamlit"] = saved
        else:
            sys.modules.pop("streamlit", None)


# ---------------------------------------------------------------------------
# exports
# ---------------------------------------------------------------------------

def test_exports_alias_and_no_underscore():
    assert callable(dx.dash)
    assert dx.dashapp is dx.dash


# ---------------------------------------------------------------------------
# dash() generation
# ---------------------------------------------------------------------------

def _default_format():
    """Mirror dash's data_format='auto' resolution (parquet engine -> parquet)."""
    try:
        import pyarrow  # noqa: F401
        return "parquet"
    except ImportError:
        try:
            import fastparquet  # noqa: F401
            return "parquet"
        except ImportError:
            return "csv"


def test_generates_runnable_app_and_data(mixed_df, tmp_path):
    out = str(tmp_path / "app.py")
    man = dx.dash(mixed_df, out=out, target="churn", include_model=True,
                  return_params=True, show=False)
    assert man["function"] == "dash"
    assert os.path.exists(out)
    assert os.path.exists(man["data_path"])
    assert os.path.exists(man["meta_path"])           # reproducibility manifest
    py_compile.compile(out, doraise=True)            # generated app is valid
    src = open(out, encoding="utf-8").read()
    assert "_build_dashboard" in src
    assert "streamlit run" in src
    assert os.path.basename(man["data_path"]) in src
    _json_ok(man)
    assert "Model" in man["tabs"]


def test_default_return_is_path(mixed_df, tmp_path):
    out = str(tmp_path / "app.py")
    assert dx.dash(mixed_df, out=out, show=False) == out


def test_default_format_is_not_pickle(mixed_df, tmp_path):
    out = str(tmp_path / "app.py")
    man = dx.dash(mixed_df, out=out, return_params=True, show=False)
    assert man["data_format"] == _default_format()
    assert not man["data_path"].endswith(".pkl")


def test_pickle_is_opt_in_warns_and_round_trips(mixed_df, tmp_path):
    out = str(tmp_path / "app.py")
    with pytest.warns(UserWarning, match="pickle"):
        man = dx.dash(mixed_df, out=out, data_format="pickle",
                      return_params=True, show=False)
    rt = pd.read_pickle(man["data_path"])
    pd.testing.assert_frame_equal(rt, mixed_df)


def test_csv_data_format(mixed_df, tmp_path):
    out = str(tmp_path / "app.py")
    man = dx.dash(mixed_df, out=out, data_format="csv", return_params=True,
                  show=False)
    assert man["data_path"].endswith("_data.csv")
    assert os.path.exists(man["data_path"])


def test_model_tab_hidden_without_flag(mixed_df, tmp_path):
    man = dx.dash(mixed_df, out=str(tmp_path / "app.py"), return_params=True,
                  show=False)
    assert "Model" not in man["tabs"]


def test_does_not_mutate_input(mixed_df, tmp_path):
    before = mixed_df.copy(deep=True)
    dx.dash(mixed_df, out=str(tmp_path / "app.py"), show=False)
    pd.testing.assert_frame_equal(mixed_df, before)
    assert "dextra_audit" not in mixed_df.attrs


def test_bad_data_format(mixed_df, tmp_path):
    with pytest.raises(ValueError):
        dx.dash(mixed_df, out=str(tmp_path / "app.py"), data_format="xml",
                show=False)


def test_bad_target(mixed_df, tmp_path):
    with pytest.raises(KeyError):
        dx.dash(mixed_df, out=str(tmp_path / "app.py"), target="nope",
                show=False)


# ---------------------------------------------------------------------------
# renderer (_build_dashboard) with a stubbed streamlit
# ---------------------------------------------------------------------------

def test_build_dashboard_renders_tabs(mixed_df, fake_st):
    from dextra.dashboard import _build_dashboard
    _build_dashboard(mixed_df, target="churn", include_model=True)
    assert "title" in fake_st.calls
    tabs = [c for c in fake_st.calls if isinstance(c, tuple) and c[0] == "tabs"]
    assert tabs and "Overview" in tabs[0][1]
    images = [c for c in fake_st.calls
              if isinstance(c, tuple) and c[0] == "image"]
    assert len(images) >= 2
    assert all(nbytes > 100 for _, nbytes in images)


def test_build_dashboard_hides_model_without_flag(mixed_df, fake_st):
    from dextra.dashboard import _build_dashboard
    _build_dashboard(mixed_df, include_model=False)
    tabs = [c for c in fake_st.calls
            if isinstance(c, tuple) and c[0] == "tabs"][0][1]
    assert "Model" not in tabs


def test_build_dashboard_tab_isolation(mixed_df, fake_st):
    from dextra import _compose
    from dextra.dashboard import _build_dashboard
    original = _compose._BUILDERS["bivariate"]

    def _boom(df, ctx):
        raise RuntimeError("boom")

    _compose._BUILDERS["bivariate"] = _boom
    try:
        _build_dashboard(mixed_df, include_model=False)
    finally:
        _compose._BUILDERS["bivariate"] = original
    warns = [c for c in fake_st.calls
             if isinstance(c, tuple) and c[0] == "warning"]
    assert warns and "boom" in warns[0][1]


# ---------------------------------------------------------------------------
# hardening: output_dir, metadata, generated-app runtime checks, parquet
# ---------------------------------------------------------------------------

def test_output_dir_collects_all_files(mixed_df, tmp_path):
    d = str(tmp_path / "dash")
    man = dx.dash(mixed_df, out="app.py", output_dir=d, target="churn",
                  include_model=True, return_params=True, show=False)
    assert man["out"] == os.path.join(d, "app.py")
    ext = {"parquet": "app_data.parquet", "csv": "app_data.csv"}[_default_format()]
    assert set(os.listdir(d)) == {"app.py", ext, "app_meta.json"}
    for k in ("out", "data_path", "meta_path"):
        assert os.path.exists(man[k])


def test_metadata_manifest_records_environment(mixed_df, tmp_path):
    man = dx.dash(mixed_df, out=str(tmp_path / "app.py"), return_params=True,
                  show=False)
    meta = json.load(open(man["meta_path"], encoding="utf-8"))
    _json_ok(meta)
    assert meta["dextra_version"] and meta["python_version"]
    assert meta["pandas_version"] == pd.__version__
    assert meta["data_format"] == _default_format()
    assert "settings" in meta and "tabs" in meta
    # the manifest mirrors the environment too
    assert man["metadata"]["dextra_version"] == meta["dextra_version"]


def test_generated_app_has_runtime_checks(mixed_df, tmp_path):
    out = str(tmp_path / "app.py")
    dx.dash(mixed_df, out=out, show=False)
    src = open(out, encoding="utf-8").read()
    assert "dextra[dash]" in src                 # friendly install hint
    assert "data file not found" in src          # missing-data guard
    fmt = _default_format()
    assert f"_FORMAT = {fmt!r}" in src
    assert f"read_{fmt}" in src and "DO NOT EDIT" in src


def test_parquet_format_or_clear_error(mixed_df, tmp_path):
    out = str(tmp_path / "app.py")
    try:
        import pyarrow  # noqa: F401
        has_engine = True
    except ImportError:
        has_engine = False
    if has_engine:
        man = dx.dash(mixed_df, out=out, data_format="parquet",
                      return_params=True, show=False)
        assert man["data_path"].endswith("_data.parquet")
        assert os.path.exists(man["data_path"])
        rt = pd.read_parquet(man["data_path"])
        assert list(rt.columns) == list(mixed_df.columns)
    else:
        with pytest.raises(ValueError, match="parquet engine"):
            dx.dash(mixed_df, out=out, data_format="parquet", show=False)
# ---------------------------------------------------------------------------
# audit trail in the manifest (audit #6)
# ---------------------------------------------------------------------------

def test_manifest_includes_audit_trail(mixed_df, tmp_path):
    man = dx.dash(mixed_df, out=str(tmp_path / "app.py"),
                  return_params=True, show=False)
    trail = man["dextra_audit"]
    assert isinstance(trail, list) and len(trail) == 1
    entry = trail[-1]
    assert entry["stage"] == "dashboard" and entry["function"] == "dash"
    assert entry["params"]["data_format"] == man["data_format"]
    assert entry["decision"]
    _json_ok(man)                                    # stays JSON-safe
    assert "dextra_audit" not in mixed_df.attrs      # input never mutated


def test_manifest_audit_keeps_input_history(mixed_df, tmp_path):
    df2 = mixed_df.copy()
    df2.attrs = {"dextra_audit": [{"stage": "cleaning", "function": "fillna"}]}
    man = dx.dash(df2, out=str(tmp_path / "app.py"),
                  return_params=True, show=False)
    trail = man["dextra_audit"]
    assert len(trail) == 2 and trail[0]["stage"] == "cleaning"
    assert trail[-1]["function"] == "dash"
    assert len(df2.attrs["dextra_audit"]) == 1       # input list untouched
# ---------------------------------------------------------------------------
# explicit model task (audit #7)
# ---------------------------------------------------------------------------

def test_task_param_validated_embedded_and_recorded(mixed_df, tmp_path):
    with pytest.raises(ValueError, match="task"):
        dx.dash(mixed_df, out=str(tmp_path / "app.py"), task="cluster",
                show=False)
    out = str(tmp_path / "app.py")
    man = dx.dash(mixed_df, out=out, task="regression", return_params=True,
                  show=False)
    assert man["task"] == "regression"
    src = open(out, encoding="utf-8").read()
    assert "task='regression'" in src                # forwarded to the app
    py_compile.compile(out, doraise=True)            # app still valid
    meta = json.load(open(man["meta_path"], encoding="utf-8"))
    assert meta["settings"]["task"] == "regression"
    assert man["dextra_audit"][-1]["params"]["task"] == "regression"


def test_renderer_task_control_with_stub(mixed_df, fake_st):
    from dextra.dashboard import _build_dashboard
    _build_dashboard(mixed_df, task="regression")    # stub returns the default
    assert any(isinstance(c, tuple) and c[0] == "tabs" for c in fake_st.calls)
