"""Coverage tests for the cleaning module's visual / show / strategy paths.

The silent paths (show=False, plot=False) are covered by the dedicated and
legacy suites; this file exercises what they skip: the ten `_plot_*` helpers,
the `show=True` console blocks, the full `handle_missing` strategy matrix,
and the guard / error branches. Assertions are deliberately loose -- the goal
is executing the paths without crashing, plus shape/type sanity checks.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402

from dextra.cleaning import (  # noqa: E402
    cast_types,
    clean_report,
    clip_outliers,
    dedupe,
    dup_show,
    handle_missing,
    na_show,
    out_show,
    standardize_columns,
    validate_rules,
)


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    plt.close("all")


# --- data builders (plain helpers, not fixtures, so paths stay explicit) ---

def _rich():
    """One frame that hits every clean_report tier / panel branch."""
    n = 50
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "ok_num": rng.normal(0.0, 1.0, n),
        "tiny_miss": np.where(np.arange(n) == 0, np.nan, 1.0 * np.arange(n)),
        "mid_miss_num": np.where(np.arange(n) < 10, np.nan, rng.normal(5, 1, n)),
        "mid_miss_txt": [None if i < 10 else f"t{i % 7}" for i in range(n)],
        "heavy_miss": np.where(np.arange(n) < 22, np.nan, 1.0 * (np.arange(n) % 3)),
        "dead_col": [np.nan] * n,
        "drop_me": np.where(np.arange(n) < 35, np.nan, 1.0 * (np.arange(n) % 2)),
        "const": ["same"] * n,
        "flag": rng.choice([True, False], n),
        "when": pd.date_range("2024-01-01", periods=n),
        "cat": pd.Categorical(rng.choice(["x", "y"], n)),
        "dur": pd.to_timedelta(np.arange(n), unit="s"),
        "ws": [" padded "] + ["ok"] * (n - 1),
        "case": ["Cairo", "cairo"] + ["giza"] * (n - 2),
    })
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)  # + 1 dup row


def _dups():
    return pd.DataFrame({"a": [1, 1, 2, 2, 3, 4, 4],
                         "b": ["x", "x", "y", "y", "z", "w", "w"]})


def _outliery(n=60):
    rng = np.random.default_rng(5)
    df = pd.DataFrame({
        "v1": np.append(rng.normal(0, 1, n - 2), [99.0, -99.0]),
        "v2": np.append(rng.normal(50, 5, n - 1), [500.0]),
        "flatc": [7.0] * n,
        "alln": [np.nan] * n,
        "txt": ["a"] * n,
    })
    return df


# --- clean_report -----------------------------------------------------------

def test_clean_report_full_visual():
    df = _rich()
    summary, fig = clean_report(df, show=True, plot=True,
                                return_df=True, return_fig=True)
    assert fig is not None
    actions = set(summary["suggested_action"])
    assert {"OK", "review", "consider_drop", "drop_column",
            "drop_constant"} <= actions
    assert {"impute_median", "impute_mode"} <= actions


def test_clean_report_zero_rows_and_numeric_only():
    out = clean_report(pd.DataFrame({"a": []}), show=True, plot=False,
                       return_df=True)
    assert len(out) == 1  # one profiled column
    num = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    assert clean_report(num, show=False, plot=False, return_df=True) is not None


# --- standardize_columns ----------------------------------------------------

def test_standardize_columns_visual_and_second_pass():
    messy = pd.DataFrame({
        " First Name ": ["  Ann ", "Bob", None],
        "Income($)": [1.0, 2.0, 3.0],
        "Income($)#": [4, 5, 6],       # collides after normalisation
        "###": ["a", " b ", "c"],      # normalises to 'unnamed'
    })
    out, fig = standardize_columns(messy, show=True, plot=True,
                                   return_df=True, return_fig=True)
    assert fig is not None
    assert "unnamed" in out.columns and "income_1" in out.columns
    # Second pass: nothing left to rename or strip (else-branches + panels).
    out2, fig2 = standardize_columns(out, show=True, plot=True,
                                     return_df=True, return_fig=True)
    assert fig2 is not None and list(out2.columns) == list(out.columns)


def test_standardize_columns_dry_run_and_many_renames():
    messy = pd.DataFrame({f"Col {i}!": [1] for i in range(25)})
    out = standardize_columns(messy, dry_run=True, show=True, plot=False,
                              return_df=True)
    assert list(out.columns) == list(messy.columns)  # dry-run: unchanged
    fig = standardize_columns(messy, show=False, plot=True, return_fig=True)
    assert fig is not None  # ">20 renames" panel branch


# --- cast_types --------------------------------------------------------------

def test_cast_types_auto_visual():
    df = pd.DataFrame({
        "b": ["yes", "no", "YES", None, "no"],
        "d": ["2024-01-01", "2024-02-01", None, "2024-03-01", "2024-04-01"],
        "m": ["$1,000", "$2,000", "$3,500", None, "$4,000"],
        "g": ["a", "b", "a", "b", "a"],
        "empty_o": pd.Series([None] * 5, dtype="object"),
        "x": [1.5, 2.5, 3.5, 4.5, 5.5],
    })
    out, fig = cast_types(df, show=True, plot=True,
                          return_df=True, return_fig=True)
    assert fig is not None
    assert pd.api.types.is_bool_dtype(out["b"])
    assert str(out["d"].dtype).startswith("datetime")
    assert pd.api.types.is_numeric_dtype(out["m"])
    assert isinstance(out["g"].dtype, pd.CategoricalDtype)


def test_cast_types_no_changes_and_schema_paths():
    clean = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    fig = cast_types(clean, show=True, plot=True, return_fig=True)
    assert fig is not None  # "No type changes applied" panel
    df = pd.DataFrame({"d": ["2024-01-01", "2024-06-01"],
                       "s": ["1", "2"],
                       "g": ["a", "b"]})
    out = cast_types(df, schema={"d": "datetime64[ns]", "s": "float64"},
                     show=False, plot=False, return_df=True)
    assert str(out["d"].dtype).startswith("datetime")
    assert pd.api.types.is_float_dtype(out["s"])
    with pytest.raises(KeyError):
        cast_types(df, schema={"zzz": "int64"}, show=False, plot=False)
    with pytest.raises(ValueError):
        cast_types(df, schema={"g": "int64"}, show=False, plot=False)
    with pytest.raises(ValueError):
        cast_types(df, parse_threshold=0.0, show=False, plot=False)


# --- validate_rules -----------------------------------------------------------

def test_validate_rules_visual_all_status_colors():
    df = pd.DataFrame({"x": np.arange(120, dtype=float)})
    rules = [
        {"name": "all_pass",  "check": "x >= 0"},
        {"name": "few_fail",  "check": "x >= 5", "severity": "warning"},
        {"name": "mid_fail",  "check": "x >= 60",
         "description": "upper half only"},
        {"name": "big_fail",  "check": "x > 200"},
        {"name": "callable_r", "check": lambda d: d["x"] < 1000},
        {"name": "err_expr",  "check": "no_such_col > 0"},
        {"name": "bad_call",  "check": lambda d: 5},
    ]
    summary, fig = validate_rules(df, rules, show=True, plot=True,
                                  return_df=True, return_fig=True)
    assert fig is not None
    st = summary["status"]
    assert st.loc["all_pass"] == "PASS" and st.loc["callable_r"] == "PASS"
    assert st.loc["few_fail"] == "FAIL" and st.loc["big_fail"] == "FAIL"
    assert st.loc["err_expr"] == "ERROR" and st.loc["bad_call"] == "ERROR"


def test_validate_rules_violations_and_guards():
    df = pd.DataFrame({"x": [0.0, 1.0, 5.0]})
    rules = [{"name": "small", "check": "x < 2"}]
    summary, flagged, fig = validate_rules(
        df, rules, show=False, plot=True,
        return_df=True, return_fig=True, return_violations=True)
    assert len(flagged) == 1 and "violated_rules" in flagged.columns
    empty = validate_rules(df, [{"name": "ok", "check": "x >= 0"}],
                           show=False, plot=False, return_violations=True)
    assert len(empty) == 0 and "violated_rules" in empty.columns
    with pytest.raises(ValueError):
        validate_rules(df, [], show=False, plot=False)
    with pytest.raises(ValueError):
        validate_rules(df, [{"name": "nocheck"}], show=False, plot=False)


# --- handle_missing -----------------------------------------------------------

_STRATEGIES = ["mean", "median", "mode", "ffill", "bfill", "constant",
               "interpolate", "random_uniform", "random_normal",
               "random_sample", "keep"]


@pytest.mark.parametrize("strategy", _STRATEGIES)
def test_handle_missing_each_strategy(strategy):
    df = pd.DataFrame({
        "num": [1.0, np.nan, 3.0, 4.0, np.nan, 2.0],
        "flat": [5.0, 5.0, np.nan, 5.0, 5.0, 5.0],
        "txt": ["a", None, "b", "a", None, "a"],
        "cat": pd.Categorical(["u", None, "v", "u", None, "u"]),
    })
    out = handle_missing(df, strategy=strategy, fill_value=0,
                         random_state=11, show=False, plot=False,
                         return_df=True)
    assert isinstance(out, pd.DataFrame) and out.shape == df.shape
    if strategy == "keep":
        assert out.isna().sum().sum() == df.isna().sum().sum()
    if strategy in ("mean", "median", "interpolate",
                    "random_uniform", "random_normal"):
        assert out["num"].isna().sum() == 0
    if strategy in ("mode", "ffill", "bfill", "constant", "random_sample"):
        assert out.isna().sum().sum() == 0


def test_handle_missing_auto_visual():
    n = 30
    rng = np.random.default_rng(9)
    df = pd.DataFrame({
        "sym": rng.normal(0, 1, n),
        "skewed": rng.lognormal(0, 1.2, n),
        "tiny": [1.0, 2.0] + [np.nan] * (n - 2),
        "txt": ["a"] * (n - 3) + [None] * 3,
        "boolean": pd.array([True, False] * (n // 2 - 1) + [None, None],
                            dtype="boolean"),
        "when": pd.Series(pd.date_range("2024-01-01", periods=n)),
    })
    df.loc[df.index[:4], "sym"] = np.nan
    df.loc[df.index[:4], "skewed"] = np.nan
    df.loc[df.index[5:7], "when"] = pd.NaT
    out, fig = handle_missing(df, strategy="auto", show=True, plot=True,
                              return_df=True, return_fig=True)
    assert fig is not None
    assert out.isna().sum().sum() == 0


def test_handle_missing_drop_dict_dry_run_and_guards():
    df = pd.DataFrame({
        "mostly_gone": [np.nan, np.nan, np.nan, np.nan, 1.0],
        "num": [1.0, np.nan, 3.0, 4.0, 5.0],
        "txt": ["a", None, "b", None, "c"],
    })
    dropped_rows = handle_missing(df, strategy="drop_rows", show=True,
                                  plot=False, return_df=True)
    assert len(dropped_rows) == 1
    dropped_cols = handle_missing(df, strategy="drop_cols",
                                  drop_threshold=0.6, show=True,
                                  plot=False, return_df=True)
    assert "mostly_gone" not in dropped_cols.columns
    # dict strategy: one specified, one skipped, one bogus (error-logged)
    out = handle_missing(df, strategy={"num": "median", "txt": "nope"},
                         show=True, plot=False, return_df=True)
    assert out["num"].isna().sum() == 0
    assert out["txt"].isna().sum() == 2  # bogus strategy left it untouched
    # dry-run with a dict that leaves a column unspecified
    dry = handle_missing(df, strategy={"num": "median"}, dry_run=True,
                         show=True, plot=False, return_df=True)
    assert dry["num"].isna().sum() == 1  # nothing actually imputed
    # complete frame -> "No missing values to handle."
    handle_missing(pd.DataFrame({"x": [1.0, 2.0]}), strategy="mean",
                   show=True, plot=False)
    with pytest.raises(ValueError):
        handle_missing(df, strategy="zzz", show=False, plot=False)
    with pytest.raises(KeyError):
        handle_missing(df, strategy={"ghost": "mean"}, show=False, plot=False)


# --- dedupe -------------------------------------------------------------------

def test_dedupe_variants_visual():
    df = _dups()
    out, fig = dedupe(df, show=True, plot=True, return_df=True,
                      return_fig=True)
    assert fig is not None and len(out) == 4
    assert len(dedupe(df, keep="last", show=False, plot=False,
                      return_df=True)) == 4
    assert len(dedupe(df, keep=False, show=False, plot=False,
                      return_df=True)) == 1
    assert len(dedupe(df, drop_indices=[1, 3], show=False, plot=False,
                      return_df=True)) == 5
    dry = dedupe(df, dry_run=True, show=True, plot=False, return_df=True)
    assert len(dry) == len(df)
    dry2 = dedupe(df, drop_indices=[1], dry_run=True, show=False,
                  plot=False, return_df=True)
    assert len(dry2) == len(df)
    removed = dedupe(df, return_dropped=True, show=False, plot=False)
    assert len(removed) == 3
    out3, removed3 = dedupe(df, return_dropped=True, return_df=True,
                            show=False, plot=False)
    assert len(out3) + len(removed3) == len(df)


def test_dedupe_guards():
    df = _dups()
    with pytest.raises(ValueError):
        dedupe(df, keep="bad", show=False, plot=False)
    with pytest.raises(KeyError):
        dedupe(df, subset=["ghost"], show=False, plot=False)
    with pytest.raises(KeyError):
        dedupe(df, drop_indices=[999], show=False, plot=False)


# --- clip_outliers --------------------------------------------------------------

def test_clip_outliers_variants_visual():
    df = _outliery()
    out = clip_outliers(df, show=True, plot=True)  # default return: frame
    assert isinstance(out, pd.DataFrame)
    assert out["v1"].max() < 99.0
    z = clip_outliers(df, cols=["v1", "flatc"], method="zscore",
                      z_threshold=3.0, show=False, plot=False,
                      return_df=True)
    assert isinstance(z, pd.DataFrame)
    dropped = clip_outliers(df, cols=["v1", "v2"], action="drop",
                            show=True, plot=True, return_df=True)
    assert len(dropped) < len(df)
    dry = clip_outliers(df, cols=["v1"], dry_run=True, show=True,
                        plot=False, return_df=True)
    assert dry["v1"].max() == df["v1"].max()


def test_clip_outliers_guards():
    df = _outliery()
    with pytest.raises(ValueError):
        clip_outliers(df, method="nope", show=False, plot=False)
    with pytest.raises(ValueError):
        clip_outliers(df, action="nope", show=False, plot=False)
    with pytest.raises(ValueError):
        clip_outliers(df, k=-1.0, show=False, plot=False)
    with pytest.raises(ValueError):
        clip_outliers(df[["txt"]], show=False, plot=False)
    with pytest.raises(KeyError):
        clip_outliers(df, cols=["ghost"], show=False, plot=False)


# --- inspectors -----------------------------------------------------------------

def test_na_show_visual_subsample_and_complete():
    n = 250
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(0, 1, n),
                       "b": rng.normal(0, 1, n),
                       "t": ["q"] * n})
    df.loc[df.index[:30], "a"] = np.nan
    df.loc[df.index[10:14], "t"] = None
    diag, fig = na_show(df, show=True, plot=True, max_rows_shown=5,
                        return_df=True, return_fig=True)
    assert fig is not None
    assert {"which_cols_missing", "n_missing_in_row"} <= set(diag.columns)
    clean_diag = na_show(df.dropna(), show=True, plot=True, return_df=True)
    assert len(clean_diag) == 0


def test_dup_show_visual_subset_and_clean():
    df = _dups()
    diag, fig = dup_show(df, show=True, plot=True, return_df=True,
                         return_fig=True)
    assert fig is not None
    assert {"dup_group_id", "group_size", "is_first_in_group"} <= set(diag.columns)
    sub = dup_show(df, subset=["a"], show=True, plot=False, return_df=True)
    assert len(sub) >= len(diag) - 1
    none_found = dup_show(df.drop_duplicates(), show=True, plot=True,
                          return_df=True)
    assert len(none_found) == 0
    with pytest.raises(KeyError):
        dup_show(df, subset=["ghost"], show=False, plot=False)


def test_out_show_visual_paths():
    df = _outliery()
    diag, fig = out_show(df, cols=["v1", "v2", "alln"], show=True,
                         plot=True, return_df=True, return_fig=True)
    assert fig is not None
    assert {"outlier_in_columns", "severity_z", "severity_iqr"} <= set(diag.columns)
    single_fig = out_show(df, cols=["v1"], show=False, plot=True,
                          return_fig=True)
    assert single_fig is not None
    zdiag = out_show(df, cols=["v2", "flatc"], method="zscore",
                     show=False, plot=True, return_df=True)
    assert len(zdiag) >= 1
    calm = out_show(pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}),
                    show=True, plot=False, return_df=True)
    assert len(calm) == 0
    assert out_show(df[["txt"]], show=True, plot=False) is None
    with pytest.raises(ValueError):
        out_show(df, method="nope", show=False, plot=False)
    with pytest.raises(KeyError):
        out_show(df, cols=["ghost"], show=False, plot=False)
