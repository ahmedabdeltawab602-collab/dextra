"""M-5 regression -- leakage-safe fit/apply for the cleaning family.

EVALUATION_REPORT.md M-5 (Major): no cleaning function accepted params /
return_params, so test data could not be filled with train statistics
through dextra at all -- while docs claimed every function honours
fit/apply. handle_missing (impute) now freezes fill values at fit and
replays them verbatim at apply; clip_outliers (winsor) freezes the
per-column bounds at fit and clips/drops held-out rows at those
train bounds.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import dextra as dx


def _train():
    return pd.DataFrame({"x": [10.0, 11.0, 12.0, None, 9.0],
                         "city": ["a", "a", None, "b", "a"],
                         "clean": [1, 2, 3, 4, 5]})


def _test():
    # deliberately far from train so any leakage shows up numerically
    return pd.DataFrame({"x": [50.0, None, 55.0],
                         "city": [None, "c", "a"],
                         "clean": [9, 9, 9]})


# --------------------------------------------------------------------------- #
# handle_missing (impute)
# --------------------------------------------------------------------------- #

def test_impute_fit_freezes_train_mean_and_apply_replays_it():
    train, test = _train(), _test()
    train_mean = train["x"].mean()
    tr, p = dx.handle_missing(train, strategy="mean", show=False, plot=False,
                              return_params=True)
    assert p["function"] == "handle_missing"
    assert abs(p["columns"]["x"]["fill_value"] - train_mean) < 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        te = dx.handle_missing(test, params=p, show=False, plot=False)
    assert abs(te["x"].iloc[1] - train_mean) < 1e-12      # train statistic
    assert abs(te["x"].iloc[1] - test["x"].mean()) > 1.0  # NOT test's
    # inputs never mutated
    assert train["x"].isna().sum() == 1 and test["x"].isna().sum() == 1


def test_impute_auto_resolves_and_freezes_on_train():
    train, test = _train(), _test()
    _, p = dx.handle_missing(train, strategy="auto", show=False, plot=False,
                             return_params=True)
    assert p["columns"]["city"]["strategy"] == "mode"
    assert p["columns"]["city"]["fill_value"] == "a"      # train mode
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        te = dx.handle_missing(test, params=p, show=False, plot=False)
    assert te["city"].iloc[0] == "a"                      # not test's mode


def test_impute_params_json_safe_and_audit_modes():
    train, test = _train(), _test()
    tr, p = dx.handle_missing(train, strategy="mean", show=False, plot=False,
                              return_params=True)
    json.dumps(p)  # must not raise
    assert tr.attrs["dextra_audit"][-1]["mode"] == "fit"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        te = dx.handle_missing(test, params=p, show=False, plot=False)
    last = te.attrs["dextra_audit"][-1]
    assert last["mode"] == "apply"
    assert "leakage-safe" in last["decision"]


def test_impute_apply_rejects_missing_fitted_column():
    train, test = _train(), _test()
    _, p = dx.handle_missing(train, strategy="mean", show=False, plot=False,
                             return_params=True)
    with pytest.raises(KeyError, match="does not match the fitted plan"):
        dx.handle_missing(test.drop(columns=["x"]), params=p,
                          show=False, plot=False)


def test_impute_apply_rejects_foreign_params():
    with pytest.raises(ValueError, match="not for 'handle_missing'"):
        dx.handle_missing(_test(), params={"function": "scale"},
                          show=False, plot=False)


def test_impute_dry_run_guards():
    train = _train()
    with pytest.raises(ValueError, match="does not fit parameters"):
        dx.handle_missing(train, dry_run=True, return_params=True,
                          show=False, plot=False)
    _, p = dx.handle_missing(train, strategy="mean", show=False, plot=False,
                             return_params=True)
    with pytest.raises(ValueError, match="cannot.*be combined|fit-mode"):
        dx.handle_missing(train, dry_run=True, params=p,
                          show=False, plot=False)


def test_impute_apply_warns_on_unfitted_missing_column():
    train, test = _train(), _test()
    _, p = dx.handle_missing(train, strategy="mean", show=False, plot=False,
                             return_params=True)
    test["newcol"] = [None, 1.0, 2.0]
    with pytest.warns(UserWarning, match="no fitted fill"):
        te = dx.handle_missing(test, params=p, show=False, plot=False)
    assert te["newcol"].isna().sum() == 1  # honestly left as NaN


def test_impute_drop_cols_decision_is_replayed_not_redecided():
    tr = pd.DataFrame({"bad": [None, None, None, 1.0],
                       "good": [1.0, 2.0, 3.0, 4.0]})
    te = pd.DataFrame({"bad": [1.0, 2.0], "good": [5.0, 6.0]})  # bad is clean
    _, p = dx.handle_missing(tr, strategy="drop_cols", drop_threshold=0.6,
                             show=False, plot=False, return_params=True)
    out = dx.handle_missing(te, params=p, show=False, plot=False)
    assert "bad" not in out.columns and "good" in out.columns


def test_impute_return_contract_unchanged_and_extended():
    train = _train()
    bare = dx.handle_missing(train, strategy="mean", show=False, plot=False)
    assert isinstance(bare, pd.DataFrame)            # bare: df, as always
    pair = dx.handle_missing(train, strategy="mean", show=False, plot=False,
                             return_params=True)
    assert isinstance(pair, tuple) and len(pair) == 2
    triple = dx.handle_missing(train, strategy="mean", show=False, plot=True,
                               return_params=True, return_fig=True)
    assert len(triple) == 3                          # (df, params, fig)
    assert isinstance(triple[0], pd.DataFrame) and isinstance(triple[1], dict)


def test_impute_alias_has_fit_apply_too():
    train, test = _train(), _test()
    _, p = dx.impute(train, strategy="median", show=False, plot=False,
                     return_params=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        te = dx.impute(test, params=p, show=False, plot=False)
    assert abs(te["x"].iloc[1] - train["x"].median()) < 1e-12


# --------------------------------------------------------------------------- #
# clip_outliers (winsor)
# --------------------------------------------------------------------------- #

def _train_out():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"x": list(rng.normal(10, 2, 50)) + [100.0],
                         "y": rng.normal(0, 1, 51)})


def test_winsor_fit_freezes_train_tukey_bounds():
    train = _train_out()
    q1, q3 = train["x"].quantile(0.25), train["x"].quantile(0.75)
    lb, ub = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    _, p = dx.clip_outliers(train, show=False, plot=False, return_params=True)
    assert p["function"] == "clip_outliers" and p["method"] == "iqr"
    assert abs(p["columns"]["x"]["lower_bound"] - lb) < 1e-12
    assert abs(p["columns"]["x"]["upper_bound"] - ub) < 1e-12
    json.dumps(p)  # JSON-safe


def test_winsor_apply_clips_at_train_bounds_not_test_bounds():
    train = _train_out()
    test = pd.DataFrame({"x": [500.0, 10.0, -200.0], "y": [0.0, 0.1, -0.1]})
    _, p = dx.clip_outliers(train, show=False, plot=False, return_params=True)
    te = dx.clip_outliers(test, params=p, show=False, plot=False)
    lb = p["columns"]["x"]["lower_bound"]
    ub = p["columns"]["x"]["upper_bound"]
    assert abs(te["x"].iloc[0] - ub) < 1e-12   # clipped to TRAIN upper
    assert abs(te["x"].iloc[2] - lb) < 1e-12   # clipped to TRAIN lower
    assert te["x"].iloc[1] == 10.0             # in-range untouched
    # test's own Tukey bound is wildly different -> proves no re-fit
    tq1, tq3 = test["x"].quantile(0.25), test["x"].quantile(0.75)
    assert abs((tq3 + 1.5 * (tq3 - tq1)) - ub) > 100
    # inputs never mutated
    assert test["x"].iloc[0] == 500.0 and train["x"].iloc[50] == 100.0


def test_winsor_drop_action_replayed_at_train_bounds():
    train = _train_out()
    test = pd.DataFrame({"x": [500.0, 10.0, -200.0], "y": [0.0, 0.1, -0.1]})
    _, p = dx.clip_outliers(train, action="drop", show=False, plot=False,
                            return_params=True)
    te = dx.clip_outliers(test, params=p, show=False, plot=False)
    assert len(te) == 1 and te["x"].iloc[0] == 10.0


def test_winsor_zscore_bounds_frozen():
    train = _train_out()
    _, p = dx.clip_outliers(train, method="zscore", z_threshold=3.0,
                            show=False, plot=False, return_params=True)
    mu, sd = train["x"].mean(), train["x"].std()
    assert abs(p["columns"]["x"]["upper_bound"] - (mu + 3 * sd)) < 1e-9


def test_winsor_apply_rejects_missing_fitted_column():
    train = _train_out()
    _, p = dx.clip_outliers(train, show=False, plot=False, return_params=True)
    with pytest.raises(KeyError, match="does not match the fitted bounds"):
        dx.clip_outliers(pd.DataFrame({"z": [1.0]}), params=p,
                         show=False, plot=False)


def test_winsor_apply_rejects_foreign_params_and_dry_run():
    train = _train_out()
    with pytest.raises(ValueError, match="not for 'clip_outliers'"):
        dx.clip_outliers(train, params={"function": "scale"},
                         show=False, plot=False)
    with pytest.raises(ValueError, match="does not fit parameters"):
        dx.clip_outliers(train, dry_run=True, return_params=True,
                         show=False, plot=False)
    _, p = dx.clip_outliers(train, show=False, plot=False, return_params=True)
    with pytest.raises(ValueError, match="fit-mode"):
        dx.clip_outliers(train, dry_run=True, params=p,
                         show=False, plot=False)


def test_winsor_audit_modes_and_return_contract():
    train = _train_out()
    tr, p = dx.clip_outliers(train, show=False, plot=False,
                             return_params=True)
    assert tr.attrs["dextra_audit"][-1]["mode"] == "fit"
    te = dx.clip_outliers(train, params=p, show=False, plot=False)
    last = te.attrs["dextra_audit"][-1]
    assert last["mode"] == "apply" and "leakage-safe" in last["decision"]
    bare = dx.clip_outliers(train, show=False, plot=False)
    assert isinstance(bare, pd.DataFrame)
    triple = dx.clip_outliers(train, show=False, plot=True,
                              return_params=True, return_fig=True)
    assert len(triple) == 3 and isinstance(triple[1], dict)


def test_winsor_alias_has_fit_apply_too():
    train = _train_out()
    test = pd.DataFrame({"x": [500.0, 10.0, -200.0], "y": [0.0, 0.1, -0.1]})
    _, p = dx.winsor(train, show=False, plot=False, return_params=True)
    te = dx.winsor(test, params=p, show=False, plot=False)
    assert abs(te["x"].iloc[0] - p["columns"]["x"]["upper_bound"]) < 1e-12

