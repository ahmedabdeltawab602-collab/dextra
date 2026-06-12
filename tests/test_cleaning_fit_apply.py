"""M-5 regression -- leakage-safe fit/apply for the cleaning family.

EVALUATION_REPORT.md M-5 (Major): no cleaning function accepted params /
return_params, so test data could not be filled with train statistics
through dextra at all -- while docs claimed every function honours
fit/apply. handle_missing (impute) now freezes fill values at fit and
replays them verbatim at apply.
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
