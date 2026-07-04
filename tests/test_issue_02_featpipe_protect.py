"""Regression for issue #2 (delegated decision 2).

``featpipe(protect=[...])`` isolates the named columns from every step
that auto-selects its columns (the CHURN {-1,1} target swallowed by a
bare ``scale`` step), and ``step_summary`` names the columns each step
actually touched.
"""
import numpy as np
import pandas as pd
import pytest

import dextra as dx


def _df():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "income": rng.normal(50_000, 15_000, 40),
            "age": rng.integers(18, 70, 40).astype(float),
            "CHURN": np.where(rng.random(40) > 0.5, 1, -1),
        }
    )


def _fp(df, **kw):
    ret = dx.featpipe(df, show=False, plot=False, return_df=True, **kw)
    return ret if isinstance(ret, tuple) else (ret, None)


def test_issue_02_bare_scale_swallows_target_without_protect():
    df = _df()
    out, _ = _fp(df, steps=[{"fn": "scale"}])
    # documents the hazard: the numeric target leaks -- either overwritten
    # in place, or emitted as a derived copy (CHURN_standard).
    churn_derived = [c for c in out.columns
                     if c != "CHURN" and c.startswith("CHURN")]
    assert churn_derived or not out["CHURN"].equals(df["CHURN"])


def test_issue_02_protect_leaves_target_untouched():
    df = _df()
    out, p = _fp(df, steps=[{"fn": "scale"}], protect=["CHURN"],
                 return_params=True)
    pd.testing.assert_series_equal(out["CHURN"], df["CHURN"])
    assert out["CHURN"].dtype == df["CHURN"].dtype
    # no derived copy of the target either
    assert not [c for c in out.columns
                if c != "CHURN" and c.startswith("CHURN")]
    # features were still scaled (in place or as derived columns)
    added = [c for c in out.columns if c not in df.columns]
    assert added or not out["income"].equals(df["income"])
    # protect is recorded in the artifact
    assert p["metadata"]["protect"] == ["CHURN"]


def test_issue_02_step_summary_names_touched_columns():
    df = _df()
    _, p = _fp(df, steps=[{"fn": "scale"}], protect=["CHURN"],
               return_params=True)
    entry = p["metadata"]["step_summary"][0]
    assert "cols_touched" in entry and "cols_added" in entry
    affected = set(entry["cols_touched"]) | set(entry["cols_added"])
    # the step literally discloses what it did -- and it did something
    assert affected
    # every affected column derives from the features, never from the target
    assert all(c.startswith(("income", "age")) for c in affected), affected
    assert not any("CHURN" in c for c in affected)


def test_issue_02_explicit_cols_naming_protected_errors():
    df = _df()
    with pytest.raises(ValueError, match="protect"):
        _fp(df, steps=[{"fn": "scale", "cols": ["CHURN", "income"]}],
            protect=["CHURN"])


def test_issue_02_fit_unknown_protect_column_errors():
    df = _df()
    with pytest.raises(KeyError):
        _fp(df, steps=[{"fn": "scale"}], protect=["nope"])


def test_issue_02_apply_tolerates_absent_protected_column():
    df = _df()
    _, p = _fp(df, steps=[{"fn": "scale"}], protect=["CHURN"],
               return_params=True)
    test_no_target = df.drop(columns=["CHURN"]).copy()
    out = dx.featpipe(test_no_target, params=p, show=False, plot=False,
                      return_df=True)
    assert "CHURN" not in out.columns
    # and with the column present, apply leaves it untouched too
    out2 = dx.featpipe(df, params=p, show=False, plot=False, return_df=True)
    pd.testing.assert_series_equal(out2["CHURN"], df["CHURN"])
