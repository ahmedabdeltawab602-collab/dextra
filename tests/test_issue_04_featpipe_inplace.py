"""Regression for issue #4 (delegated decision 3).

Inside ``featpipe``, steps that support ``inplace`` (transform / scale /
bin / encode) default to ``inplace=True`` so a plain recipe yields a
model-ready frame (no raw 'Male' column reaching the estimator, no
duplicated raw+derived pairs). An explicit ``'inplace': False`` in a step
is honoured, and single calls outside the pipeline keep their default.
"""
import numpy as np
import pandas as pd

import dextra as dx


def _df():
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "income": rng.normal(50_000, 15_000, 30),
            "age": rng.integers(18, 70, 30).astype(float),
            "city": rng.choice(["Cairo", "Giza", "Alex"], 30),
        }
    )


def _fp(df, **kw):
    ret = dx.featpipe(df, show=False, plot=False, return_df=True, **kw)
    return ret if isinstance(ret, tuple) else (ret, None)


def test_issue_04_default_recipe_is_model_ready():
    df = _df()
    out, _ = _fp(df, steps=[{"fn": "scale"}, {"fn": "encode"}])
    # scaled in place: same column name, transformed values
    assert "income" in out.columns
    assert "income_standard" not in out.columns
    assert not out["income"].equals(df["income"])
    # encoded in place: raw categorical gone, dummies present
    assert "city" not in out.columns
    assert any(c.startswith("city_") for c in out.columns)
    # nothing non-numeric survives -> ready for modeling
    assert out.select_dtypes(exclude="number").shape[1] == 0


def test_issue_04_explicit_false_is_honoured():
    df = _df()
    out, _ = _fp(df, steps=[{"fn": "scale", "inplace": False}])
    pd.testing.assert_series_equal(out["income"], df["income"])
    assert "income_standard" in out.columns


def test_issue_04_single_call_outside_pipeline_unchanged():
    df = _df()
    out = dx.scale(df, cols=["income"], show=False, plot=False,
                   return_df=True)
    if isinstance(out, tuple):
        out = out[0]
    # standalone default stays non-inplace
    pd.testing.assert_series_equal(out["income"], df["income"])
    assert "income_standard" in out.columns


def test_issue_04_apply_replays_forced_inplace():
    df = _df()
    fit_out, p = _fp(df, steps=[{"fn": "scale"}, {"fn": "encode"}],
                     return_params=True)
    df2 = _df().iloc[::-1].reset_index(drop=True)
    apply_out = dx.featpipe(df2, params=p, show=False, plot=False,
                            return_df=True)
    assert list(apply_out.columns) == list(fit_out.columns)
    assert apply_out.select_dtypes(exclude="number").shape[1] == 0


def test_issue_04_stale_suffix_reference_gets_migration_hint():
    """A 0.5.x-style recipe referencing a suffixed column that the forced
    inplace default no longer creates must fail WITH migration guidance."""
    import pytest

    df = _df()
    recipe = [
        {"fn": "transform", "cols": ["income"], "method": "log1p"},
        {"fn": "scale", "cols": ["income_log1p"], "method": "robust"},
    ]
    with pytest.raises(KeyError) as ei:
        dx.featpipe(df, steps=recipe, show=False, plot=False)
    msg = str(ei.value)
    assert "income_log1p" in msg
    assert "inplace" in msg and "False" in msg, msg
    assert "0.6.0" in msg, msg


def test_issue_04_hint_also_when_failing_step_lacks_inplace():
    import pytest

    df = _df()
    recipe = [
        {"fn": "transform", "cols": ["income"], "method": "log1p"},
        {"fn": "cross", "pairs": [("income_log1p", "age")],
         "method": "ratio"},
    ]
    with pytest.raises((KeyError, ValueError)) as ei:
        dx.featpipe(df, steps=recipe, show=False, plot=False)
    msg = str(ei.value)
    if "not in df" in msg:
        assert "inplace" in msg and "0.6.0" in msg, msg
