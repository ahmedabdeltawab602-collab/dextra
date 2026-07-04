"""Regression for issue #8.

``edareport`` must not leak scipy ConstantInputWarning when a numeric
column is constant: such columns are excluded from the correlation matrix
and named in the report.
"""
import warnings

import numpy as np
import pandas as pd

import dextra as dx


def _frame():
    rng = np.random.default_rng(0)
    x = np.arange(30, dtype=float)
    return pd.DataFrame(
        {
            "x": x,
            "y": 2.0 * x + rng.normal(0, 1.0, 30),
            "const_col": np.full(30, 7.0),  # zero variance
        }
    )


def _constant_warnings(rec):
    out = []
    for w in rec:
        name = w.category.__name__
        msg = str(w.message).lower()
        if "Constant" in name or "constant" in msg:
            out.append(f"{name}: {w.message}")
    return out


def test_issue_08_edareport_no_constant_input_warning(tmp_path):
    out = str(tmp_path / "r.html")
    df = _frame()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        dx.edareport(df, out=out, show=False, include_model=False)
    assert not _constant_warnings(rec), _constant_warnings(rec)
    html = open(out, encoding="utf-8").read()
    # the constant column is named as excluded from correlations
    assert "constant column(s) from correlations" in html, "no exclusion note"
    assert "const_col" in html


def test_issue_08_correlation_matrix_excludes_constant(tmp_path):
    df = _frame()
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with redirect_stdout(buf):
            dx.correlation_matrix(df, show=True, plot=False)
    assert not _constant_warnings(rec), _constant_warnings(rec)
    assert "const_col" in buf.getvalue()
