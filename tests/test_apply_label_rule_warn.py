"""m-3: model apply-mode displays label their metrics as fit-time (so a reader
does not mistake them for metrics on the new data).
m-6: validate_rules warns when a rule could not be evaluated (ERROR), even when
show=False (the silent path used to hide it).
"""
import pandas as pd
import pytest

import dextra as dx


def test_regress_apply_labels_fit_time_metrics(capsys):
    df = pd.DataFrame({"x": list(range(40)),
                       "y": [2 * i + 1 for i in range(40)]})
    _, params = dx.regress(df, y="y", method="linear", return_params=True,
                           show=False, plot=False)
    dx.regress(df, params=params, show=True, plot=False)   # apply mode
    out = capsys.readouterr().out
    assert "fit-time metrics" in out


def test_validate_rules_warns_on_silent_error():
    df = pd.DataFrame({"a": [1, 2, 3]})
    rules = [{"name": "broken", "check": "nonexistent_col > 0"}]  # raises -> ERROR
    with pytest.warns(UserWarning, match="could not be evaluated"):
        dx.validate_rules(df, rules, show=False, plot=False)
