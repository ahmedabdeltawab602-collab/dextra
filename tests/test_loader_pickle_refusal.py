"""m-7: dextra never loads pickle sources (pickle can execute arbitrary code);
the misleading ``allow_pickle`` parameter has been removed -- pickle is always
refused (fail closed).
"""
import pandas as pd
import pytest

import dextra as dx


def test_pickle_source_is_always_refused(tmp_path):
    p = tmp_path / "data.pkl"
    pd.DataFrame({"a": [1, 2, 3]}).to_pickle(p)
    with pytest.raises(Exception, match="pickle"):
        dx.load(str(p), show=False)


def test_allow_pickle_parameter_removed():
    # the dead/misleading flag is gone: passing it raises, not a silent no-op
    with pytest.raises(TypeError, match="allow_pickle"):
        dx.load("anything.csv", allow_pickle=True, show=False)
