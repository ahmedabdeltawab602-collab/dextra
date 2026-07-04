"""Regression for issue #1.

Optional-dependency *install hints* must name the PyPI distribution
(``pydextra``), not the import package (``dextra``).  ``pip install
"dextra[ts]"`` points at the wrong (or non-existent) project.
"""
import os
import re
import sys

import dextra


def test_issue_01_tsstat_hint_names_pydextra(monkeypatch):
    # Force statsmodels to look missing so the gate raises regardless of
    # whether statsmodels is installed in this environment.
    monkeypatch.setitem(sys.modules, "statsmodels", None)
    from dextra import timeseries

    try:
        timeseries._require_statsmodels("tsstat")
    except ImportError as exc:
        msg = str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ImportError when statsmodels is absent")

    assert "pydextra[ts]" in msg
    assert '"dextra[ts]"' not in msg
    assert "dextra[ts]" not in msg.replace("pydextra[ts]", "")


def test_issue_01_no_bare_dextra_extra_hint_in_runtime_source():
    pkg_dir = os.path.dirname(dextra.__file__)
    bad = re.compile(r"(?<!py)dextra\[[a-z]+\]")
    offenders = []
    for root, _dirs, files in os.walk(pkg_dir):
        if "__pycache__" in root:
            continue
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    for m in bad.finditer(line):
                        offenders.append(f"{fn}:{i}: {m.group(0)}")
    assert not offenders, "bare dextra[...] hints: " + "; ".join(offenders)
