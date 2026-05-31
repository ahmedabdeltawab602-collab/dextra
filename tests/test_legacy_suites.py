"""Run the staged verification suites (Phases 2-5) under pytest / CI.

During development each phase shipped with a standalone ``test_*`` script
containing dozens of assertions (~600 checks total). Those scripts are
vendored into ``tests/legacy/`` and executed here so their assertions run in
continuous integration instead of by hand on a single machine.

Phase 2/3 suites assert directly (AssertionError -> failed test). Phase 4/5
suites use a check() counter and ``raise SystemExit(1)`` when any check fails;
we translate that into an assertion.
"""
from __future__ import annotations

import importlib

import matplotlib
import pytest

matplotlib.use("Agg", force=True)

# dextra hard-imports scipy (stats_advanced) and plotly (plots). Skip cleanly
# if a core dependency is missing in the runner; CI installs the full set.
pytest.importorskip("scipy")
pytest.importorskip("plotly")

LEGACY_SUITES = [
    # Phase 2 - advanced statistics
    "test_stage1", "test_stage2", "test_stage3",
    "test_stage4", "test_stage5", "test_stage6",
    # Phase 3 - cleaning
    "test_phase3_stage1", "test_phase3_stage2", "test_phase3_stage3",
    "test_col_type_fix",
    # Phase 4 - feature engineering
    "test_phase4_stage1", "test_phase4_stage2",
    "test_phase4_stage3", "test_phase4_stage4",
    # Phase 5 - feature selection
    "test_phase5_stage1", "test_phase5_stage2", "test_phase5_stage3",
]


@pytest.mark.parametrize("suite", LEGACY_SUITES)
def test_legacy_suite(suite):
    mod = importlib.import_module(f"tests.legacy.{suite}")
    assert hasattr(mod, "main"), f"{suite} has no main()"
    try:
        mod.main()
    except SystemExit as exc:
        assert not exc.code, f"{suite}.main() reported failing checks"
