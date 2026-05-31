"""
test_phase5_stage3.py
---------------------
Verification script for dextra Phase 5 Stage 5.3 -- selectpipe().

Run inside the project venv:
    cd "D:\\06 PythonProjects\\dextra-project"
    .\\.venv\\Scripts\\Activate.ps1
    python "C:\\Users\\manim\\OneDrive\\المستندات\\Claude\\Projects\\شرح بايثون وعمل مكتبة dextra\\test_phase5_stage3.py"

Covers: the fit/apply contract of the selection-pipeline wrapper, the combined
params artifact (JSON-serialisable), the round-trip (fit -> apply -> save ->
load -> apply == identical), idempotency, immutability, the compare ban,
mode-conflict errors, step-error context, the shared-target injection, and the
shielding of the target column from target-free redundancy steps.
"""
from __future__ import annotations

import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import dextra as dx

try:
    import sklearn  # noqa: F401
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False

ok = fail = 0


def check(name: str, cond: bool) -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  [PASS] {name}")
    else:
        fail += 1
        print(f"  [FAIL] {name}")


def section(title: str) -> None:
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def main() -> None:
    print(f"dextra version: {dx.__version__}")
    print(f"scikit-learn available: {HAVE_SKLEARN}")
    check("selectpipe exported", hasattr(dx, "selectpipe"))
    check("alias selpipe exported", hasattr(dx, "selpipe"))

    rng = np.random.default_rng(13)
    n = 400
    income = rng.gamma(2.0, 1000.0, n)
    age = rng.normal(40, 12, n)
    tenure = rng.normal(5, 2, n)
    base = pd.DataFrame({
        "income": income,
        "age": age,
        "tenure": tenure,
        "const": np.full(n, 7.0),
        "dup": income + rng.normal(0, 0.001, n),     # ~duplicate of income
        "noise1": rng.normal(0, 1, n),
        "noise2": rng.normal(0, 1, n),
    })
    lin = ((income - income.mean()) / income.std() * 2.0
           + (age - 40) / 12 * 1.2)
    base["churn"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-lin))).astype(int)
    train = base.iloc[:300].reset_index(drop=True)
    test = base.iloc[300:].reset_index(drop=True)
    KW = dict(show=False, plot=False)

    # a sklearn-free recipe exercising redundancy x2 + relevance
    recipe = [
        {"fn": "redundancy", "method": "variance"},
        {"fn": "redundancy", "method": "correlation", "threshold": 0.95},
        {"fn": "relevance", "method": "anova", "keep": 3},
    ]

    # ==================================================================
    section("1. selectpipe FIT -- chaining, combined params, immutability")
    tr_sel, combined = dx.selectpipe(train, steps=recipe, y="churn",
                                     return_params=True, **KW)
    check("variance step dropped the constant column",
          "const" not in tr_sel.columns)
    check("correlation step dropped the near-duplicate 'dup'",
          "dup" not in tr_sel.columns)
    check("relevance step kept exactly 3 features + the target",
          tr_sel.shape[1] == 4 and "churn" in tr_sel.columns)
    check("original train NOT mutated",
          train.shape[1] == 8 and "const" in train.columns)
    check("combined params function == 'selectpipe'",
          combined["function"] == "selectpipe")
    check("combined params has 3 step dicts",
          len(combined["steps"]) == 3)
    check("each step dict carries its own 'function' tag",
          [s["function"] for s in combined["steps"]]
          == ["redundancy", "redundancy", "relevance"])
    check("metadata records input/output shape + kept_final",
          combined["metadata"]["input_shape"] == [300, 8]
          and combined["metadata"]["kept_final"] == list(tr_sel.columns))
    check("combined params is JSON-serialisable",
          json.loads(json.dumps(combined))["function"] == "selectpipe")

    # ==================================================================
    section("2. selectpipe -- target injection + redundancy shielding")
    check("shared y injected into the relevance step",
          combined["steps"][2]["target"] == "churn")
    check("target 'churn' shielded from the redundancy steps "
          "(never a candidate)",
          "churn" not in combined["steps"][0]["candidates"]
          and "churn" not in combined["steps"][1]["candidates"])
    check("target survives the whole pipeline", "churn" in tr_sel.columns)

    # ==================================================================
    section("3. selectpipe APPLY -- subset held-out data, no re-scoring")
    te_sel = dx.selectpipe(test, params=combined, **KW)
    check("apply reduces test to the same column set as fit",
          list(te_sel.columns) == list(tr_sel.columns))
    check("apply leaves original test untouched",
          test.shape[1] == 8 and "dup" in test.columns)

    # ==================================================================
    section("4. ROUND-TRIP -- save -> load -> apply == identical")
    tmp = os.path.join(tempfile.gettempdir(), "dextra_selectpipe_test.json")
    tr_sel2, combined2 = dx.selectpipe(train, steps=recipe, y="churn",
                                       return_params=True, save_path=tmp,
                                       **KW)
    check("save_path wrote a JSON file", os.path.isfile(tmp))
    with open(tmp, "r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    check("on-disk artifact round-trips cleanly",
          on_disk["function"] == "selectpipe"
          and len(on_disk["steps"]) == 3)
    te_loaded = dx.selectpipe(test, load_path=tmp, **KW)
    check("load_path apply == params apply (round-trip identical)",
          te_sel.equals(te_loaded))

    # ==================================================================
    section("5. IDEMPOTENCY -- applying twice is a no-op")
    te_again = dx.selectpipe(te_sel, params=combined, **KW)
    check("apply on already-selected data is identical",
          te_sel.equals(te_again))

    # ==================================================================
    section("6. The compare ban -- selectpipe is a commit tool")
    try:
        dx.selectpipe(train, steps=[{"fn": "redundancy",
                                     "method": "compare"}], y="churn", **KW)
        check("compare step rejected", False)
    except ValueError as exc:
        check("compare step raises ValueError", "compare" in str(exc))
    try:
        dx.selectpipe(train, steps=[{"fn": "rfe", "estimator": "compare"}],
                      y="churn", **KW)
        check("rfe estimator='compare' step rejected", False)
    except ValueError:
        check("rfe estimator='compare' step raises ValueError", True)

    # ==================================================================
    section("7. Mode-conflict and malformed-input guards")
    try:
        dx.selectpipe(train, steps=recipe, params=combined, **KW)
        check("steps + params together rejected", False)
    except ValueError:
        check("steps + params together raises ValueError", True)
    try:
        dx.selectpipe(train, **KW)
        check("neither steps nor params rejected", False)
    except ValueError:
        check("neither steps nor params raises ValueError", True)
    try:
        dx.selectpipe(test, params=combined, save_path=tmp, **KW)
        check("save_path in apply mode rejected", False)
    except ValueError:
        check("save_path in apply mode raises ValueError", True)
    try:
        dx.selectpipe(test, load_path=tmp, params=combined, **KW)
        check("load_path + params together rejected", False)
    except ValueError:
        check("load_path + params together raises ValueError", True)
    try:
        dx.selectpipe(train, steps=[{"method": "variance"}], y="churn", **KW)
        check("step missing 'fn' key rejected", False)
    except ValueError:
        check("step missing 'fn' key raises ValueError", True)
    try:
        dx.selectpipe(train, steps=[{"fn": "nope"}], y="churn", **KW)
        check("unknown fn rejected", False)
    except ValueError:
        check("unknown fn raises ValueError", True)

    # ==================================================================
    section("8. Step-error context -- failures name the step")
    try:
        dx.selectpipe(train, steps=[{"fn": "relevance", "method": "anova"}],
                      **KW)   # no y anywhere -> relevance fails
        check("missing-target error rejected", False)
    except ValueError as exc:
        check("step error names 'selectpipe step'",
              "selectpipe step" in str(exc))

    # ==================================================================
    section("9. audit trail + show/plot independence")
    check("selectpipe appends a feature_selection audit entry",
          tr_sel.attrs.get("dextra_audit", [])
          and tr_sel.attrs["dextra_audit"][-1]["stage"] == "feature_selection"
          and tr_sel.attrs["dextra_audit"][-1]["function"] == "selectpipe")
    check("apply audit entry is mode='apply'",
          te_sel.attrs["dextra_audit"][-1]["mode"] == "apply")
    fig = dx.selectpipe(train, steps=recipe, y="churn", show=False,
                        plot=True, return_fig=True, return_df=False)
    check("plot=True returns a Figure object", fig is not None)
    single = dx.selectpipe(train, steps=[{"fn": "redundancy",
                                          "method": "variance"}],
                           y="churn", **KW)
    check("single-step pipeline works", "const" not in single.columns)
    check("alias selpipe is selectpipe", dx.selpipe is dx.selectpipe)

    # ==================================================================
    section("10. model-based step in a pipeline (needs scikit-learn)")
    if HAVE_SKLEARN:
        recipe_ml = [
            {"fn": "redundancy", "method": "correlation", "threshold": 0.95},
            {"fn": "importance", "method": "tree", "keep": 3},
        ]
        ml_tr, ml_p = dx.selectpipe(train, steps=recipe_ml, y="churn",
                                    return_params=True, **KW)
        check("pipeline with an importance step fits",
              ml_p["steps"][1]["function"] == "importance")
        ml_te = dx.selectpipe(test, params=ml_p, **KW)
        check("model-based pipeline applies to held-out data",
              list(ml_te.columns) == list(ml_tr.columns))
        ml_te2 = dx.selectpipe(ml_te, params=ml_p, **KW)
        check("model-based pipeline apply is idempotent",
              ml_te.equals(ml_te2))
    else:
        check("scikit-learn absent -- model-based pipeline step skipped", True)

    try:
        os.remove(tmp)
    except OSError:
        pass

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  RESULT: {ok} passed, {fail} failed")
    print("=" * 74)
    if fail:
        raise SystemExit(1)
    print("  ALL PHASE 5 STAGE 5.3 CHECKS PASSED.")


if __name__ == "__main__":
    main()
