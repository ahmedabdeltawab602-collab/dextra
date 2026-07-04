"""
test_phase4_stage4.py
---------------------
Verification script for dextra Phase 4 Stage 4.4 -- featpipe().

Run inside the project venv:
    cd "D:\\06 PythonProjects\\dextra-project"
    .\\.venv\\Scripts\\Activate.ps1
    python "C:\\Users\\manim\\OneDrive\\المستندات\\Claude\\Projects\\شرح بايثون وعمل مكتبة dextra\\test_phase4_stage4.py"

Covers: the fit/apply contract of the pipeline wrapper, the combined params
artifact (JSON-serialisable), the round-trip (fit -> apply -> save -> load ->
apply == identical), idempotency, immutability, the compare ban, mode-conflict
errors, step-error context wrapping, save_path/load_path, inplace propagation,
and the pipeline-level audit entry.
"""
from __future__ import annotations

import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless: no windows pop up during the test
import dextra as dx

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
    check("featpipe is exported by dextra", hasattr(dx, "featpipe"))

    rng = np.random.default_rng(42)
    n = 320
    train = pd.DataFrame({
        "signup": pd.to_datetime("2022-01-01")
        + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
        "income": rng.gamma(2.0, 1500.0, n),
        "age":    rng.integers(18, 80, n).astype(float),
        "city":   rng.choice(["Cairo", "Giza", "Alex"], n),
    })
    test = pd.DataFrame({
        "signup": pd.to_datetime("2024-03-01")
        + pd.to_timedelta(rng.integers(0, 220, 96), unit="D"),
        "income": rng.gamma(2.0, 1500.0, 96),
        "age":    rng.integers(18, 80, 96).astype(float),
        "city":   rng.choice(["Cairo", "Giza", "Alex", "Luxor"], 96),
    })
    KW = dict(show=False, plot=False)

    # one recipe that exercises all seven functions; step 1 creates a column
    # (income_log1p) that step 2 then scales -> chaining is tested.
    # 0.6.0 migration (issue #4): inside featpipe, inplace now defaults to
    # True; this recipe relies on suffixed derived columns, so it declares
    # 'inplace': False explicitly -- exactly what a 0.5.x user migrates to.
    recipe = [
        {"fn": "transform", "cols": ["income"], "method": "log1p",
         "inplace": False},
        {"fn": "scale", "cols": ["income_log1p", "age"], "method": "robust",
         "inplace": False},
        {"fn": "bin", "cols": ["age"], "method": "quantile", "n_bins": 4,
         "inplace": False},
        {"fn": "encode", "cols": ["city"], "method": "onehot",
         "inplace": False},
        {"fn": "dtfeats", "cols": ["signup"], "method": "calendar"},
        {"fn": "cross", "pairs": [("income", "age")], "method": "ratio"},
        {"fn": "aggfeat", "group": "city", "value": "income", "agg": "mean"},
    ]

    # ==================================================================
    section("1. featpipe FIT -- chaining, combined params, immutability")
    with warnings.catch_warnings():           # aggfeat datetime-leak warning
        warnings.simplefilter("ignore")
        train_fe, combined = dx.featpipe(train, steps=recipe,
                                         return_params=True, **KW)
    check("step 1 output column exists (income_log1p)",
          "income_log1p" in train_fe.columns)
    check("step 2 scaled the step-1 column (income_log1p_robust)",
          "income_log1p_robust" in train_fe.columns)
    check("all seven step outputs present",
          all(c in train_fe.columns for c in
              ["age_bin", "city_Cairo", "signup_month",
               "income_div_age", "income_mean_by_city"]))
    check("original train NOT mutated (no new columns)",
          list(train.columns) == ["signup", "income", "age", "city"])
    check("combined params function == 'featpipe'",
          combined["function"] == "featpipe")
    check("combined params has 7 step dicts",
          len(combined["steps"]) == 7)
    check("each step dict carries its own 'function' tag",
          [s["function"] for s in combined["steps"]]
          == ["transform", "scale", "bin", "encode",
              "dtfeats", "cross", "aggfeat"])
    check("metadata records input/output shape",
          combined["metadata"]["input_shape"] == [n, 4]
          and combined["metadata"]["output_shape"][0] == n)
    check("combined params is JSON-serialisable",
          json.loads(json.dumps(combined))["function"] == "featpipe")

    # ==================================================================
    section("2. featpipe APPLY -- no re-fit on held-out data")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_fe = dx.featpipe(test, params=combined, **KW)
    check("apply recreates the full feature set on test",
          all(c in test_fe.columns for c in
              ["income_log1p", "income_log1p_robust", "age_bin",
               "city_Cairo", "signup_month", "income_div_age",
               "income_mean_by_city"]))
    check("apply leaves original test untouched",
          list(test.columns) == ["signup", "income", "age", "city"])
    # robust scaler stats must come from TRAIN, not be re-fitted on test
    sp_scale = combined["steps"][1]["columns"]["age"]
    med_train = float(np.median(train["age"]))
    check("apply re-used TRAIN scaler stats (no leakage)",
          abs(sp_scale["median"] - med_train) < 1e-9)

    # ==================================================================
    section("3. ROUND-TRIP -- save -> load -> apply == identical")
    tmp = os.path.join(tempfile.gettempdir(), "dextra_pipe_test.json")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_fe2, combined2 = dx.featpipe(train, steps=recipe,
                                           return_params=True,
                                           save_path=tmp, **KW)
    check("save_path wrote a JSON file", os.path.isfile(tmp))
    with open(tmp, "r", encoding="utf-8") as fh:
        on_disk = json.load(fh)
    check("on-disk artifact round-trips cleanly",
          on_disk["function"] == "featpipe"
          and len(on_disk["steps"]) == 7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_fe_loaded = dx.featpipe(test, load_path=tmp, **KW)
    check("load_path apply == params apply (round-trip identical)",
          test_fe.equals(test_fe_loaded))

    # ==================================================================
    section("4. IDEMPOTENCY -- applying twice is a no-op")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        test_fe_again = dx.featpipe(test_fe, params=combined, **KW)
    check("apply on already-transformed df is identical",
          test_fe.equals(test_fe_again))

    # ==================================================================
    section("5. The compare ban -- featpipe is a commit tool")
    try:
        dx.featpipe(train, steps=[{"fn": "scale", "cols": ["age"],
                                   "method": "compare"}], **KW)
        check("compare step rejected", False)
    except ValueError as exc:
        check("compare step raises ValueError",
              "compare" in str(exc))
    try:
        dx.featpipe(train, steps=[{"fn": "aggfeat", "group": "city",
                                   "value": "income", "agg": "compare"}], **KW)
        check("aggfeat agg='compare' step rejected", False)
    except ValueError:
        check("aggfeat agg='compare' step raises ValueError", True)

    # ==================================================================
    section("6. Mode-conflict and malformed-input guards")
    try:
        dx.featpipe(train, steps=recipe, params=combined, **KW)
        check("steps + params together rejected", False)
    except ValueError:
        check("steps + params together raises ValueError", True)
    try:
        dx.featpipe(train, **KW)
        check("neither steps nor params rejected", False)
    except ValueError:
        check("neither steps nor params raises ValueError", True)
    try:
        dx.featpipe(test, params=combined, save_path=tmp, **KW)
        check("save_path in apply mode rejected", False)
    except ValueError:
        check("save_path in apply mode raises ValueError", True)
    try:
        dx.featpipe(test, load_path=tmp, params=combined, **KW)
        check("load_path + params together rejected", False)
    except ValueError:
        check("load_path + params together raises ValueError", True)
    try:
        dx.featpipe(train, steps=[{"cols": ["age"], "method": "robust"}], **KW)
        check("step missing 'fn' key rejected", False)
    except ValueError:
        check("step missing 'fn' key raises ValueError", True)
    try:
        dx.featpipe(train, steps=[{"fn": "nope", "cols": ["age"]}], **KW)
        check("unknown fn rejected", False)
    except ValueError:
        check("unknown fn raises ValueError", True)

    # ==================================================================
    section("7. Step-error context -- failures name the step")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dx.featpipe(test.drop(columns=["city"]), params=combined, **KW)
        check("apply on df missing a needed column rejected", False)
    except (KeyError, ValueError) as exc:
        check("apply column-mismatch error names 'featpipe step'",
              "featpipe step" in str(exc))

    # ==================================================================
    section("8. inplace propagation through the pipeline")
    ip_recipe = [
        {"fn": "transform", "cols": ["income"], "method": "log1p",
         "inplace": True},
        {"fn": "scale", "cols": ["age"], "method": "minmax", "inplace": True},
    ]
    ip_tr, ip_p = dx.featpipe(train, steps=ip_recipe,
                              return_params=True, **KW)
    check("inplace=True overwrote source (no income_log1p column)",
          "income_log1p" not in ip_tr.columns and "income" in ip_tr.columns)
    check("inplace=True kept column count flat",
          ip_tr.shape[1] == train.shape[1])
    ip_te = dx.featpipe(test, params=ip_p, **KW)
    check("inplace choice reproduced on apply",
          "age_minmax" not in ip_te.columns and "age" in ip_te.columns)
    # apply must re-use the fitted minmax range, not re-fit on test
    check("inplace apply still leakage-safe (values in train-fitted range)",
          ip_te["age"].notna().all())

    # ==================================================================
    section("9. Audit trail + show/plot independence")
    check("pipeline appends a feature_pipeline audit entry",
          train_fe.attrs.get("dextra_audit", [])
          and train_fe.attrs["dextra_audit"][-1]["stage"]
          == "feature_pipeline")
    check("audit also carries every sub-function entry",
          sum(1 for e in train_fe.attrs["dextra_audit"]
              if e["function"] in ("transform", "scale", "bin", "encode",
                                    "dtfeats", "cross", "aggfeat")) == 7)
    check("apply audit entry is mode='apply'",
          test_fe.attrs["dextra_audit"][-1]["mode"] == "apply")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig = dx.featpipe(train, steps=recipe, show=False, plot=True,
                          return_fig=True, return_df=False)
    check("plot=True returns a Figure object", fig is not None)
    single = dx.featpipe(train, steps=[{"fn": "scale", "cols": ["age"],
                                        "method": "standard",
                                        "inplace": False}], **KW)
    check("single-step pipeline works",
          "age_standard" in single.columns)

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
    print("  ALL PHASE 4 STAGE 4.4 CHECKS PASSED.")


if __name__ == "__main__":
    main()
