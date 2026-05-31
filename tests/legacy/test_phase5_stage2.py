"""
test_phase5_stage2.py
---------------------
Verification script for dextra Phase 5 Stage 5.2 -- importance + rfe
(the Embedded and Wrapper families of feature selection).

Run inside the project venv:
    cd "D:\\06 PythonProjects\\dextra-project"
    .\\.venv\\Scripts\\Activate.ps1
    python "C:\\Users\\manim\\OneDrive\\المستندات\\Claude\\Projects\\شرح بايثون وعمل مكتبة dextra\\test_phase5_stage2.py"

Both functions need scikit-learn. If it is not installed the script verifies
the clean ImportError and exits; if it is installed it runs the full suite:
the fit/apply contract, the params dict (JSON-serialisable), apply = subset
only (leakage-safe), idempotency, immutability, compare mode, all methods /
estimators, task inference and override, and the guard errors.
"""
from __future__ import annotations

import json
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


def make_data():
    rng = np.random.default_rng(7)
    n = 400
    income = rng.gamma(2.0, 1000.0, n)
    age = rng.normal(40, 12, n)
    tenure = rng.normal(5, 2, n)
    noise1 = rng.normal(0, 1, n)
    noise2 = rng.normal(0, 1, n)
    lin = ((income - income.mean()) / income.std() * 2.0
           + (age - 40) / 12 * 1.0)
    prob = 1.0 / (1.0 + np.exp(-lin))
    churn = (rng.random(n) < prob).astype(int)
    spend = income * 0.4 + age * 5.0 + rng.normal(0, 200, n)
    df = pd.DataFrame({"income": income, "age": age, "tenure": tenure,
                       "noise1": noise1, "noise2": noise2,
                       "churn": churn, "spend": spend})
    train = df.iloc[:300].reset_index(drop=True)
    test = df.iloc[300:].reset_index(drop=True)
    return train, test


def main() -> None:
    print(f"dextra version: {dx.__version__}")
    print(f"scikit-learn available: {HAVE_SKLEARN}")
    check("importance + rfe exported",
          hasattr(dx, "importance") and hasattr(dx, "rfe"))
    check("alias imps exported", hasattr(dx, "imps"))

    train, test = make_data()
    KW = dict(show=False, plot=False)
    feats = ["income", "age", "tenure", "noise1", "noise2"]

    # ------------------------------------------------------------------
    if not HAVE_SKLEARN:
        section("scikit-learn NOT installed -- verifying the clean ImportError")
        for fn, kw in [("importance", dict(y="churn", method="tree")),
                       ("rfe", dict(y="churn", estimator="tree", keep=2))]:
            try:
                getattr(dx, fn)(train, cols=feats, **kw, **KW)
                check(f"{fn} without sklearn rejected", False)
            except ImportError as exc:
                check(f"{fn} raises a clean ImportError naming scikit-learn",
                      "scikit-learn" in str(exc))
        print("\n" + "=" * 74)
        print(f"  RESULT: {ok} passed, {fail} failed  "
              f"(install scikit-learn for the full suite)")
        print("=" * 74)
        if fail:
            raise SystemExit(1)
        print("  STAGE 5.2 IMPORT-GUARD CHECKS PASSED.")
        return

    # ==================================================================
    section("1. importance -- tree: rank features by a random forest")
    i_tree, p_tree = dx.importance(train, y="churn", cols=feats,
                                   method="tree", keep=3,
                                   return_params=True, **KW)
    check("tree kept exactly keep=3 features", len(p_tree["kept"]) == 3)
    check("tree ranks 'income' above pure noise",
          p_tree["scores"]["income"] > p_tree["scores"]["noise1"]
          and p_tree["scores"]["income"] > p_tree["scores"]["noise2"])
    check("tree keeps the strong signal 'income'", "income" in p_tree["kept"])
    check("original train NOT mutated", "noise1" in train.columns)
    check("target 'churn' auto-protected", "churn" not in p_tree["candidates"]
          and "churn" in i_tree.columns)
    check("params is JSON-serialisable",
          json.loads(json.dumps(p_tree))["function"] == "importance")
    check("params lists kept / dropped / scores",
          all(k in p_tree for k in ("kept", "dropped", "scores")))

    # ==================================================================
    section("2. importance -- l1, linear, regression target")
    i_l1, p_l1 = dx.importance(train, y="churn", cols=feats, method="l1",
                               keep=3, return_params=True, **KW)
    check("l1 ranks 'income' above noise",
          p_l1["scores"]["income"] > p_l1["scores"]["noise1"])
    i_lin, p_lin = dx.importance(train, y="churn", cols=feats, method="linear",
                                 keep=3, return_params=True, **KW)
    check("linear ranks 'income' above noise",
          p_lin["scores"]["income"] > p_lin["scores"]["noise1"])
    i_reg, p_reg = dx.importance(train, y="spend", cols=feats, method="tree",
                                 keep=2, return_params=True, **KW)
    check("regression target inferred", p_reg["target_kind"] == "regression")
    check("tree regression ranks 'income' top for 'spend'",
          p_reg["scores"]["income"] >= max(p_reg["scores"][c]
                                           for c in ["noise1", "noise2"]))
    i_task = dx.importance(train, y="churn", cols=feats, method="tree",
                           keep=2, task="classification", **KW)
    check("explicit task='classification' works",
          isinstance(i_task, pd.DataFrame))

    # ==================================================================
    section("3. importance -- fit/apply, idempotency, immutability")
    i_te = dx.importance(test, params=p_tree, **KW)
    check("apply subsets test to the fitted kept set",
          all(c not in i_te.columns for c in p_tree["dropped"]))
    check("apply keeps every fitted kept column",
          all(c in i_te.columns for c in p_tree["kept"]))
    check("apply needs no y (subset-only, leakage-safe)",
          "churn" in i_te.columns)
    check("apply leaves original test untouched", "noise1" in test.columns)
    i_te2 = dx.importance(i_te, params=p_tree, **KW)
    check("importance apply is idempotent",
          list(i_te.columns) == list(i_te2.columns))
    try:
        dx.importance(test.drop(columns=p_tree["kept"][:1]), params=p_tree,
                      **KW)
        check("apply column-mismatch rejected", False)
    except KeyError:
        check("apply column-mismatch raises KeyError", True)

    # ==================================================================
    section("4. importance -- compare + guard errors")
    i_cmp = dx.importance(train, y="churn", cols=feats, method="compare", **KW)
    check("compare drops nothing",
          all(c in i_cmp.columns for c in feats))
    try:
        dx.importance(train, y="churn", cols=feats, method="compare",
                      return_params=True, **KW)
        check("compare + return_params rejected", False)
    except ValueError:
        check("compare + return_params raises ValueError", True)
    try:
        dx.importance(train, cols=feats, method="tree", **KW)
        check("importance without y rejected", False)
    except ValueError:
        check("importance without y raises ValueError", True)
    try:
        dx.importance(train, y="churn", cols=feats, method="bogus", **KW)
        check("bad method rejected", False)
    except ValueError:
        check("bad method raises ValueError", True)
    try:
        dx.importance(train, y="churn", cols=feats, method="tree", keep=0,
                      **KW)
        check("keep < 1 rejected", False)
    except ValueError:
        check("keep < 1 raises ValueError", True)

    # ==================================================================
    section("5. rfe -- recursive feature elimination, tree + linear")
    r_tree, pr_tree = dx.rfe(train, y="churn", cols=feats, keep=2,
                             estimator="tree", return_params=True, **KW)
    check("rfe tree keeps exactly keep=2 features",
          len(pr_tree["kept"]) == 2)
    check("rfe tree keeps the strong signal 'income'",
          "income" in pr_tree["kept"])
    check("rfe params records the per-feature ranking",
          "rfe_ranking" in pr_tree["metadata"])
    check("rfe params is JSON-serialisable",
          json.loads(json.dumps(pr_tree))["function"] == "rfe")
    r_lin, pr_lin = dx.rfe(train, y="churn", cols=feats, keep=3,
                           estimator="linear", step=1,
                           return_params=True, **KW)
    check("rfe linear keeps keep=3 features", len(pr_lin["kept"]) == 3)
    r_big = dx.rfe(train, y="churn", cols=feats, keep=99, estimator="tree",
                   **KW)
    check("rfe with keep >= n_candidates drops nothing",
          all(c in r_big.columns for c in feats))

    # ==================================================================
    section("6. rfe -- fit/apply, idempotency, immutability")
    r_te = dx.rfe(test, params=pr_tree, **KW)
    check("rfe apply subsets test to the kept set",
          all(c not in r_te.columns for c in pr_tree["dropped"])
          and all(c in r_te.columns for c in pr_tree["kept"]))
    check("rfe apply leaves original test untouched",
          "noise1" in test.columns)
    r_te2 = dx.rfe(r_te, params=pr_tree, **KW)
    check("rfe apply is idempotent",
          list(r_te.columns) == list(r_te2.columns))
    try:
        dx.rfe(test.drop(columns=pr_tree["kept"][:1]), params=pr_tree, **KW)
        check("rfe apply column-mismatch rejected", False)
    except KeyError:
        check("rfe apply column-mismatch raises KeyError", True)

    # ==================================================================
    section("7. rfe -- compare + guard errors")
    r_cmp = dx.rfe(train, y="churn", cols=feats, estimator="compare",
                   keep=2, **KW)
    check("rfe compare drops nothing",
          all(c in r_cmp.columns for c in feats))
    try:
        dx.rfe(train, y="churn", cols=feats, estimator="compare",
               return_params=True, **KW)
        check("rfe compare + return_params rejected", False)
    except ValueError:
        check("rfe compare + return_params raises ValueError", True)
    try:
        dx.rfe(train, y="churn", cols=feats, estimator="bogus", **KW)
        check("rfe bad estimator rejected", False)
    except ValueError:
        check("rfe bad estimator raises ValueError", True)
    try:
        dx.rfe(train, y="churn", cols=feats, keep=0, **KW)
        check("rfe keep < 1 rejected", False)
    except ValueError:
        check("rfe keep < 1 raises ValueError", True)
    try:
        dx.rfe(train, y="churn", cols=feats, keep=2, step=0, **KW)
        check("rfe step < 1 rejected", False)
    except ValueError:
        check("rfe step < 1 raises ValueError", True)

    # ==================================================================
    section("8. audit trail + show/plot independence")
    check("importance appends a feature_selection audit entry",
          i_tree.attrs.get("dextra_audit", [])
          and i_tree.attrs["dextra_audit"][-1]["stage"] == "feature_selection")
    check("rfe apply audit entry is mode='apply'",
          r_te.attrs["dextra_audit"][-1]["mode"] == "apply")
    fig = dx.importance(train, y="churn", cols=feats, method="tree", keep=3,
                        show=False, plot=True, return_fig=True,
                        return_df=False)
    check("importance plot=True returns a Figure", fig is not None)
    fig2 = dx.rfe(train, y="churn", cols=feats, keep=2, show=False, plot=True,
                  return_fig=True, return_df=False)
    check("rfe plot=True returns a Figure", fig2 is not None)
    check("alias imps is importance", dx.imps is dx.importance)

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  RESULT: {ok} passed, {fail} failed")
    print("=" * 74)
    if fail:
        raise SystemExit(1)
    print("  ALL PHASE 5 STAGE 5.2 CHECKS PASSED.")


if __name__ == "__main__":
    main()
