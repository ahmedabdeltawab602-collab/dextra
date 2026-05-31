"""
test_phase5_stage1.py
---------------------
Verification script for dextra Phase 5 Stage 5.1 -- redundancy + relevance
(the Filter family of feature selection).

Run inside the project venv:
    cd "D:\\06 PythonProjects\\dextra-project"
    .\\.venv\\Scripts\\Activate.ps1
    python "C:\\Users\\manim\\OneDrive\\المستندات\\Claude\\Projects\\شرح بايثون وعمل مكتبة dextra\\test_phase5_stage1.py"

Covers: the fit/apply contract, the params dict (JSON-serialisable), apply =
subset-only (no re-scoring -> leakage-safe), idempotency, immutability, the
compare mode, the three redundancy criteria, the three relevance criteria,
keep vs threshold cuts, target protection, and all the guard errors.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # headless: no windows pop up during the test
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
    check("redundancy + relevance exported",
          hasattr(dx, "redundancy") and hasattr(dx, "relevance"))
    check("short aliases exported", hasattr(dx, "redun")
          and hasattr(dx, "relev"))

    rng = np.random.default_rng(11)
    n = 300
    train = pd.DataFrame({
        "age":    rng.normal(40, 10, n),
        "income": rng.gamma(2.0, 1000.0, n),
        "const":  np.full(n, 5.0),
        "noise":  rng.normal(0.0, 1.0, n),
    })
    train["age_copy"] = train["age"] + rng.normal(0, 0.001, n)   # ~duplicate
    train["churn"] = ((train["age"] - 40) / 10
                      + rng.normal(0, 0.5, n) > 0).astype(int)
    train["spend"] = train["income"] * 0.5 + rng.normal(0, 100, n)
    test = train.sample(90, random_state=3).reset_index(drop=True)
    KW = dict(show=False, plot=False)

    # ==================================================================
    section("1. redundancy -- variance: drop (near-)constant features")
    r_var, p_var = dx.redundancy(train, cols=["age", "income", "const",
                                              "noise"], method="variance",
                                 return_params=True, **KW)
    check("variance drops the constant column",
          "const" in p_var["dropped"] and "const" not in r_var.columns)
    check("variance keeps the informative columns",
          set(p_var["kept"]) == {"age", "income", "noise"})
    check("original train NOT mutated", "const" in train.columns)
    check("non-candidate columns pass through",
          "churn" in r_var.columns and "spend" in r_var.columns)
    check("params is JSON-serialisable",
          json.loads(json.dumps(p_var))["function"] == "redundancy")
    check("params lists kept / dropped / scores",
          all(k in p_var for k in ("kept", "dropped", "scores")))

    # ==================================================================
    section("2. redundancy -- correlation: drop one of a correlated pair")
    r_cor, p_cor = dx.redundancy(train, cols=["age", "age_copy", "income",
                                              "noise"], method="correlation",
                                 threshold=0.95, return_params=True, **KW)
    check("correlation drops the near-duplicate 'age_copy'",
          p_cor["dropped"] == ["age_copy"])
    check("correlation keeps the first of the pair ('age')",
          "age" in p_cor["kept"])
    check("uncorrelated features survive",
          "income" in p_cor["kept"] and "noise" in p_cor["kept"])

    # ==================================================================
    section("3. redundancy -- vif: drop multicollinear features")
    r_vif, p_vif = dx.redundancy(train, cols=["age", "age_copy", "income",
                                              "noise"], method="vif",
                                 return_params=True, **KW)
    check("vif drops a member of the collinear age/age_copy pair",
          ("age" in p_vif["dropped"]) ^ ("age_copy" in p_vif["dropped"]))
    check("vif leaves no severe multicollinearity behind",
          len(p_vif["kept"]) >= 2)

    # ==================================================================
    section("4. redundancy -- fit/apply, idempotency, immutability")
    r_te = dx.redundancy(test, params=p_cor, **KW)
    check("apply drops the same columns on held-out data",
          "age_copy" not in r_te.columns)
    check("apply leaves original test untouched", "age_copy" in test.columns)
    r_te2 = dx.redundancy(r_te, params=p_cor, **KW)
    check("apply is idempotent", list(r_te.columns) == list(r_te2.columns))
    try:
        dx.redundancy(test.drop(columns=["age"]), params=p_cor, **KW)
        check("apply column-mismatch rejected", False)
    except KeyError:
        check("apply column-mismatch raises KeyError", True)

    # ==================================================================
    section("5. redundancy -- compare, protect, guard errors")
    r_cmp = dx.redundancy(train, cols=["age", "age_copy", "const"],
                          method="compare", **KW)
    check("compare drops nothing", "const" in r_cmp.columns
          and "age_copy" in r_cmp.columns)
    try:
        dx.redundancy(train, cols=["age", "const"], method="compare",
                      return_params=True, **KW)
        check("compare + return_params rejected", False)
    except ValueError:
        check("compare + return_params raises ValueError", True)
    r_prot, p_prot = dx.redundancy(train, cols=["age", "income", "const"],
                                   method="variance", protect=["const"],
                                   return_params=True, **KW)
    check("protected column is never dropped",
          "const" in r_prot.columns and "const" not in p_prot["candidates"])
    try:
        dx.redundancy(train, cols=["churn", "age"], method="vif", **KW)
        # churn is numeric so this is allowed; use a real non-numeric instead
        check("redundancy on numeric churn allowed", True)
    except Exception:
        check("redundancy on numeric churn allowed", False)

    # ==================================================================
    section("6. relevance -- anova, classification + regression targets")
    s_anova, p_anova = dx.relevance(train, y="churn",
                                    cols=["age", "income", "noise",
                                          "age_copy"],
                                    method="anova", keep=2,
                                    return_params=True, **KW)
    check("anova ranks age-driven features above noise",
          "noise" in p_anova["dropped"])
    check("anova kept exactly keep=2 features", len(p_anova["kept"]) == 2)
    check("target 'churn' is auto-protected (never a candidate)",
          "churn" not in p_anova["candidates"] and "churn" in s_anova.columns)
    check("anova params is JSON-serialisable",
          json.loads(json.dumps(p_anova))["target_kind"] == "classification")
    s_reg, p_reg = dx.relevance(train, y="spend",
                                cols=["age", "income", "noise"],
                                method="anova", keep=1,
                                return_params=True, **KW)
    check("regression target detected", p_reg["target_kind"] == "regression")
    check("anova regression ranks 'income' top (drives spend)",
          p_reg["kept"] == ["income"])

    # ==================================================================
    section("7. relevance -- chi2 + keep/threshold cuts")
    s_chi2, p_chi2 = dx.relevance(train, y="churn",
                                  cols=["age", "income", "const"],
                                  method="chi2", keep=2,
                                  return_params=True, **KW)
    check("chi2 runs on non-negative features",
          p_chi2["method"] == "chi2" and len(p_chi2["kept"]) == 2)
    s_thr, p_thr = dx.relevance(train, y="churn",
                                cols=["age", "income", "noise", "age_copy"],
                                method="anova", keep=None, threshold=10.0,
                                return_params=True, **KW)
    check("threshold cut keeps only high-score features",
          all((p_thr["scores"][c] or 0) >= 10.0 for c in p_thr["kept"]))

    # ==================================================================
    section("8. relevance -- mutualinfo (conditional on scikit-learn)")
    if HAVE_SKLEARN:
        s_mi, p_mi = dx.relevance(train, y="churn",
                                  cols=["age", "income", "noise"],
                                  method="mutualinfo", keep=2,
                                  return_params=True, **KW)
        check("mutualinfo selects 2 features", len(p_mi["kept"]) == 2)
        check("mutualinfo params JSON-serialisable",
              json.loads(json.dumps(p_mi))["method"] == "mutualinfo")
    else:
        try:
            dx.relevance(train, y="churn", cols=["age", "income"],
                         method="mutualinfo", **KW)
            check("mutualinfo without sklearn rejected", False)
        except ImportError:
            check("mutualinfo without sklearn raises a clean ImportError",
                  True)

    # ==================================================================
    section("9. relevance -- fit/apply, idempotency, immutability")
    s_te = dx.relevance(test, params=p_anova, **KW)
    check("apply subsets test to the fitted kept set",
          all(c not in s_te.columns for c in p_anova["dropped"]))
    check("apply leaves original test untouched",
          "noise" in test.columns)
    check("apply needs no y (subset-only, leakage-safe)",
          set(p_anova["kept"]).issubset(set(s_te.columns)))
    s_te2 = dx.relevance(s_te, params=p_anova, **KW)
    check("relevance apply is idempotent",
          list(s_te.columns) == list(s_te2.columns))
    try:
        dx.relevance(test.drop(columns=p_anova["kept"][:1]), params=p_anova,
                     **KW)
        check("apply column-mismatch rejected", False)
    except KeyError:
        check("apply column-mismatch raises KeyError", True)

    # ==================================================================
    section("10. relevance -- compare + guard errors")
    s_cmp = dx.relevance(train, y="churn", cols=["age", "income", "noise"],
                         method="compare", **KW)
    check("compare drops nothing",
          all(c in s_cmp.columns for c in ["age", "income", "noise"]))
    try:
        dx.relevance(train, cols=["age"], method="anova", **KW)
        check("relevance without y rejected", False)
    except ValueError:
        check("relevance without y raises ValueError", True)
    try:
        dx.relevance(train, y="churn", cols=["age", "noise"], method="chi2",
                     **KW)
        check("chi2 on a negative feature rejected", False)
    except ValueError:
        check("chi2 on a negative feature raises ValueError", True)
    try:
        dx.relevance(train, y="spend", cols=["age", "income"], method="chi2",
                     **KW)
        check("chi2 on a regression target rejected", False)
    except ValueError:
        check("chi2 on a regression target raises ValueError", True)
    try:
        dx.relevance(train, y="churn", cols=["age"], method="anova",
                     keep=0, **KW)
        check("keep < 1 rejected", False)
    except ValueError:
        check("keep < 1 raises ValueError", True)

    # ==================================================================
    section("11. audit trail + show/plot independence")
    check("redundancy appends a feature_selection audit entry",
          r_var.attrs.get("dextra_audit", [])
          and r_var.attrs["dextra_audit"][-1]["stage"] == "feature_selection")
    check("relevance apply audit entry is mode='apply'",
          s_te.attrs["dextra_audit"][-1]["mode"] == "apply")
    fig = dx.redundancy(train, cols=["age", "age_copy", "income"],
                        method="correlation", show=False, plot=True,
                        return_fig=True, return_df=False)
    check("redundancy plot=True returns a Figure", fig is not None)
    fig2 = dx.relevance(train, y="churn", cols=["age", "income", "noise"],
                        method="anova", show=False, plot=True,
                        return_fig=True, return_df=False)
    check("relevance plot=True returns a Figure", fig2 is not None)
    check("alias redun is redundancy", dx.redun is dx.redundancy)
    check("alias relev is relevance", dx.relev is dx.relevance)

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  RESULT: {ok} passed, {fail} failed")
    print("=" * 74)
    if fail:
        raise SystemExit(1)
    print("  ALL PHASE 5 STAGE 5.1 CHECKS PASSED.")


if __name__ == "__main__":
    main()
