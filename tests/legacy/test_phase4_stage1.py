"""
test_phase4_stage1.py
---------------------
Verification script for dextra Phase 4 Stage 4.1 -- transform + scale.

Run inside the project venv:
    cd "D:\\06 PythonProjects\\dextra-project"
    .\\.venv\\Scripts\\Activate.ps1
    python "C:\\Users\\manim\\OneDrive\\المستندات\\Claude\\Projects\\شرح بايثون وعمل مكتبة dextra\\test_phase4_stage1.py"

Covers: fit/apply modes, the params dict, idempotency, immutability,
leakage prevention, domain guards, column-mismatch rejection, the compare
mode, inplace= behaviour, and boxcox / yeojohnson (which need SciPy).
"""
from __future__ import annotations

import json
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
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame({
        "price": rng.lognormal(3, 1, n),           # right-skewed, positive
        "age":   rng.normal(40, 12, n).clip(1),    # ~normal, positive
        "score": rng.normal(0, 5, n),              # contains negatives
    })
    df.loc[5:9, "price"] = np.nan                  # a few missing values

    # ------------------------------------------------------------------
    section("1. transform -- fit, params dict, apply, idempotency")
    tr, p = dx.transform(df, cols=["price"], method="log",
                         return_params=True, show=False, plot=False)
    check("new column 'price_log' created", "price_log" in tr.columns)
    check("source column 'price' preserved", "price" in tr.columns)
    check("original df NOT mutated", "price_log" not in df.columns)
    check("params is JSON-serialisable",
          json.loads(json.dumps(p))["function"] == "transform")
    check("params carries a version", p.get("version") is not None)
    check("log reduced |skew|",
          abs(tr["price_log"].skew()) < abs(df["price"].skew()))
    check("NaN positions preserved",
          tr["price_log"].isna().sum() == df["price"].isna().sum())

    df_test = df.iloc[:120].copy()
    tr_te = dx.transform(df_test, params=p, show=False, plot=False)
    check("apply mode creates 'price_log'", "price_log" in tr_te.columns)
    tr_te2 = dx.transform(tr_te, params=p, show=False, plot=False)
    check("idempotent under apply (inplace=False)",
          np.allclose(tr_te["price_log"].dropna(),
                      tr_te2["price_log"].dropna()))

    # ------------------------------------------------------------------
    section("2. transform -- domain guards & column-mismatch rejection")
    try:
        dx.transform(df, cols=["score"], method="log", show=False, plot=False)
        check("log on negatives raises", False)
    except ValueError as e:
        check("log on negative values raises ValueError", "<= 0" in str(e))
    try:
        dx.transform(df_test.drop(columns=["price"]), params=p,
                     show=False, plot=False)
        check("apply on a missing column raises", False)
    except KeyError as e:
        check("apply column-mismatch raises KeyError", "price" in str(e))

    tl = dx.transform(df, cols=["price"], method="log1p",
                      show=False, plot=False)
    check("log1p produces 'price_log1p'", "price_log1p" in tl.columns)
    ts = dx.transform(df, cols=["price"], method="sqrt",
                      show=False, plot=False)
    check("sqrt produces 'price_sqrt'", "price_sqrt" in ts.columns)

    # ------------------------------------------------------------------
    section("3. transform -- boxcox & yeojohnson  (need SciPy)")
    bc, pbc = dx.transform(df, cols=["price"], method="boxcox",
                           return_params=True, show=False, plot=False)
    check("boxcox produces 'price_boxcox'", "price_boxcox" in bc.columns)
    check("boxcox stored a fitted lambda",
          "lambda" in pbc["columns"]["price"])
    bc_te = dx.transform(df_test, params=pbc, show=False, plot=False)
    check("boxcox apply re-uses the saved lambda (no re-fit)",
          "price_boxcox" in bc_te.columns)
    lam = pbc["columns"]["price"]["lambda"]
    bc_manual = dx.transform(df_test, cols=["price"], method="boxcox",
                             show=False, plot=False)
    # apply with saved lambda must differ from a fresh fit on the test slice
    check("apply (saved lambda) != fresh fit on test slice  [leakage-safe]",
          not np.allclose(bc_te["price_boxcox"].dropna(),
                          bc_manual["price_boxcox"].dropna()))

    yj = dx.transform(df, cols=["score"], method="yeojohnson",
                      show=False, plot=False)
    check("yeojohnson works on negative values",
          "score_yeojohnson" in yj.columns)

    # ------------------------------------------------------------------
    section("4. transform -- compare mode (don't decide, give choices)")
    cmp = dx.transform(df, cols=["price"], method="compare",
                       show=True, plot=False)
    check("compare writes NOTHING (df unchanged)",
          list(cmp.columns) == list(df.columns))
    try:
        dx.transform(df, cols=["price"], method="compare",
                     return_params=True, show=False, plot=False)
        check("compare + return_params raises", False)
    except ValueError:
        check("compare + return_params raises ValueError", True)

    # ------------------------------------------------------------------
    section("5. scale -- four methods, fit/apply, idempotency")
    for m in ("standard", "minmax", "robust", "maxabs"):
        sc, sp = dx.scale(df, cols=["price", "age"], method=m,
                          return_params=True, show=False, plot=False)
        check(f"{m}: new columns created",
              f"price_{m}" in sc.columns and f"age_{m}" in sc.columns)
        check(f"{m}: original df NOT mutated", f"price_{m}" not in df.columns)
        check(f"{m}: params JSON-serialisable",
              json.loads(json.dumps(sp))["method"] == m)
        sc_te = dx.scale(df.iloc[:120].copy(), params=sp,
                         show=False, plot=False)
        sc_te2 = dx.scale(sc_te, params=sp, show=False, plot=False)
        check(f"{m}: idempotent under apply",
              np.allclose(sc_te[f"price_{m}"].dropna(),
                          sc_te2[f"price_{m}"].dropna()))
    sc_std = dx.scale(df, cols=["age"], method="standard",
                      show=False, plot=False)
    check("standard: mean~0, std~1",
          abs(sc_std["age_standard"].mean()) < 1e-9
          and abs(sc_std["age_standard"].std(ddof=0) - 1) < 1e-9)
    sc_mm = dx.scale(df, cols=["age"], method="minmax",
                     show=False, plot=False)
    check("minmax: output range in [0, 1]",
          sc_mm["age_minmax"].min() >= -1e-9
          and sc_mm["age_minmax"].max() <= 1 + 1e-9)

    # ------------------------------------------------------------------
    section("6. scale -- leakage prevention (fit on train only)")
    train, test = df.iloc[:300], df.iloc[300:]
    _, p_tr = dx.scale(train, cols=["age"], method="standard",
                       return_params=True, show=False, plot=False)
    check("train mean differs from full-data mean (no statistic leak)",
          abs(p_tr["columns"]["age"]["mean"] - df["age"].mean()) > 1e-9)
    test_sc = dx.scale(test.copy(), params=p_tr, show=False, plot=False)
    check("test scaled with TRAIN params", "age_standard" in test_sc.columns)

    # ------------------------------------------------------------------
    section("7. inplace= behaviour + show/plot independence + audit")
    ip = dx.scale(df, cols=["age"], method="minmax", inplace=True,
                  show=False, plot=False)
    check("inplace=True overwrites 'age' (no extra column)",
          "age_minmax" not in ip.columns and ip["age"].max() <= 1 + 1e-9)
    check("inplace: original df still untouched", df["age"].max() > 1)
    fig = dx.scale(df, cols=["price"], method="standard", show=False,
                   plot=True, return_fig=True, return_df=False)
    check("plot=True returns a Figure", fig is not None)
    aud = dx.scale(df, cols=["price"], method="standard",
                   show=False, plot=False)
    check("audit entry appended to df.attrs",
          len(aud.attrs.get("dextra_audit", [])) >= 1)
    check("audit decision sentence is English",
          "Fitted" in aud.attrs["dextra_audit"][-1]["decision"])

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  RESULT: {ok} passed, {fail} failed")
    print("=" * 74)
    if fail:
        raise SystemExit(1)
    print("  ALL PHASE 4 STAGE 4.1 CHECKS PASSED.")


if __name__ == "__main__":
    main()
