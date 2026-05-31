"""
test_phase4_stage3.py
---------------------
Verification script for dextra Phase 4 Stage 4.3 -- dtfeats + cross + aggfeat.

Run inside the project venv:
    cd "D:\\06 PythonProjects\\dextra-project"
    .\\.venv\\Scripts\\Activate.ps1
    python "C:\\Users\\manim\\OneDrive\\المستندات\\Claude\\Projects\\شرح بايثون وعمل مكتبة dextra\\test_phase4_stage3.py"

Covers: fit/apply modes, the params dict (JSON-serialisable), idempotency,
immutability, the compare mode, the cyclical sin/cos identity, division-by-
zero safety, the aggfeat datetime-leak warning, and -- most importantly --
the as_of expanding window that prevents temporal leakage.
"""
from __future__ import annotations

import json
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
    rng = np.random.default_rng(7)
    n = 360

    train = pd.DataFrame({
        "signup": pd.to_datetime("2022-01-01")
        + pd.to_timedelta(rng.integers(0, 900, n), unit="D")
        + pd.to_timedelta(rng.integers(0, 24, n), unit="h"),
        "price": rng.gamma(2.0, 30.0, n),
        "area":  rng.gamma(3.0, 20.0, n),
        "city":  rng.choice(["Cairo", "Giza", "Alex"], n),
    })
    test = pd.DataFrame({
        "signup": pd.to_datetime("2024-06-01")
        + pd.to_timedelta(rng.integers(0, 200, 110), unit="D"),
        "price": rng.gamma(2.0, 30.0, 110),
        "area":  rng.gamma(3.0, 20.0, 110),
        "city":  rng.choice(["Cairo", "Giza", "Alex", "Luxor"], 110),
    })
    KW = dict(show=False, plot=False)

    # ==================================================================
    section("1. dtfeats -- methods, params, fit/apply, idempotency")
    d_cal = dx.dtfeats(train, cols=["signup"], method="calendar", **KW)
    check("calendar adds signup_month + signup_is_weekend",
          "signup_month" in d_cal.columns
          and "signup_is_weekend" in d_cal.columns)
    check("original train NOT mutated", "signup_month" not in train.columns)

    d_cyc, p_cyc = dx.dtfeats(train, cols=["signup"], method="cyclical",
                              return_params=True, **KW)
    check("cyclical adds signup_month_sin / _cos",
          "signup_month_sin" in d_cyc.columns
          and "signup_month_cos" in d_cyc.columns)
    s2c2 = d_cyc["signup_month_sin"] ** 2 + d_cyc["signup_month_cos"] ** 2
    check("cyclical identity holds: sin^2 + cos^2 == 1",
          np.allclose(s2c2.dropna(), 1.0))
    check("dtfeats params is JSON-serialisable",
          json.loads(json.dumps(p_cyc))["function"] == "dtfeats")

    d_tr, p = dx.dtfeats(train, cols=["signup"], method="both",
                         return_params=True, **KW)
    d_te = dx.dtfeats(test, params=p, **KW)
    check("apply recreates the identical feature set",
          set(p["columns"]["signup"]["new_cols"]) <= set(d_te.columns))
    d_te2 = dx.dtfeats(d_te, params=p, **KW)
    check("dtfeats apply idempotent",
          d_te["signup_month"].equals(d_te2["signup_month"]))

    nat = pd.DataFrame({"signup": [pd.Timestamp("2023-01-01"), pd.NaT,
                                   pd.Timestamp("2023-06-15")]})
    d_nat = dx.dtfeats(nat, cols=["signup"], method="calendar", **KW)
    check("NaT -> NaN in extracted features",
          d_nat["signup_month"].isna().sum() == 1)
    d_drop = dx.dtfeats(train, cols=["signup"], method="calendar",
                        drop_original=True, **KW)
    check("drop_original removes the source datetime column",
          "signup" not in d_drop.columns)
    try:
        dx.dtfeats(train, cols=["signup"], method="compare",
                   return_params=True, **KW)
        check("dtfeats compare + return_params raises", False)
    except ValueError:
        check("dtfeats compare + return_params raises ValueError", True)
    try:
        dx.dtfeats(test.drop(columns=["signup"]), params=p, **KW)
        check("dtfeats apply column-mismatch raises", False)
    except KeyError:
        check("dtfeats apply column-mismatch raises KeyError", True)

    # ==================================================================
    section("2. cross -- methods, division safety, fit/apply")
    c_ratio, p_ratio = dx.cross(train, pairs=[("price", "area")],
                                method="ratio", return_params=True, **KW)
    check("ratio adds price_div_area", "price_div_area" in c_ratio.columns)
    check("cross params is JSON-serialisable",
          json.loads(json.dumps(p_ratio))["method"] == "ratio")
    check("original train NOT mutated",
          "price_div_area" not in train.columns)

    c_prod = dx.cross(train, pairs=[("price", "area")], method="product", **KW)
    check("product adds price_x_area", "price_x_area" in c_prod.columns)
    c_diff = dx.cross(train, pairs=[("price", "area")], method="diff", **KW)
    check("diff adds price_minus_area",
          "price_minus_area" in c_diff.columns)

    zdf = pd.DataFrame({"a": [10.0, 20.0, 30.0], "b": [2.0, 0.0, 5.0]})
    c_z = dx.cross(zdf, pairs=[("a", "b")], method="ratio", **KW)
    check("ratio: division by zero -> NaN, never Inf",
          c_z["a_div_b"].isna().sum() == 1
          and np.isfinite(c_z["a_div_b"].dropna()).all())

    c_poly = dx.cross(train, cols=["price", "area"], method="polynomial",
                      degree=3, **KW)
    check("polynomial adds power terms (pow2, pow3)",
          "price_pow2" in c_poly.columns and "price_pow3" in c_poly.columns)
    check("polynomial adds the pairwise product",
          "price_x_area" in c_poly.columns)

    c_tr, pc = dx.cross(train, cols=["price", "area"], method="product",
                        return_params=True, **KW)
    c_te = dx.cross(test, params=pc, **KW)
    check("cross apply recreates the recipe",
          "price_x_area" in c_te.columns)
    c_te2 = dx.cross(c_te, params=pc, **KW)
    check("cross apply idempotent",
          np.allclose(c_te["price_x_area"].dropna(),
                      c_te2["price_x_area"].dropna()))
    try:
        dx.cross(test.drop(columns=["price"]), params=pc, **KW)
        check("cross apply missing-input raises", False)
    except KeyError:
        check("cross apply missing-input raises KeyError", True)
    try:
        dx.cross(train, cols=["price", "area"], method="compare",
                 return_params=True, **KW)
        check("cross compare + return_params raises", False)
    except ValueError:
        check("cross compare + return_params raises ValueError", True)
    c_cmp = dx.cross(train, cols=["price", "area"], method="compare", **KW)
    check("cross compare writes nothing",
          "price_div_area" not in c_cmp.columns)

    # ==================================================================
    section("3. aggfeat -- static fit / apply")
    a_tr, pa = dx.aggfeat(train, group="city", value="price", agg="mean",
                          return_params=True, **KW)
    check("static adds price_mean_by_city",
          "price_mean_by_city" in a_tr.columns)
    check("aggfeat params is JSON-serialisable",
          json.loads(json.dumps(pa))["function"] == "aggfeat")
    naive = train.groupby("city")["price"].mean().to_dict()
    cp = pa["columns"]["price_mean_by_city"]
    check("static mapping == per-group means",
          all(abs(cp["mapping"][k] - naive[k]) < 1e-9 for k in naive))

    a_te = dx.aggfeat(test, params=pa, **KW)
    luxor = a_te.loc[test["city"] == "Luxor", "price_mean_by_city"]
    check("unknown group 'Luxor' -> default (global mean)",
          np.allclose(luxor, cp["default"]))
    a_te2 = dx.aggfeat(a_te, params=pa, **KW)
    check("aggfeat static apply idempotent",
          np.allclose(a_te["price_mean_by_city"],
                      a_te2["price_mean_by_city"]))

    a_cnt = dx.aggfeat(train, group="city", value="city", agg="count", **KW)
    check("agg='count' works on a non-numeric value column",
          "city_count_by_city" in a_cnt.columns)
    a_multi = dx.aggfeat(train, group=["city"], value=["price", "area"],
                         agg="median", **KW)
    check("multiple value columns -> multiple features",
          "price_median_by_city" in a_multi.columns
          and "area_median_by_city" in a_multi.columns)
    try:
        dx.aggfeat(train, group="city", value="city", agg="mean", **KW)
        check("non-numeric value with agg='mean' raises", False)
    except TypeError:
        check("non-numeric value with agg='mean' raises TypeError", True)
    try:
        dx.aggfeat(train, group="city", value="price", agg="compare",
                   return_params=True, **KW)
        check("aggfeat compare + return_params raises", False)
    except ValueError:
        check("aggfeat compare + return_params raises ValueError", True)
    a_cmp = dx.aggfeat(train, group="city", value="price", agg="compare", **KW)
    check("aggfeat compare writes nothing",
          "price_mean_by_city" not in a_cmp.columns)

    # ==================================================================
    section("4. aggfeat -- the temporal-leakage guard")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dx.aggfeat(train, group="city", value="price", agg="mean", **KW)
    check("datetime column present + no as_of -> leak warning",
          any("temporal" in str(x.message).lower() for x in w))

    ts = pd.DataFrame({
        "g": ["A", "A", "A", "B", "B"],
        "t": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03",
                             "2023-01-01", "2023-01-02"]),
        "v": [10.0, 20.0, 30.0, 5.0, 15.0],
    })
    a_as, p_as = dx.aggfeat(ts, group="g", value="v", agg="mean",
                            as_of="t", return_params=True, **KW)
    feat = a_as["v_mean_by_g"].to_numpy()
    check("as_of: earliest row of each group has no past -> NaN",
          np.isnan(feat[0]) and np.isnan(feat[3]))
    check("as_of: 2nd row of A sees only [10] -> mean 10",
          abs(feat[1] - 10.0) < 1e-9)
    check("as_of: 3rd row of A sees [10,20] -> 15, NOT 30  [leak-safe]",
          abs(feat[2] - 15.0) < 1e-9)
    check("as_of params carry the per-group history",
          "history" in p_as["columns"]["v_mean_by_g"])

    ts_new = pd.DataFrame({
        "g": ["A", "B", "Z"],
        "t": pd.to_datetime(["2023-01-04", "2023-01-03", "2023-01-09"]),
        "v": [99.0, 99.0, 99.0],
    })
    a_new = dx.aggfeat(ts_new, params=p_as, **KW)
    nf = a_new["v_mean_by_g"].to_numpy()
    check("as_of apply: A at t=04 -> mean of train [10,20,30] = 20",
          abs(nf[0] - 20.0) < 1e-9)
    check("as_of apply: B at t=03 -> mean of train [5,15] = 10",
          abs(nf[1] - 10.0) < 1e-9)
    check("as_of apply: unknown group 'Z' -> default",
          abs(nf[2] - p_as["columns"]["v_mean_by_g"]["default"]) < 1e-9)

    # ==================================================================
    section("5. show/plot independence + audit trail")
    fig = dx.dtfeats(train, cols=["signup"], method="both", show=False,
                     plot=True, return_fig=True, return_df=False)
    check("dtfeats plot=True returns a Figure", fig is not None)
    aud = dx.cross(train, cols=["price", "area"], method="product", **KW)
    check("cross appends an audit entry",
          aud.attrs.get("dextra_audit", [])
          and aud.attrs["dextra_audit"][-1]["stage"] == "feature_interaction")
    aud2 = dx.aggfeat(train, group="city", value="price", agg="mean", **KW)
    check("aggfeat appends an audit entry",
          aud2.attrs["dextra_audit"][-1]["stage"] == "feature_aggregation")

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  RESULT: {ok} passed, {fail} failed")
    print("=" * 74)
    if fail:
        raise SystemExit(1)
    print("  ALL PHASE 4 STAGE 4.3 CHECKS PASSED.")


if __name__ == "__main__":
    main()
