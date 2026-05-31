"""
test_phase4_stage2.py
---------------------
Verification script for dextra Phase 4 Stage 4.2 -- bin + encode.

Run inside the project venv:
    cd "D:\\06 PythonProjects\\dextra-project"
    .\\.venv\\Scripts\\Activate.ps1
    python "C:\\Users\\manim\\OneDrive\\المستندات\\Claude\\Projects\\شرح بايثون وعمل مكتبة dextra\\test_phase4_stage2.py"

Covers: fit/apply modes, the params dict (JSON-serialisable), idempotency,
immutability, the compare mode, inplace= behaviour, out-of-range clipping
(bin), unknown-category handling (encode), and -- most importantly -- the
target-encoding K-fold guard that prevents target leakage.
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
    rng = np.random.default_rng(42)
    n = 400

    train = pd.DataFrame({
        "price": rng.gamma(2.0, 30.0, n),                  # right-skewed
        "age":   rng.integers(18, 80, n).astype(float),
        "city":  rng.choice(["Cairo", "Giza", "Alex", "Aswan"], n,
                            p=[0.50, 0.25, 0.20, 0.05]),
        "grade": rng.choice(["low", "mid", "high"], n),
    })
    # a target whose rate depends on the city -> good for target encoding
    city_rate = train["city"].map({"Cairo": 0.7, "Giza": 0.4,
                                   "Alex": 0.2, "Aswan": 0.9})
    train["churn"] = (rng.random(n) < city_rate).astype(int)
    train.loc[3:7, "price"] = np.nan                       # a few NaN

    test = pd.DataFrame({
        "price": rng.gamma(2.0, 30.0, 130),
        "age":   rng.integers(18, 80, 130).astype(float),
        "city":  rng.choice(["Cairo", "Giza", "Alex", "Aswan", "Luxor"], 130),
        "grade": rng.choice(["low", "mid", "high"], 130),
    })

    KW = dict(show=False, plot=False)

    # ==================================================================
    section("1. bin -- fit, params dict, three methods")
    b_ew, p_ew = dx.bin(train, cols=["price"], method="equal_width",
                        n_bins=5, return_params=True, **KW)
    check("equal_width creates 'price_bin'", "price_bin" in b_ew.columns)
    check("source column 'price' preserved", "price" in b_ew.columns)
    check("original train NOT mutated", "price_bin" not in train.columns)
    check("binned column is an ORDERED categorical",
          isinstance(b_ew["price_bin"].dtype, pd.CategoricalDtype)
          and b_ew["price_bin"].dtype.ordered)
    check("params is JSON-serialisable",
          json.loads(json.dumps(p_ew))["function"] == "bin")
    check("params carries a version", p_ew.get("version") is not None)
    check("equal_width: 5 bins -> 6 edges",
          len(p_ew["columns"]["price"]["edges"]) == 6)
    check("NaN positions preserved",
          b_ew["price_bin"].isna().sum() == train["price"].isna().sum())

    b_q, p_q = dx.bin(train, cols=["price"], method="quantile",
                      n_bins=4, return_params=True, **KW)
    qc = b_q["price_bin"].value_counts()
    check("quantile: bins hold ~equal counts (max-min <= 6)",
          int(qc.max() - qc.min()) <= 6)

    b_k, p_k = dx.bin(train, cols=["price"], method="kmeans",
                      n_bins=4, return_params=True, **KW)
    check("kmeans: edges strictly increasing",
          all(np.diff(p_k["columns"]["price"]["edges"]) > 0))

    # ------------------------------------------------------------------
    section("2. bin -- fit/apply, idempotency, out-of-range clipping")
    b_tr, pb = dx.bin(train, cols=["price"], method="quantile",
                      n_bins=4, return_params=True, **KW)
    b_te = dx.bin(test, params=pb, **KW)
    check("apply mode creates 'price_bin' on test",
          "price_bin" in b_te.columns)
    check("apply mode did NOT re-fit",
          b_te.attrs["dextra_audit"][-1]["mode"] == "apply")
    b_te2 = dx.bin(b_te, params=pb, **KW)
    check("idempotent under apply (inplace=False)",
          b_te["price_bin"].astype(str).equals(b_te2["price_bin"].astype(str)))

    extreme = pd.DataFrame({"price": [-1e6, 0.0, 1e9]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b_ex = dx.bin(extreme, params=pb, **KW)
    check("out-of-range values clipped into outer bins (no NaN)",
          b_ex["price_bin"].notna().all())
    check("out-of-range clipping emits a warning",
          any("outside the fitted edge range" in str(x.message) for x in w))

    # ------------------------------------------------------------------
    section("3. bin -- compare mode + error handling")
    b_cmp = dx.bin(train, cols=["price"], method="compare", n_bins=4, **KW)
    check("compare writes NOTHING (df unchanged)",
          "price_bin" not in b_cmp.columns)
    try:
        dx.bin(train, cols=["price"], method="compare",
               return_params=True, **KW)
        check("compare + return_params raises", False)
    except ValueError:
        check("compare + return_params raises ValueError", True)
    try:
        dx.bin(train, cols=["price"], n_bins=1, **KW)
        check("n_bins < 2 raises", False)
    except ValueError:
        check("n_bins < 2 raises ValueError", True)
    try:
        dx.bin(test.drop(columns=["price"]), params=pb, **KW)
        check("apply column-mismatch raises", False)
    except KeyError:
        check("apply column-mismatch raises KeyError", True)
    try:
        dx.bin(pd.DataFrame({"x": [5.0, 5.0, 5.0]}), cols=["x"], **KW)
        check("constant column raises", False)
    except ValueError:
        check("constant column raises ValueError", True)

    b_ip = dx.bin(train, cols=["age"], method="equal_width",
                  n_bins=4, inplace=True, **KW)
    check("inplace=True overwrites 'age' with the binned column",
          isinstance(b_ip["age"].dtype, pd.CategoricalDtype))
    check("inplace: original train still numeric",
          pd.api.types.is_numeric_dtype(train["age"]))

    # ==================================================================
    section("4. encode -- onehot")
    e_oh, p_oh = dx.encode(train, cols=["city"], method="onehot",
                           return_params=True, **KW)
    oh = [c for c in e_oh.columns if c.startswith("city_")]
    check("onehot: one 0/1 column per category", len(oh) == 4)
    check("onehot: dummy columns are integers",
          all(pd.api.types.is_integer_dtype(e_oh[c]) for c in oh))
    check("onehot: each row sums to exactly 1", (e_oh[oh].sum(axis=1) == 1).all())
    check("onehot: params JSON-serialisable",
          json.loads(json.dumps(p_oh))["method"] == "onehot")
    e_df = dx.encode(train, cols=["city"], method="onehot",
                     drop_first=True, **KW)
    check("drop_first: drops one dummy column",
          len([c for c in e_df.columns if c.startswith("city_")]) == 3)

    hc = pd.DataFrame({"id": [f"v{i}" for i in range(120)]})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dx.encode(hc, cols=["id"], method="onehot", **KW)
    check("high-cardinality onehot warns (curse of dimensionality)",
          any("categories" in str(x.message) for x in w))

    # ------------------------------------------------------------------
    section("5. encode -- ordinal & frequency")
    e_or, p_or = dx.encode(train, cols=["grade"], method="ordinal",
                           order=["low", "mid", "high"],
                           return_params=True, **KW)
    mo = p_or["columns"]["grade"]["mapping"]
    check("ordinal: explicit order respected (low<mid<high)",
          mo["low"] == 0 and mo["mid"] == 1 and mo["high"] == 2)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dx.encode(train, cols=["grade"], method="ordinal", **KW)
    check("ordinal without order= warns it imposes an arbitrary order",
          any("imposes a rank order" in str(x.message) for x in w))

    e_fq, p_fq = dx.encode(train, cols=["city"], method="frequency",
                           return_params=True, **KW)
    fm = p_fq["columns"]["city"]["mapping"]
    check("frequency: proportions sum to ~1",
          abs(sum(fm.values()) - 1.0) < 1e-9)
    check("frequency: 'Cairo' is the most frequent category",
          max(fm, key=fm.get) == "Cairo")

    # ------------------------------------------------------------------
    section("6. encode -- target encoding & K-fold leakage guard")
    e_tg, p_tg = dx.encode(train, cols=["city"], method="target",
                           y=train["churn"], return_params=True, **KW)
    check("target: creates 'city_target'", "city_target" in e_tg.columns)
    naive = train.groupby("city")["churn"].mean().to_dict()
    tm = p_tg["columns"]["city"]["mapping"]
    check("target: saved mapping == full-data category means (for apply)",
          all(abs(tm[k] - naive[k]) < 1e-9 for k in naive))

    try:
        dx.encode(train, cols=["city"], method="target", **KW)
        check("target without y raises", False)
    except ValueError:
        check("target without y= raises ValueError", True)

    # leakage test: a singleton category must NOT see its own target
    leak = pd.DataFrame({
        "cat": ["A"] * 80 + ["B"] * 80 + ["Z"],
        "y":   [0.0] * 80 + [0.0] * 80 + [100.0],
    })
    e_lk, p_lk = dx.encode(leak, cols=["cat"], method="target",
                           y=leak["y"], n_folds=5, return_params=True, **KW)
    z_oof = e_lk.loc[e_lk["cat"] == "Z", "cat_target"].iloc[0]
    gmean = leak["y"].mean()
    check("OOF training value of singleton 'Z' is NOT its own y=100 "
          "[leakage-safe]", abs(z_oof - 100.0) > 1.0)
    check("OOF value of 'Z' falls back to the global mean",
          abs(z_oof - gmean) < 1e-6)
    check("saved mapping['Z'] == 100 (full-data, used at apply time)",
          abs(p_lk["columns"]["cat"]["mapping"]["Z"] - 100.0) < 1e-9)

    # ------------------------------------------------------------------
    section("7. encode -- fit/apply, idempotency, unknown categories")
    for method, extra in (("onehot", {}),
                          ("ordinal", {"order": ["low", "mid", "high"]}),
                          ("frequency", {}),
                          ("target", {"y": train["churn"]})):
        col = "grade" if method == "ordinal" else "city"
        e_tr, pr = dx.encode(train, cols=[col], method=method,
                             return_params=True, **extra, **KW)
        e_te = dx.encode(test, params=pr, **KW)
        new_cols = pr["columns"][col]["new_cols"]
        check(f"{method}: apply mode runs on test",
              all(c in e_te.columns for c in new_cols))
        e_te2 = dx.encode(e_te, params=pr, **KW)
        check(f"{method}: idempotent under apply (inplace=False)",
              all(e_te[c].astype(str).equals(e_te2[c].astype(str))
                  for c in new_cols))

    # 'Luxor' is unseen in train -> handled at apply time
    _, p_uf = dx.encode(train, cols=["city"], method="frequency",
                        return_params=True, **KW)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        e_uf = dx.encode(test, params=p_uf, **KW)
    check("unknown category -> default 0.0 (frequency)",
          (e_uf.loc[test["city"] == "Luxor", "city_freq"] == 0.0).all())
    check("unknown category emits a warning",
          any("unseen during fit" in str(x.message) for x in w))

    _, p_ue = dx.encode(train, cols=["city"], method="frequency",
                        handle_unknown="error", return_params=True, **KW)
    try:
        dx.encode(test, params=p_ue, **KW)
        check("handle_unknown='error' raises on unknown category", False)
    except ValueError:
        check("handle_unknown='error' raises ValueError on unknown", True)

    e_uoh = dx.encode(test, params=p_oh, **KW)
    ohc = p_oh["columns"]["city"]["new_cols"]
    check("onehot: unknown 'Luxor' -> all-zero row",
          (e_uoh.loc[test["city"] == "Luxor", ohc].sum(axis=1) == 0).all())

    # ------------------------------------------------------------------
    section("8. encode -- inplace, compare, audit, plot")
    e_ip = dx.encode(train, cols=["city"], method="onehot",
                     inplace=True, **KW)
    check("inplace onehot drops the source 'city' column",
          "city" not in e_ip.columns
          and any(c.startswith("city_") for c in e_ip.columns))

    e_cmp = dx.encode(train, cols=["city"], method="compare", **KW)
    check("encode compare writes NOTHING",
          not any(c.startswith("city_") for c in e_cmp.columns))
    try:
        dx.encode(train, cols=["city"], method="compare",
                  return_params=True, **KW)
        check("encode compare + return_params raises", False)
    except ValueError:
        check("encode compare + return_params raises ValueError", True)
    try:
        dx.encode(test.drop(columns=["city"]), params=p_oh, **KW)
        check("encode apply column-mismatch raises", False)
    except KeyError:
        check("encode apply column-mismatch raises KeyError", True)

    aud = dx.encode(train, cols=["city"], method="frequency", **KW)
    check("audit entry appended to df.attrs",
          aud.attrs.get("dextra_audit", [])
          and aud.attrs["dextra_audit"][-1]["stage"] == "feature_encoding")

    fig_b = dx.bin(train, cols=["price"], method="quantile", show=False,
                   plot=True, return_fig=True, return_df=False)
    fig_e = dx.encode(train, cols=["city"], method="target", y=train["churn"],
                      show=False, plot=True, return_fig=True, return_df=False)
    check("bin plot=True returns a Figure", fig_b is not None)
    check("encode plot=True returns a Figure", fig_e is not None)

    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"  RESULT: {ok} passed, {fail} failed")
    print("=" * 74)
    if fail:
        raise SystemExit(1)
    print("  ALL PHASE 4 STAGE 4.2 CHECKS PASSED.")


if __name__ == "__main__":
    main()
