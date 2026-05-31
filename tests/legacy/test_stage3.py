"""
test_stage3.py
--------------
اختبار للمرحلة 3 من dextra.stats_advanced - EDA market tools:
    - missing_report
    - frequency_table
    - cross_tab
    - group_compare

البيانات المُصمَّمة:
    - 200 صف، 5 أعمدة
    - amount     : رقمي، 0% مفقود
    - score      : رقمي، 10% مفقود (يجب يقترح impute)
    - notes      : نص، 70% مفقود (يجب يقترح drop_column)
    - region     : فئوي 4 فئات
    - product    : فئوي 3 فئات
    - الفئات مرتبطة: A->X, B->Y, C->Z غالباً (Cramér's V قوي)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import dextra as dx


def build_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 200

    region = rng.choice(["North", "South", "East", "West"], size=n,
                        p=[0.4, 0.3, 0.2, 0.1])

    # product strongly tied to region
    product = np.empty(n, dtype=object)
    for i, r in enumerate(region):
        if r == "North":
            product[i] = rng.choice(["X", "Y", "Z"], p=[0.7, 0.2, 0.1])
        elif r == "South":
            product[i] = rng.choice(["X", "Y", "Z"], p=[0.1, 0.7, 0.2])
        elif r == "East":
            product[i] = rng.choice(["X", "Y", "Z"], p=[0.1, 0.2, 0.7])
        else:
            product[i] = rng.choice(["X", "Y", "Z"], p=[0.33, 0.33, 0.34])

    # amount: depends on region
    base = {"North": 100, "South": 60, "East": 80, "West": 50}
    amount = np.array([base[r] for r in region]) + rng.normal(0, 10, n)

    # score: 10% missing
    score = rng.uniform(0, 100, n)
    miss_idx = rng.choice(n, size=int(n * 0.10), replace=False)
    score[miss_idx] = np.nan

    # notes: 70% missing
    notes = np.array(["note"] * n, dtype=object)
    miss_idx2 = rng.choice(n, size=int(n * 0.70), replace=False)
    notes[miss_idx2] = None

    return pd.DataFrame({
        "amount":  amount,
        "score":   score,
        "notes":   notes,
        "region":  region,
        "product": product,
    })


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> None:
    print(f"dextra version: {dx.__version__}")
    df = build_sample()
    print(f"\nBuilt sample DataFrame shape={df.shape}, columns={list(df.columns)}")

    # ----------------------------------------------- 7) missing_report
    section("Test 7: dx.missing_report(df)")
    print("Expected:")
    print("  - amount : pct_missing = 0%   -> OK")
    print("  - score  : pct_missing ~ 10%  -> review")
    print("  - notes  : pct_missing ~ 70%  -> drop_column")
    dx.missing_report(df, plot=False)

    # ----------------------------------------------- 8) frequency_table
    section("Test 8a: dx.frequency_table(df, 'region')")
    print("Expected:")
    print("  - North ~ 40%, South ~ 30%, East ~ 20%, West ~ 10%")
    print("  - Top category 'North' covers about 40%")
    dx.frequency_table(df, "region", plot=False)

    section("Test 8b: dx.frequency_table(df, 'product', top_n=2)")
    print("Expected:")
    print("  - Two largest products + an <other> row collapsing the rest.")
    dx.frequency_table(df, "product", top_n=2, plot=False)

    # ----------------------------------------------- 9) cross_tab
    section("Test 9a: dx.cross_tab(df, row='region', col='product')")
    print("Expected:")
    print("  - chi-square should be significant (region drives product)")
    print("  - Cramér's V should be moderate to strong (>= 0.3)")
    dx.cross_tab(df, row="region", col="product", plot=False)

    section("Test 9b: dx.cross_tab(...) with return_test=True")
    table, test = dx.cross_tab(df, row="region", col="product",
                                show=False, plot=False, return_test=True)
    print(f"chi2 = {test['chi2']:.4f}")
    print(f"dof = {test['dof']}")
    print(f"p_value = {test['p_value']:.6f}")
    print(f"Cramer's V = {test['cramers_v']:.4f}  ({test['strength']})")
    print(f"n = {test['n']}")

    # ----------------------------------------------- 10) group_compare
    section("Test 10: dx.group_compare(df, group_col='region', value_cols=['amount', 'score'])")
    print("Expected:")
    print("  - North highest mean amount (~100), West lowest (~50)")
    print("  - score should be roughly similar across regions (drawn uniformly)")
    dx.group_compare(df, group_col="region", value_cols=["amount", "score"], plot=False)

    # ----------------------------------------------- aliases sanity check
    section("Test 11: aliases (missrep, freqtab, xtab, gcmp)")
    assert dx.missrep is dx.missing_report
    assert dx.freqtab is dx.frequency_table
    assert dx.xtab    is dx.cross_tab
    assert dx.gcmp    is dx.group_compare
    print("[OK] Stage 3 aliases are linked to full functions.")

    print("\n" + "=" * 78)
    print("  STAGE 3 - ALL FUNCTIONS RAN WITHOUT ERROR")
    print("=" * 78)
    print("\nTip: To see the visuals, run in Jupyter without plot=False:")
    print("     dx.missing_report(df)")
    print("     dx.frequency_table(df, 'region')")
    print("     dx.cross_tab(df, row='region', col='product')")
    print("     dx.group_compare(df, group_col='region', value_cols=['amount'])\n")


if __name__ == "__main__":
    main()
