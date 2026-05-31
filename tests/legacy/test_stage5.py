"""
test_stage5.py
--------------
اختبار للمرحلة 5 - Hypothesis tests:
    - normality_test
    - t_test_one_sample
    - t_test_two_sample
    - t_test_paired
    - anova_oneway
    - chi_square_independence

كل اختبار يُجرى على بيانات مصمَّمة بنتائج متوقَّعة معروفة:
    - عينة طبيعية يجب ألا يتم رفض H0 لـ normality
    - عينة لوغ-طبيعية يجب أن يتم رفضها
    - مجموعتان متطابقتان (H0 صحيح): يجب ألا يتم رفض H0 لـ t-test
    - مجموعتان مختلفتان (مع تأثير): يجب رفض H0
    - مجموعات ANOVA متشابهة vs مختلفة
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import dextra as dx


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> None:
    print(f"dextra version: {dx.__version__}")
    rng = np.random.default_rng(42)

    # =================================================================
    # Test 18: normality_test
    # =================================================================
    section("Test 18a: normality_test على عينة طبيعية")
    normal = rng.normal(0, 1, 200)
    print("Expected: Shapiro-Wilk should NOT reject H0 (p > 0.05)")
    dx.normality_test(normal, plot=False)

    section("Test 18b: normality_test على عينة لوغ-طبيعية (مش طبيعية)")
    lognorm = rng.lognormal(0, 1, 200)
    print("Expected: Shapiro-Wilk should REJECT H0 (p < 0.05)")
    dx.normality_test(lognorm, plot=False)

    # =================================================================
    # Test 19: t_test_one_sample
    # =================================================================
    section("Test 19: t_test_one_sample (H0: mean = 100)")
    data = rng.normal(loc=100, scale=15, size=50)
    print("Generated: n=50 from N(100, 15). H0: mean = 100 should NOT be rejected.")
    dx.t_test_one_sample(data, popmean=100, plot=False)

    section("Test 19b: t_test_one_sample (H0: mean = 110, FALSE)")
    print("نفس البيانات (mean ~100), H0: mean = 110 should be REJECTED.")
    dx.t_test_one_sample(data, popmean=110, plot=False)

    # =================================================================
    # Test 20: t_test_two_sample (F-M05-L06-01)
    # =================================================================
    section("Test 20a: t_test_two_sample - مجموعتان متطابقتان")
    g1 = rng.normal(50, 10, 100)
    g2 = rng.normal(50, 10, 100)
    print("Both groups N(50, 10). Expected: NOT to reject H0 (no real diff).")
    dx.t_test_two_sample(g1, g2, name1="control", name2="placebo", plot=False)

    section("Test 20b: t_test_two_sample - مجموعتان مختلفتان")
    g1 = rng.normal(50, 10, 100)
    g2 = rng.normal(55, 10, 100)  # 5-point shift
    print("Group2 has +5 shift. Expected: REJECT H0, mean_diff ~ -5.")
    dx.t_test_two_sample(g1, g2, name1="control", name2="treatment", plot=False)

    # =================================================================
    # Test 21: t_test_paired
    # =================================================================
    section("Test 21: t_test_paired - قبل/بعد مع تحسّن منهجي")
    before = rng.normal(70, 12, 30)
    improvement = rng.normal(5, 3, 30)  # mean +5 with noise
    after = before + improvement
    print("After = before + N(5, 3). Expected: REJECT H0, mean_diff ~ +5.")
    dx.t_test_paired(before, after, name_before="pre", name_after="post", plot=False)

    # =================================================================
    # Test 22: anova_oneway
    # =================================================================
    section("Test 22a: anova_oneway - مجموعات متشابهة")
    df_anova = pd.DataFrame({
        "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        "value": np.concatenate([
            rng.normal(50, 10, 30),
            rng.normal(50, 10, 30),
            rng.normal(50, 10, 30),
        ]),
    })
    print("All 3 groups N(50, 10). Expected: NOT to reject H0.")
    dx.anova_oneway(df_anova, group_col="group", value_col="value", plot=False)

    section("Test 22b: anova_oneway - مجموعات مختلفة")
    df_anova2 = pd.DataFrame({
        "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        "value": np.concatenate([
            rng.normal(50, 8, 30),
            rng.normal(60, 8, 30),
            rng.normal(70, 8, 30),
        ]),
    })
    print("Group means: 50/60/70. Expected: REJECT H0 strongly. eta² large.")
    dx.anova_oneway(df_anova2, group_col="group", value_col="value", plot=False)

    # =================================================================
    # Test 23: chi_square_independence
    # =================================================================
    section("Test 23: chi_square_independence")
    # Build dependent two-variable data
    n = 200
    region = rng.choice(["N", "S", "E"], size=n, p=[0.4, 0.4, 0.2])
    product = np.empty(n, dtype=object)
    for i, r in enumerate(region):
        if r == "N":
            product[i] = rng.choice(["X", "Y"], p=[0.8, 0.2])
        elif r == "S":
            product[i] = rng.choice(["X", "Y"], p=[0.2, 0.8])
        else:
            product[i] = rng.choice(["X", "Y"], p=[0.5, 0.5])
    df_chi = pd.DataFrame({"region": region, "product": product})
    print("Strong region->product dependence. Expected: REJECT H0, Cramer's V moderate-strong.")
    dx.chi_square_independence(df_chi, row="region", col="product", plot=False)

    # =================================================================
    # Aliases
    # =================================================================
    section("Test 24: Stage 5 aliases (normtest, t1, t2, tpair, aov1, chi2ind)")
    assert dx.normtest is dx.normality_test
    assert dx.t1      is dx.t_test_one_sample
    assert dx.t2      is dx.t_test_two_sample
    assert dx.tpair   is dx.t_test_paired
    assert dx.aov1    is dx.anova_oneway
    assert dx.chi2ind is dx.chi_square_independence
    print("[OK] all Stage 5 aliases linked correctly.")

    print("\n" + "=" * 78)
    print("  STAGE 5 - ALL FUNCTIONS RAN WITHOUT ERROR")
    print("=" * 78)
    print("\nTip: Run in Jupyter (without plot=False) to see all visuals.\n")


if __name__ == "__main__":
    main()
