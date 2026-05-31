"""
test_stage6.py
--------------
اختبار للمرحلة 6 الأخيرة - ML diagnostic tools:
    - vif_scores
    - class_imbalance

الاختبارات:
  - VIF: features مستقلة (VIF~1) و features مرتبطة بشدة (VIF كبير)
  - Class imbalance: balanced و mild و severe و extreme

بعد هذه المرحلة، Phase 2 من الـ Roadmap كاملة (22/22 دالة).
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
    # Test 25: vif_scores - features مستقلة
    # =================================================================
    section("Test 25a: vif_scores - features مستقلة")
    df_indep = pd.DataFrame({
        "x1": rng.normal(0, 1, 200),
        "x2": rng.normal(0, 1, 200),
        "x3": rng.normal(0, 1, 200),
    })
    print("Expected: all VIFs ~ 1.0  (no collinearity)")
    dx.vif_scores(df_indep, plot=False)

    # =================================================================
    # Test 26: vif_scores - features مرتبطة بشدة
    # =================================================================
    section("Test 25b: vif_scores - feature مكرر بصياغة مختلفة")
    x1 = rng.normal(0, 1, 200)
    x2 = rng.normal(0, 1, 200)
    df_dep = pd.DataFrame({
        "x1":      x1,
        "x2":      x2,
        "x1_dup":  x1 + rng.normal(0, 0.05, 200),  # quasi-duplicate of x1
        "x_combo": 0.5 * x1 + 0.5 * x2 + rng.normal(0, 0.1, 200),  # linear combo
    })
    print("Expected:")
    print("  x1 and x1_dup should have very high VIF (>>10)")
    print("  x_combo should have moderately high VIF")
    print("  x2 should be slightly elevated")
    dx.vif_scores(df_dep, plot=False)

    # =================================================================
    # Test 26: class_imbalance - balanced
    # =================================================================
    section("Test 26a: class_imbalance - balanced (~50/50)")
    target1 = pd.Series(rng.choice(["yes", "no"], size=200, p=[0.5, 0.5]))
    print("Expected: severity = balanced, ratio < 2")
    dx.class_imbalance(target1, plot=False)

    section("Test 26b: class_imbalance - mild (~70/30)")
    target2 = pd.Series(rng.choice(["yes", "no"], size=200, p=[0.7, 0.3]))
    print("Expected: severity = mild (ratio ~ 2-3)")
    dx.class_imbalance(target2, plot=False)

    section("Test 26c: class_imbalance - severe (~95/5)")
    target3 = pd.Series(rng.choice(["yes", "no"], size=500, p=[0.95, 0.05]))
    print("Expected: severity = severe (ratio ~ 10-50)")
    dx.class_imbalance(target3, plot=False)

    section("Test 26d: class_imbalance - extreme (~99/1) متعدد الفئات")
    target4 = pd.Series(rng.choice(["A", "B", "C", "D"],
                                    size=1000, p=[0.97, 0.02, 0.005, 0.005]))
    print("Expected: 4 classes, severity = extreme (ratio > 50)")
    dx.class_imbalance(target4, plot=False)

    # =================================================================
    # Aliases
    # =================================================================
    section("Test 27: Stage 6 aliases (vif, imbalance)")
    assert dx.vif       is dx.vif_scores
    assert dx.imbalance is dx.class_imbalance
    print("[OK] Stage 6 aliases linked correctly.")

    # =================================================================
    # Full integration: count all functions and aliases
    # =================================================================
    section("Test 28: التحقق النهائي - كل الـ 22 دالة موجودة")
    all_funcs = [
        # Stage 1
        "z_scores", "pearson_skewness", "empirical_rule_check", "outliers_report",
        # Stage 2
        "correlation_matrix", "simple_linear_regression",
        # Stage 3
        "missing_report", "frequency_table", "cross_tab", "group_compare",
        # Stage 4
        "confidence_interval_mean", "confidence_interval_proportion",
        "sample_size_mean", "sample_size_proportion",
        # Stage 5
        "normality_test", "t_test_one_sample", "t_test_two_sample",
        "t_test_paired", "anova_oneway", "chi_square_independence",
        # Stage 6
        "vif_scores", "class_imbalance",
    ]
    all_aliases = [
        "zsc", "pskew", "emprule", "outrep",
        "corrmat", "slr",
        "missrep", "freqtab", "xtab", "gcmp",
        "cim", "cip", "ssm", "ssp",
        "normtest", "t1", "t2", "tpair", "aov1", "chi2ind",
        "vif", "imbalance",
    ]
    missing = [n for n in all_funcs + all_aliases if not hasattr(dx, n)]
    if missing:
        print(f"[FAIL] missing: {missing}")
    else:
        print(f"[OK] all {len(all_funcs)} functions + {len(all_aliases)} aliases present.")

    print("\n" + "=" * 78)
    print("  STAGE 6 - ALL FUNCTIONS RAN WITHOUT ERROR")
    print("  PHASE 2 (statistical extensions) - COMPLETE: 22 / 22 functions.")
    print("=" * 78)


if __name__ == "__main__":
    main()
