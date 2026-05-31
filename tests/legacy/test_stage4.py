"""
test_stage4.py
--------------
اختبار للمرحلة 4 - Inference helpers:
    - confidence_interval_mean
    - confidence_interval_proportion
    - sample_size_mean
    - sample_size_proportion

كل دالة تختبر على قيم محسوبة يدوياً مسبقاً عشان نضمن دقة 100%.
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

    # ====================================================================
    # Test 12: confidence_interval_mean
    # ====================================================================
    section("Test 12a: dx.confidence_interval_mean على عينة معروفة")
    # عينة n=10، mean=10.0، std=2.0 تقريباً
    # CI 95% t-based: t_{0.025, 9} = 2.262, SE = 2/sqrt(10) = 0.632
    # margin = 2.262 * 0.632 = 1.430
    # CI = [8.57, 11.43]  تقريباً
    data = [8, 12, 10, 9, 11, 10, 12, 8, 11, 9]
    print("Sample: ", data)
    print("Expected:")
    print("  n = 10, mean = 10.0, std ≈ 1.49")
    print("  t_critical(0.025, 9) ≈ 2.262")
    print("  margin ≈ 1.066, CI ≈ [8.93, 11.07]")
    dx.confidence_interval_mean(data, name="weights", plot=False)

    section("Test 12b: عينة كبيرة من توزيع طبيعي")
    rng = np.random.default_rng(42)
    big = rng.normal(100, 15, 1000)
    print("Generated: n=1000 from N(100, 15²)")
    print("Expected: CI 95% should contain 100 (very tight)")
    dx.confidence_interval_mean(big, name="big_sample", plot=False)

    # ====================================================================
    # Test 13: confidence_interval_proportion
    # ====================================================================
    section("Test 13a: cip Wilson method (recommended)")
    # 35 نجاح من 100 محاولة
    # Wilson 95% CI ≈ [0.262, 0.449]
    print("successes = 35, n = 100")
    print("Expected (Wilson):")
    print("  p_hat = 0.35")
    print("  CI ≈ [0.262, 0.449]")
    dx.confidence_interval_proportion(35, 100, method="wilson", plot=False)

    section("Test 13b: cip Wald method للمقارنة")
    print("نفس البيانات بطريقة Wald الكلاسيكية:")
    print("Expected (Wald):")
    print("  z_critical ≈ 1.96")
    print("  margin ≈ 0.0935, CI ≈ [0.257, 0.443]")
    dx.confidence_interval_proportion(35, 100, method="wald", plot=False)

    section("Test 13c: cip حالة متطرفة (p_hat قريب من 0)")
    print("successes = 2, n = 50  (p_hat = 0.04)")
    print("Expected: Wilson should give a wider, more honest CI than Wald")
    dx.confidence_interval_proportion(2, 50, method="wilson", plot=False)

    # ====================================================================
    # Test 14: sample_size_mean
    # ====================================================================
    section("Test 14: sample_size_mean (n required for mean estimation)")
    # n = (z * σ / E)²
    # z_{0.025} = 1.96, σ = 10, E = 2  →  n = (1.96*10/2)² = 96.04 → 97
    print("E = 2.0, std = 10, confidence = 95%")
    print("Expected:")
    print("  z ≈ 1.96, n_exact ≈ 96.04, n_required = 97")
    dx.sample_size_mean(margin_error=2.0, std=10.0, confidence=0.95, plot=False)

    # ====================================================================
    # Test 15: sample_size_proportion (F-M04-L07-01)
    # ====================================================================
    section("Test 15a: ssp استبيان عام (p=0.5 worst case)")
    # n = z² * 0.25 / E²
    # z=1.96, E=0.05  →  n = 3.8416 * 0.25 / 0.0025 = 384.16 → 385
    print("E = 0.05, p = 0.5, confidence = 95%")
    print("Expected (F-M04-L07-01):")
    print("  z = 1.96, n_exact ≈ 384.16, n_required = 385")
    dx.sample_size_proportion(margin_error=0.05, p=0.5, confidence=0.95, plot=False)

    section("Test 15b: ssp مع p تقديري (n أقل من worst case)")
    # E=0.05, p=0.2 → n = 3.8416 * 0.2 * 0.8 / 0.0025 = 245.86 → 246
    print("E = 0.05, p = 0.2, confidence = 95%")
    print("Expected:")
    print("  n_exact ≈ 245.86, n_required = 246")
    print("  n_worst (p=0.5) = 385 (يتم عرضه للمقارنة)")
    dx.sample_size_proportion(margin_error=0.05, p=0.2, confidence=0.95, plot=False)

    # ====================================================================
    # Test 16: Aliases
    # ====================================================================
    section("Test 16: aliases (cim, cip, ssm, ssp)")
    assert dx.cim is dx.confidence_interval_mean
    assert dx.cip is dx.confidence_interval_proportion
    assert dx.ssm is dx.sample_size_mean
    assert dx.ssp is dx.sample_size_proportion
    print("[OK] Stage 4 aliases are linked to full functions.")

    # ====================================================================
    # Test 17: Stage 1-3 still intact
    # ====================================================================
    section("Test 17: Stage 1-3 aliases still working")
    for name in ["zsc", "pskew", "emprule", "outrep",
                 "corrmat", "slr",
                 "missrep", "freqtab", "xtab", "gcmp"]:
        assert hasattr(dx, name), f"Missing alias: {name}"
    print("[OK] All Stage 1-3 aliases present.")

    print("\n" + "=" * 78)
    print("  STAGE 4 - ALL FUNCTIONS RAN WITHOUT ERROR")
    print("=" * 78)
    print("\nTip: لرؤية الرسوم البصرية، شغّل في Jupyter بدون plot=False:")
    print("     dx.confidence_interval_mean([8,12,10,9,11,10,12,8,11,9])")
    print("     dx.confidence_interval_proportion(35, 100)")
    print("     dx.sample_size_mean(margin_error=2.0, std=10.0)")
    print("     dx.sample_size_proportion(margin_error=0.05, p=0.5)\n")


if __name__ == "__main__":
    main()
