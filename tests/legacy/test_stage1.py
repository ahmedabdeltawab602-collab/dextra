"""
test_stage1.py
--------------
سكربت اختبار للمرحلة 1 من dextra.stats_advanced.

نختبر 4 دوال جديدة على DataFrame بخصائص معروفة:
  - 'normal'   : توزيع طبيعي قياسي (μ≈0, σ≈1)
  - 'skewed'   : توزيع لوغ-طبيعي (ملتوٍ يمين بقوة)
  - 'outlier'  : توزيع طبيعي + 5 قيم متطرفة مزروعة يدوياً

طريقة التشغيل:
    (بعد تفعيل الـ venv ووجود dextra مثبتة)
    python test_stage1.py

المتوقع لكل دالة موثَّق في commentary أسفل كل قسم.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import dextra as dx


def build_sample() -> pd.DataFrame:
    """يبني DataFrame بثلاث أعمدة ذات خصائص معروفة لاختبار الدوال."""
    rng = np.random.default_rng(42)

    normal = rng.normal(loc=0.0, scale=1.0, size=1000)

    # log-normal: skewed right strongly
    skewed = rng.lognormal(mean=0.0, sigma=0.8, size=1000)

    # normal + planted outliers
    outlier_col = rng.normal(loc=50.0, scale=5.0, size=995).tolist()
    outlier_col += [120.0, 125.0, 130.0, -20.0, -25.0]  # 5 outliers
    outlier_col = np.array(outlier_col)
    rng.shuffle(outlier_col)

    return pd.DataFrame({
        "normal":  normal,
        "skewed":  skewed,
        "outlier": outlier_col,
    })


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> None:
    print(f"dextra version: {dx.__version__}")
    df = build_sample()
    print(f"\nBuilt sample DataFrame shape={df.shape}, columns={list(df.columns)}")

    # ---------------------------------------------------------------- 1) z_scores
    section("Test 1: dx.z_scores(df)")
    print("Expected:")
    print("  - mean_z ≈ 0, std_z ≈ 1 for every column.")
    print("  - 'outlier' column shows several |Z| > 3.")
    print("  - 'normal' column shows ~0.3% extreme values (theoretical for |Z|>3 on Normal).")
    dx.z_scores(df, plot=False)  # plot=False so the script doesn't hang on plots

    # ---------------------------------------------------------------- 2) pearson_skewness
    section("Test 2: dx.pearson_skewness(df)")
    print("Expected:")
    print("  - 'normal'  → skew_pearson ≈ 0,  direction=symmetric, magnitude=low.")
    print("  - 'skewed'  → skew_pearson > 0.5, direction=right, magnitude=moderate/high.")
    print("  - 'outlier' → skew_pearson small but not zero (extremes pull the tail).")
    dx.pearson_skewness(df, plot=False)

    # ---------------------------------------------------------------- 3) empirical_rule_check
    section("Test 3: dx.empirical_rule_check(df)")
    print("Expected:")
    print("  - 'normal'  → all bands within ±2% of 68.27/95.45/99.73 → looks_normal=True.")
    print("  - 'skewed'  → 1σ band well above 68% (mass concentrated near mean for lognormal).")
    print("  - 'outlier' → 3σ band below 99.73% (outliers spill outside).")
    dx.empirical_rule_check(df, plot=False)

    # ---------------------------------------------------------------- 4a) outliers_report (IQR)
    section("Test 4a: dx.outliers_report(df, method='iqr')")
    print("Expected:")
    print("  - 'normal'  → ~0.7% rows flagged (theoretical for 1.5·IQR on Normal).")
    print("  - 'skewed'  → many rows flagged (heavy right tail).")
    print("  - 'outlier' → at least 5 rows (the planted extremes) — likely more.")
    dx.outliers_report(df, method="iqr", plot=False)

    # ---------------------------------------------------------------- 4b) outliers_report (zscore)
    section("Test 4b: dx.outliers_report(df, method='zscore', z_threshold=3)")
    print("Expected:")
    print("  - 'normal'  → ~0.3% rows.")
    print("  - 'outlier' → at least 5 rows flagged for the planted extremes.")
    dx.outliers_report(df, method="zscore", z_threshold=3, plot=False)

    # ---------------------------------------------------------------- 4c) outliers_report (rows)
    section("Test 4c: dx.outliers_report(df, return_rows=True) — actual rows")
    rows = dx.outliers_report(df, method="iqr", show=False, plot=False, return_rows=True)
    print(f"Number of outlier rows extracted: {len(rows)}")
    print("First 5 outlier rows:")
    print(rows.head().to_string())

    # ---------------------------------------------------------------- aliases sanity check
    section("Test 5: aliases (zsc, pskew, emprule, outrep)")
    assert dx.zsc      is dx.z_scores
    assert dx.pskew    is dx.pearson_skewness
    assert dx.emprule  is dx.empirical_rule_check
    assert dx.outrep   is dx.outliers_report
    print("[OK] جميع الـ aliases مرتبطة بالدوال الكاملة.")

    print("\n" + "=" * 72)
    print("  STAGE 1 — ALL FUNCTIONS RAN WITHOUT ERROR")
    print("=" * 72)
    print("\nTip: لتجربة الرسوم البصرية، شغّل في Jupyter notebook بدلاً من السكربت،")
    print("     أو شيل علامة plot=False من أي مكالمة.\n")


if __name__ == "__main__":
    main()
