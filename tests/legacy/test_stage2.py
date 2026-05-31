"""
test_stage2.py
--------------
اختبار للمرحلة 2 من dextra.stats_advanced:
    - correlation_matrix
    - simple_linear_regression

البيانات المُصمَّمة (1000 صف):
    x        : توزيع طبيعي N(0,1)              (المتغير الأساسي)
    y_linear : 2*x + 5 + ضوضاء N(0,0.3)         (ارتباط خطي شديد بـ x)
    y_nonlin : x^3 + ضوضاء N(0,1.0)            (ارتباط رتبي قوي + خطي متوسط)
    z_indep  : N(0,1) مستقلة عن x               (ارتباط ≈ 0)

التوقعات الكمية الدقيقة:
    Pearson(x, y_linear) ≈ +0.99
    Pearson(x, y_nonlin) ≈ +0.85
    Spearman(x, y_nonlin) ≈ +0.99  (يفوق Pearson لأن العلاقة رتبية تامة)
    Pearson(x, z_indep)  ≈ 0       (مستقل)

    SLR(y_linear ~ x) : slope ≈ 2.0, intercept ≈ 5.0, r² ≈ 0.98+
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import dextra as dx


def build_sample() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 1000
    x = rng.normal(0, 1, n)
    y_linear = 2.0 * x + 5.0 + rng.normal(0, 0.3, n)
    y_nonlin = x ** 3 + rng.normal(0, 1.0, n)
    z_indep = rng.normal(0, 1, n)
    return pd.DataFrame({
        "x": x,
        "y_linear": y_linear,
        "y_nonlin": y_nonlin,
        "z_indep": z_indep,
    })


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main() -> None:
    print(f"dextra version: {dx.__version__}")
    df = build_sample()
    print(f"\nBuilt sample DataFrame shape={df.shape}, columns={list(df.columns)}")

    # ----------------------------------------------- 5) correlation_matrix
    section("Test 5a: dx.correlation_matrix(df, method='pearson')")
    print("Expected:")
    print("  - Pearson(x, y_linear) ≈ +0.99 (***)")
    print("  - Pearson(x, y_nonlin) ≈ +0.85 (***)")
    print("  - Pearson(x, z_indep)  ≈ 0 (ns)")
    dx.correlation_matrix(df, method="pearson", plot=False)

    section("Test 5b: dx.correlation_matrix(df, method='spearman')")
    print("Expected:")
    print("  - Spearman(x, y_nonlin) ≈ +0.99 (يفوق Pearson لأن العلاقة رتبية تامة)")
    dx.correlation_matrix(df, method="spearman", plot=False)

    section("Test 5c: dx.correlation_matrix → return_p=True يُعيد (r_mat, p_mat)")
    r, p = dx.correlation_matrix(df, show=False, plot=False, return_p=True)
    print(f"r matrix shape: {r.shape}")
    print(f"p matrix shape: {p.shape}")
    print(f"r[x, y_linear] = {r.loc['x', 'y_linear']:+.4f}")
    print(f"p[x, y_linear] = {p.loc['x', 'y_linear']:.2e}")
    print(f"r[x, z_indep]  = {r.loc['x', 'z_indep']:+.4f}")
    print(f"p[x, z_indep]  = {p.loc['x', 'z_indep']:.4f}")

    # ----------------------------------------------- 6) simple_linear_regression
    section("Test 6a: dx.simple_linear_regression(df, x='x', y='y_linear')")
    print("Expected:")
    print("  - slope ≈ +2.00, intercept ≈ +5.00, r² > 0.97, p < 0.001")
    print("  - residuals normality: should pass (we generated Gaussian noise)")
    dx.simple_linear_regression(df, x="x", y="y_linear", plot=False)

    section("Test 6b: dx.simple_linear_regression(df, x='x', y='y_nonlin')")
    print("Expected:")
    print("  - slope significant but r² ≈ 0.72 (linear approx of cubic)")
    print("  - residuals may be NON-normal (cubic model fits poorly with line)")
    dx.simple_linear_regression(df, x="x", y="y_nonlin", plot=False)

    section("Test 6c: dx.simple_linear_regression → return_residuals=True")
    summary, residuals = dx.simple_linear_regression(
        df, x="x", y="y_linear", show=False, plot=False,
        return_df=True, return_residuals=True,
    )
    print(f"Summary index: {list(summary.index)[:5]}...")
    print(f"Residuals: count={len(residuals)}, "
          f"mean={residuals.mean():+.4f}, std={residuals.std():.4f}")

    # ----------------------------------------------- aliases sanity check
    section("Test 7: aliases (corrmat, slr)")
    assert dx.corrmat is dx.correlation_matrix
    assert dx.slr     is dx.simple_linear_regression
    print("[OK] aliases مرتبطة بالدوال الكاملة.")

    # ----------------------------------------------- Stage 1 aliases still work
    section("Test 8: تأكد إن Stage 1 لسة سليمة")
    assert dx.zsc      is dx.z_scores
    assert dx.pskew    is dx.pearson_skewness
    assert dx.emprule  is dx.empirical_rule_check
    assert dx.outrep   is dx.outliers_report
    print("[OK] aliases الـ Stage 1 شغّالة كلها.")

    print("\n" + "=" * 78)
    print("  STAGE 2 — ALL FUNCTIONS RAN WITHOUT ERROR")
    print("=" * 78)
    print("\nTip: لتجربة الرسوم البصرية، شغّل في Jupyter ومش هتحتاج plot=False:")
    print("     dx.correlation_matrix(df)          # heatmap")
    print("     dx.simple_linear_regression(df, x='x', y='y_linear')  # 3-panel\n")


if __name__ == "__main__":
    main()
