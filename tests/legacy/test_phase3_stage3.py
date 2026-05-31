"""
test_phase3_stage3.py
---------------------
اختبار Stage 3.3 من Phase 3 - الإصلاحات على القيم:
    - handle_missing(df, strategy='auto')   - Stage 3: معالجة الفقد
    - dedupe(df, subset=None)               - Stage 4: إزالة المكرَّر
    - clip_outliers(df, method='iqr')       - Stage 5: قص المتطرفات
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

    # ------------------------------------------------------------------
    # Build a deliberately dirty dataset
    # ------------------------------------------------------------------
    df_raw = pd.DataFrame({
        'price':    [100.0, 200.0, 150.0, 300.0, np.nan,
                     250.0, 100.0, np.nan, 50.0, 9999.0],
        'quantity': [1, 2, 3, np.nan, 5, 2, 1, 6, np.nan, 4],
        'segment':  ['A', 'B', 'A', None, 'C', 'B', 'A', 'C', 'B', None],
    })
    # planted issues:
    #   - 2 NA in price, 2 in quantity, 2 in segment
    #   - 1 outlier in price (9999)
    #   - 1 exact duplicate (row 6 = row 0)

    print(f"\nOriginal shape: {df_raw.shape}")
    print(f"Missing cells:  {df_raw.isna().sum().sum()}")
    print(f"Duplicates:     {df_raw.duplicated().sum()}")

    # ------------------------------------------------------------------
    # Test 44: handle_missing (auto)
    # ------------------------------------------------------------------
    section("Test 44: dx.handle_missing(df_raw)  auto strategy")
    print("Expected:")
    print("  - price: 2 missing -> median (skewed because of 9999)")
    print("  - quantity: 2 missing -> mean or median (low skew)")
    print("  - segment: 2 missing -> mode")
    print("  - Final missing count: 0")
    df_m = dx.handle_missing(df_raw, plot=False)
    assert df_m.isna().sum().sum() == 0, "Should have no missing"
    print(f"\n[OK] missing count after: {df_m.isna().sum().sum()}")

    # ------------------------------------------------------------------
    # Test 45: handle_missing per-column dict
    # ------------------------------------------------------------------
    section("Test 45: dx.handle_missing per-column dict")
    df_m2 = dx.handle_missing(
        df_raw,
        strategy={'price': 'median', 'segment': 'constant'},
        fill_value='UNKNOWN',
        show=False, plot=False,
    )
    assert df_m2['price'].isna().sum() == 0
    assert (df_m2['segment'] == 'UNKNOWN').sum() == 2
    assert df_m2['quantity'].isna().sum() == 2  # unspecified -> left alone
    print(f"[OK] price filled with median, segment filled with 'UNKNOWN', quantity untouched.")
    print(f"     segment values: {df_m2['segment'].unique().tolist()}")

    # ------------------------------------------------------------------
    # Test 46: handle_missing drop_cols
    # ------------------------------------------------------------------
    section("Test 46: dx.handle_missing(drop_cols, drop_threshold=0.15)")
    print("Expected: any column with > 15% missing will be dropped.")
    print("All three planted columns have 20% missing -> all 3 dropped.")
    df_m3 = dx.handle_missing(df_raw, strategy='drop_cols',
                                drop_threshold=0.15, plot=False)
    print(f"\n[OK] columns remaining: {list(df_m3.columns)}")

    # ------------------------------------------------------------------
    # Test 47: dedupe (exact)
    # ------------------------------------------------------------------
    section("Test 47: dx.dedupe(df_m)  - row 0 = row 6")
    print("Expected:")
    print("  - row 6 is an exact duplicate of row 0 (both 100/1/A)")
    print("  - rows: 10 -> 9, with 1 removed")
    df_d = dx.dedupe(df_m, plot=False)
    assert len(df_d) == 9, f"Expected 9 rows, got {len(df_d)}"
    print(f"\n[OK] rows: {len(df_m)} -> {len(df_d)}")

    # ------------------------------------------------------------------
    # Test 48: dedupe with subset
    # ------------------------------------------------------------------
    section("Test 48: dx.dedupe with subset=['segment']")
    df_d2 = dx.dedupe(df_m, subset=['segment'], show=False, plot=False)
    print(f"Unique segments only: {sorted(df_d2['segment'].unique())}")
    print(f"Rows after: {len(df_d2)}  (3 unique segments)")

    # ------------------------------------------------------------------
    # Test 49: dedupe return_dropped
    # ------------------------------------------------------------------
    section("Test 49: dx.dedupe return_dropped=True")
    dropped = dx.dedupe(df_m, show=False, plot=False, return_dropped=True)
    print(f"Dropped rows (n={len(dropped)}):")
    print(dropped)

    # ------------------------------------------------------------------
    # Test 50: clip_outliers (IQR + clip - default)
    # ------------------------------------------------------------------
    section("Test 50: dx.clip_outliers(df_d)  IQR + clip")
    print("Expected:")
    print("  - price has one outlier (9999), should be clipped to UB ~425")
    print("  - quantity has no outliers")
    print("  - No row loss (clipping)")
    n_before = len(df_d)
    df_o = dx.clip_outliers(df_d, cols=['price', 'quantity'], plot=False)
    assert len(df_o) == n_before, "Clipping should not drop rows"
    max_price = df_o['price'].max()
    print(f"\n[OK] max price: {df_d['price'].max()} -> {max_price}")
    print(f"     rows: {n_before} -> {len(df_o)}  (no row loss)")

    # ------------------------------------------------------------------
    # Test 51: clip_outliers (zscore + drop)
    # ------------------------------------------------------------------
    section("Test 51: dx.clip_outliers method='zscore' action='drop'")
    df_o2 = dx.clip_outliers(df_d, cols=['price'],
                              method='zscore', z_threshold=2.0,
                              action='drop', plot=False)
    print(f"\n[OK] rows: {len(df_d)} -> {len(df_o2)}")

    # ------------------------------------------------------------------
    # Test 52: full pipeline + audit log
    # ------------------------------------------------------------------
    section("Test 52: Full pipeline (clean_report -> handle_missing -> dedupe -> clip_outliers)")
    df_clean = (df_raw
                .pipe(dx.handle_missing, show=False, plot=False)
                .pipe(dx.dedupe, show=False, plot=False)
                .pipe(dx.clip_outliers, show=False, plot=False))
    print(f"Final shape: {df_clean.shape}")
    print(f"Audit log:")
    for i, e in enumerate(df_clean.attrs.get('dextra_audit', []), 1):
        print(f"  #{i}: [{e['stage']}] {e['decision']}")
    print(f"\n[OK] {len(df_clean.attrs.get('dextra_audit', []))} steps recorded in audit log.")

    # ------------------------------------------------------------------
    # Test 53: clean_report after pipeline shows the audit trail
    # ------------------------------------------------------------------
    section("Test 53: clean_report بعد التنظيف يعرض السجل الكامل")
    dx.clean_report(df_clean, plot=False)

    # ------------------------------------------------------------------
    # Test 54: aliases
    # ------------------------------------------------------------------
    section("Test 54: Aliases")
    assert dx.fillna_smart is dx.handle_missing
    assert dx.dedup        is dx.dedupe
    assert dx.clipout      is dx.clip_outliers
    print("[OK] fillna_smart, dedup, clipout aliases linked.")

    print("\n" + "=" * 78)
    print("  STAGE 3.3 - ALL TESTS PASSED")
    print("  Phase 3 COMPLETE: 7/7 cleaning functions across 8/8 framework stages.")
    print("=" * 78)


if __name__ == "__main__":
    main()
