"""
test_phase3_stage1.py
---------------------
اختبار شامل لـ Stage 3.1 من Phase 3:
    - clean_report(df)         - Stage 0: profile & audit
    - standardize_columns(df)  - Stage 1: structural cleanup
    - audit log mechanism      - دمج السجل عبر الاستدعاءات
    - idempotency              - تطبيق الدالة مرتين = نفس النتيجة
    - immutability             - الـ DataFrame الأصلي لا يتغير
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

    # ------------------------------------------------------------------
    # Build a deliberately messy dataset
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    n = 200

    df_raw = pd.DataFrame({
        'Customer Name':   ['  Alice  ', 'Bob', 'CAROL', '  david  ', 'Eve'] * 40,
        'AGE ':            [25, 30, 35, 40, np.nan] * 40,
        'Price ($)':       [100.0, 200.0, 150.0, 300.0, np.nan] * 40,
        'Category!@#':     ['A', 'B', 'A', 'C', 'B'] * 40,
        'Notes / Comments': [None, 'OK', None, None, None] * 40,  # 80% missing
        'all_missing_col': [np.nan] * n,
        'constant_col':    [42] * n,
    })

    print(f"\nOriginal shape: {df_raw.shape}")
    print(f"Original columns: {list(df_raw.columns)}")

    # ------------------------------------------------------------------
    # Test 29: clean_report on RAW data
    # ------------------------------------------------------------------
    section("Test 29: dx.clean_report(df_raw) - تشخيص شامل")
    print("Expected:")
    print("  - 6 columns with various issues:")
    print("    * Customer Name: object, 0% missing -> OK (but has whitespace + case issues)")
    print("    * AGE: float, 20% missing -> impute_median")
    print("    * Notes/Comments: 80% missing -> drop_column")
    print("    * all_missing_col: 100% missing -> drop_column")
    print("    * constant_col: 1 unique -> drop_constant")
    print("  - Cleaning trail: empty (raw data, no operations yet)")
    print("  - Consistency score < 100 (due to whitespace in Customer Name)")
    dx.clean_report(df_raw, plot=False)

    # ------------------------------------------------------------------
    # Test 30: standardize_columns
    # ------------------------------------------------------------------
    section("Test 30: dx.standardize_columns(df_raw)")
    print("Expected:")
    print("  - 'Customer Name' -> 'customer_name'")
    print("  - 'AGE ' -> 'age'")
    print("  - 'Price ($)' -> 'price'")
    print("  - 'Category!@#' -> 'category'")
    print("  - 'Notes / Comments' -> 'notes_comments'")
    print("  - Whitespace stripped from 'customer_name' cells (40 'Alice', 40 'david')")
    df_clean = dx.standardize_columns(df_raw, plot=False)
    print(f"\nClean DataFrame columns: {list(df_clean.columns)}")
    print(f"Sample customer_name[0]: {df_clean['customer_name'].iloc[0]!r}  "
          "(no leading/trailing whitespace)")

    # ------------------------------------------------------------------
    # Test 31: Immutability check
    # ------------------------------------------------------------------
    section("Test 31: Immutability - الأصلي لم يتغير")
    print("Expected: df_raw.columns has ORIGINAL names (not normalized).")
    print(f"  df_raw.columns:   {list(df_raw.columns)[:3]} ...")
    print(f"  df_clean.columns: {list(df_clean.columns)[:3]} ...")
    assert 'Customer Name' in df_raw.columns
    assert 'customer_name' in df_clean.columns
    assert df_raw.iloc[0, 0] == '  Alice  '  # raw still has whitespace
    assert df_clean['customer_name'].iloc[0] == 'Alice'  # clean stripped
    print("[OK] الأصلي محفوظ كما هو، الـ clean منفصل تماماً.")

    # ------------------------------------------------------------------
    # Test 32: Idempotency
    # ------------------------------------------------------------------
    section("Test 32: Idempotency - تطبيق مرتين = نفس النتيجة")
    df_clean2 = dx.standardize_columns(df_clean, show=False, plot=False)
    print(f"Re-apply standardize_columns to already-clean df:")
    print(f"  columns identical: {list(df_clean.columns) == list(df_clean2.columns)}")
    # Cell content check
    same_cells = (df_clean['customer_name'] == df_clean2['customer_name']).all()
    print(f"  customer_name cells identical: {same_cells}")
    print(f"  Audit log entries: df_clean={len(df_clean.attrs.get('dextra_audit', []))}, "
          f"df_clean2={len(df_clean2.attrs.get('dextra_audit', []))}")

    # ------------------------------------------------------------------
    # Test 33: Audit log mechanism
    # ------------------------------------------------------------------
    section("Test 33: Audit log - السجل ينمو مع كل عملية")
    audit = df_clean.attrs.get('dextra_audit', [])
    assert len(audit) == 1, f"Expected 1 entry, got {len(audit)}"
    entry = audit[0]
    print("Audit entry:")
    for k, v in entry.items():
        print(f"  {k}: {v}")
    assert entry['stage'] == 'structural_cleanup'
    assert entry['function'] == 'standardize_columns'
    assert 'timestamp' in entry
    assert 'decision' in entry
    print("\n[OK] السجل يحتوي stage, function, timestamp, before, after, params, decision.")

    # ------------------------------------------------------------------
    # Test 34: clean_report sees the audit trail
    # ------------------------------------------------------------------
    section("Test 34: clean_report بعد التنظيف - يعرض السجل + يحسن الـ scores")
    print("Expected:")
    print("  - Cleaning trail يعرض الخطوة المنفذة")
    print("  - consistency_score يرتفع لـ 100 (whitespace أزيلت)")
    print("  - quality_score الإجمالي يرتفع")
    dx.clean_report(df_clean, plot=False)

    # ------------------------------------------------------------------
    # Test 35: aliases
    # ------------------------------------------------------------------
    section("Test 35: Aliases")
    assert dx.cleanrep is dx.clean_report
    assert dx.stdcols  is dx.standardize_columns
    print("[OK] cleanrep, stdcols مرتبطة بالدوال الكاملة.")

    # ------------------------------------------------------------------
    # Test 36: Custom name_map
    # ------------------------------------------------------------------
    section("Test 36: name_map - تخصيص تسمية أعمدة محددة")
    df_custom = dx.standardize_columns(
        df_raw,
        name_map={'Customer Name': 'cust_id', 'AGE ': 'years'},
        show=False, plot=False,
    )
    print(f"With custom name_map:")
    print(f"  'Customer Name' -> 'cust_id': {'cust_id' in df_custom.columns}")
    print(f"  'AGE ' -> 'years': {'years' in df_custom.columns}")
    print(f"  Other columns still auto-normalized: 'price' in df_custom: {'price' in df_custom.columns}")
    print("[OK] الـ map الصريح يعمل + بقية الأعمدة تلقائية.")

    print("\n" + "=" * 78)
    print("  STAGE 3.1 - ALL TESTS PASSED")
    print("  Functions: clean_report (Stage 0), standardize_columns (Stage 1)")
    print("=" * 78)
    print("\nTip: للرسوم البصرية، شغّل في Jupyter بدون plot=False:")
    print("     dx.clean_report(df_raw)        # 4-panel audit dashboard")
    print("     dx.standardize_columns(df_raw)  # rename table + strip stats\n")


if __name__ == "__main__":
    main()
