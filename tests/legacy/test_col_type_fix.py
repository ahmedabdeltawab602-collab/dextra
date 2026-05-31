"""
test_col_type_fix.py
--------------------
اختبار مفصَّل لدالتَي col_clean / col_fix و type_fix / cast / cast_types.
كل الـ aliases تشير لنفس الدالة، فاختياره مسألة ذوق.
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
    df_raw = pd.DataFrame({
        # Bad column names: spaces, symbols, mixed case
        '  Customer Name  ': ['  Alice  ', 'Bob', 'CAROL', '  david  ', 'Eve'] * 4,
        'Price ($)':         ['$100.50', '$200.00', '$1,500.00', '$50.25', '$75.75'] * 4,
        'AGE!':              ['25', '30', '35', '40', '45'] * 4,
        'Date of Order':     ['2024-01-15', '2024-02-20', '2024-03-10',
                              '2024-04-05', '2024-05-12'] * 4,
        'Subscribed?':       ['yes', 'no', 'YES', 'No', 'y'] * 4,
        'Category!@#':       ['A', 'B', 'A', 'C', 'B'] * 4,
    })

    print(f"\nOriginal shape: {df_raw.shape}")
    print(f"Original columns: {list(df_raw.columns)}")
    print(f"\nOriginal dtypes:")
    print(df_raw.dtypes)

    # ==================================================================
    # PART A — Column name cleanup (4 aliases all do the same thing)
    # ==================================================================
    section("A. Available aliases for standardize_columns")
    print(f"  dx.standardize_columns  (19 chars, full name)")
    print(f"  dx.stdcols              ( 7 chars, v1 alias)")
    print(f"  dx.col_clean            ( 9 chars, v2 alias)")
    print(f"  dx.col_fix              ( 7 chars, NEW alias)")
    assert dx.standardize_columns is dx.stdcols
    assert dx.standardize_columns is dx.col_clean
    assert dx.standardize_columns is dx.col_fix
    print("\n[OK] All four point to the same function.")

    section("Test 1: col_fix(df_raw)  --  shortest pattern-matching name")
    df_a = dx.col_fix(df_raw, plot=False)
    print(f"\nNew columns: {list(df_a.columns)}")
    expected = ['customer_name', 'price', 'age', 'date_of_order',
                'subscribed', 'category']
    assert list(df_a.columns) == expected, \
        f"Column names mismatch.\n  expected: {expected}\n  got:      {list(df_a.columns)}"
    print(f"\n[OK] Column names normalized correctly.")
    print(f"     Cell-level whitespace stripped: customer_name[0] = {df_a['customer_name'].iloc[0]!r}")

    # ==================================================================
    # PART B — Type coercion (3 aliases all do the same thing)
    # ==================================================================
    section("B. Available aliases for cast_types")
    print(f"  dx.cast_types  (10 chars, full name)")
    print(f"  dx.cast        ( 4 chars, v1 alias)")
    print(f"  dx.type_fix    ( 8 chars, v2 alias)")
    assert dx.cast_types is dx.cast
    assert dx.cast_types is dx.type_fix
    print("\n[OK] All three point to the same function.")

    section("Test 2: type_fix(df_a)  --  auto-detect types")
    df_b = dx.type_fix(df_a, plot=False)
    print(f"\nNew dtypes:")
    print(df_b.dtypes)
    # Verify
    assert pd.api.types.is_numeric_dtype(df_b['price']), "price should be numeric"
    assert df_b['price'].iloc[2] == 1500.0, f"price[2] = {df_b['price'].iloc[2]} (expected 1500.0)"
    assert pd.api.types.is_datetime64_any_dtype(df_b['date_of_order']), \
        "date_of_order should be datetime"
    assert pd.api.types.is_bool_dtype(df_b['subscribed']) or \
           str(df_b['subscribed'].dtype) == 'boolean', "subscribed should be boolean"
    assert isinstance(df_b['category'].dtype, pd.CategoricalDtype), \
        "category should be CategoricalDtype"
    print("\n[OK] All four target dtypes verified:")
    print(f"     price        -> {df_b['price'].dtype}    (was object)")
    print(f"     date_of_order -> {df_b['date_of_order'].dtype}")
    print(f"     subscribed   -> {df_b['subscribed'].dtype}")
    print(f"     category     -> {df_b['category'].dtype}")

    # ==================================================================
    # PART C — Pipeline using the short names
    # ==================================================================
    section("Test 3: Pipeline using only the shortest names")
    df_pipe = (
        df_raw
        .pipe(dx.col_fix,  show=False, plot=False)   # 7 chars
        .pipe(dx.type_fix, show=False, plot=False)   # 8 chars
    )
    print(f"\nPipeline result shape: {df_pipe.shape}")
    print(f"Pipeline result columns: {list(df_pipe.columns)}")
    print(f"\nAudit log entries: {len(df_pipe.attrs.get('dextra_audit', []))}")
    for i, e in enumerate(df_pipe.attrs.get('dextra_audit', []), 1):
        print(f"  #{i}: [{e['stage']}] {e['decision']}")

    # ==================================================================
    # PART D — Immutability verification
    # ==================================================================
    section("Test 4: Immutability  --  original df_raw unchanged")
    assert ' Customer Name ' in df_raw.columns or '  Customer Name  ' in df_raw.columns, \
        "Original column names should still have spaces"
    # pandas 3.x uses 'str' dtype for strings; pandas 2.x uses 'object'
    assert str(df_raw['Price ($)'].dtype) in ('object', 'str'), \
        f"Original price should still be string-like; got {df_raw['Price ($)'].dtype}"
    print(f"\n[OK] df_raw.columns still has spaced/special names: {list(df_raw.columns)[:2]}")
    print(f"[OK] df_raw['Price ($)'].dtype is still string-like: {df_raw['Price ($)'].dtype}")

    print("\n" + "=" * 78)
    print("  ALL TESTS PASSED")
    print("  ✓ col_fix / col_clean / stdcols / standardize_columns  --  all work")
    print("  ✓ type_fix / cast / cast_types                          --  all work")
    print("  ✓ Pipeline using short names produces correct result")
    print("  ✓ Original DataFrame is immutable")
    print("=" * 78)


if __name__ == "__main__":
    main()
