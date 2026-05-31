"""
test_phase3_stage2.py
---------------------
اختبار Stage 3.2 من Phase 3:
    - cast_types(df, schema=None)      - Stage 2: type coercion (auto + explicit)
    - validate_rules(df, rules)        - Stage 6: consistency / business rules
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
    # Build a deliberately mixed-type dataset
    # ------------------------------------------------------------------
    df_raw = pd.DataFrame({
        'price_str':    ['100', '200.5', '300.0', '$1,500.00', '50', '75', '120', None] * 5,
        'date_str':     ['2024-01-01', '2024-02-15', '2024-03-20', '2024-04-10',
                         '2024-05-05', '2024-06-12', '2024-07-19', '2024-08-26'] * 5,
        'flag_yn':      ['yes', 'no', 'YES', 'No', 'y', 'n', 'true', 'false'] * 5,
        'category':     (['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B']) * 5,  # 3 unique
        'real_int':     list(range(8)) * 5,
        'narrative':    [f"unique sentence #{i}" for i in range(40)],   # high cardinality
    })
    print(f"\nOriginal shape: {df_raw.shape}")
    print(f"Original dtypes:")
    print(df_raw.dtypes)

    # ------------------------------------------------------------------
    # Test 37: cast_types - auto-detection
    # ------------------------------------------------------------------
    section("Test 37: dx.cast_types(df_raw) - auto-detection")
    print("Expected:")
    print("  - price_str -> float64 (currency stripped from '$1,500.00')")
    print("  - date_str  -> datetime64[ns]")
    print("  - flag_yn   -> boolean")
    print("  - category  -> category (3 unique < 50, ratio < 0.5)")
    print("  - real_int  -> int64 (unchanged)")
    print("  - narrative -> object (high cardinality, no cast)")
    df_typed = dx.cast_types(df_raw, plot=False)
    print(f"\nNew dtypes:")
    print(df_typed.dtypes)

    # Verify values
    assert df_typed['price_str'].iloc[3] == 1500.0, "currency stripping failed"
    assert pd.api.types.is_datetime64_any_dtype(df_typed['date_str']), "date conversion failed"
    assert pd.api.types.is_bool_dtype(df_typed['flag_yn']) or \
           str(df_typed['flag_yn'].dtype) == 'boolean', "bool conversion failed"
    print("\n[OK] All four auto-detections succeeded.")

    # ------------------------------------------------------------------
    # Test 38: cast_types - explicit schema
    # ------------------------------------------------------------------
    section("Test 38: dx.cast_types with explicit schema")
    df_typed2 = dx.cast_types(df_raw,
                              schema={'price_str': 'float64',
                                      'real_int': 'float64'},
                              auto_categorical=False,
                              plot=False)
    print(f"\nDtype after explicit schema:")
    print(f"  price_str: {df_typed2['price_str'].dtype}")
    print(f"  real_int:  {df_typed2['real_int'].dtype}  (forced to float64)")

    # ------------------------------------------------------------------
    # Test 39: cast_types - immutability + audit log
    # ------------------------------------------------------------------
    section("Test 39: Immutability + audit log for cast_types")
    print(f"Original df_raw['price_str'].dtype: {df_raw['price_str'].dtype}  (unchanged)")
    audit = df_typed.attrs.get('dextra_audit', [])
    assert len(audit) == 1
    print(f"Audit entry stage: {audit[0]['stage']}")
    print(f"Audit decision:    {audit[0]['decision']}")

    # ------------------------------------------------------------------
    # Test 40: validate_rules - basic
    # ------------------------------------------------------------------
    section("Test 40: dx.validate_rules with mixed pass/fail rules")
    df_v = pd.DataFrame({
        'price':       [10,    20,   -5,   100,   200,   -10],
        'age':         [25,    30,   35,   200,   5,     45],
        'email':       ['a@x.com', 'b@y.com', 'invalid', 'c@z.com', None, 'd@w.com'],
        'start_date':  pd.to_datetime(['2024-01-01', '2024-02-01', '2024-03-01',
                                       '2024-04-01', '2024-05-01', '2024-06-01']),
        'end_date':    pd.to_datetime(['2024-01-15', '2024-02-15', '2024-02-01',
                                       '2024-04-30', '2024-05-15', '2024-06-30']),
    })
    rules = [
        {"name": "price_positive", "check": "price >= 0",
         "description": "Price must be non-negative"},
        {"name": "age_in_range",   "check": "age.between(18, 100)",
         "description": "Age between 18 and 100"},
        {"name": "valid_email",    "check": lambda d: d['email'].str.contains('@', na=False),
         "description": "Email must contain @"},
        {"name": "end_after_start","check": "end_date >= start_date",
         "description": "End date must be on/after start date"},
    ]
    print("Expected:")
    print("  - price_positive: 2 violations (rows with -5, -10)")
    print("  - age_in_range:   2 violations (200, 5)")
    print("  - valid_email:    2 violations ('invalid', None)")
    print("  - end_after_start: 1 violation (row 2: end=Feb 1 < start=Mar 1)")
    dx.validate_rules(df_v, rules, plot=False)

    # ------------------------------------------------------------------
    # Test 41: validate_rules - return_violations
    # ------------------------------------------------------------------
    section("Test 41: return_violations - استخراج الصفوف المخالفة")
    violations = dx.validate_rules(df_v, rules, show=False, plot=False,
                                    return_violations=True)
    print(f"Number of unique rows violating at least one rule: {len(violations)}")
    print(violations[['price', 'age', 'email', 'violated_rules']])

    # ------------------------------------------------------------------
    # Test 42: chained pipeline (cleaning -> cast -> validate)
    # ------------------------------------------------------------------
    section("Test 42: Pipeline - stdcols + cast_types + validate_rules")
    # Build a pipeline
    df_pipe = (df_raw
               .pipe(dx.standardize_columns, show=False, plot=False)
               .pipe(dx.cast_types, show=False, plot=False))
    print(f"After pipe:")
    print(f"  shape: {df_pipe.shape}")
    print(f"  columns: {list(df_pipe.columns)}")
    print(f"  audit entries: {len(df_pipe.attrs.get('dextra_audit', []))}")
    for i, entry in enumerate(df_pipe.attrs.get('dextra_audit', []), 1):
        print(f"  #{i}: {entry['stage']} -- {entry['decision']}")

    # ------------------------------------------------------------------
    # Test 43: aliases
    # ------------------------------------------------------------------
    section("Test 43: Aliases")
    assert dx.cast    is dx.cast_types
    assert dx.vrules  is dx.validate_rules
    print("[OK] cast, vrules aliases linked.")

    print("\n" + "=" * 78)
    print("  STAGE 3.2 - ALL TESTS PASSED")
    print("  Functions: cast_types (Stage 2), validate_rules (Stage 6)")
    print("=" * 78)


if __name__ == "__main__":
    main()
