"""Macro-benchmark: dextra on ~1M rows (load -> clean -> features).

Times the documented one-liner workflow on a synthetic million-row frame:
``dx.load`` (CSV + measured type inference), the cleaning chain
(``audit`` / ``tidycols`` / ``recast`` / ``impute`` / ``dedup`` / ``winsor`` /
``verify``) and the Phase 4 feature pipeline (``featpipe``: scale + one-hot),
all in silent mode (show=False, plot=False). Single cold pass per step.

Run on your machine (writes a temp CSV, removed afterwards):

    python benchmarks/macro_bench.py                 # 1,000,000 rows
    python benchmarks/macro_bench.py --rows 100000   # quicker sanity run

Prints a ready-to-paste markdown table; results live in docs/benchmarks.md.
Complements benchmarks/test_bench.py (pytest-benchmark micro-benchmarks,
tracked in CI).
"""
from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import dextra as dx

KW = dict(show=False, plot=False)


def make_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic but realistically dirty frame: messy column names,
    padded / case-varying categories, date and yes/no strings, ~2% NaN,
    ~0.1% extreme outliers and ~1% duplicate rows."""
    cities = ["Cairo", "Giza", "Alex", " Cairo ", "giza", "Luxor"]
    dates = (np.datetime64("2022-01-01")
             + rng.integers(0, 1000, n).astype("timedelta64[D]"))
    df = pd.DataFrame({
        "Customer ID": np.arange(n, dtype=np.int64),
        " Age ": rng.normal(40, 12, n).round(1),
        "Income($)": rng.lognormal(9.0, 0.6, n).round(2),
        "score": rng.normal(50.0, 15.0, n).round(3),
        "City": rng.choice(cities, n),
        "signup_date": dates.astype(str),
        "active": rng.choice(["yes", "no"], n),
    })
    df.loc[rng.choice(n, max(1, n // 50), replace=False), " Age "] = np.nan
    df.loc[rng.choice(n, max(1, n // 50), replace=False), "Income($)"] = np.nan
    out_idx = rng.choice(n, max(1, n // 1000), replace=False)
    df.loc[out_idx, "Income($)"] = float(df["Income($)"].max()) * 50.0
    dup = df.sample(max(1, n // 100), random_state=0)
    return pd.concat([df, dup], ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="dextra macro-benchmark (~1M rows: load/clean/features)")
    ap.add_argument("--rows", type=int, default=1_000_000,
                    help="base row count before the ~1%% duplicates")
    args = ap.parse_args(argv)

    print(f"dextra {dx.__version__} | python {platform.python_version()} | "
          f"pandas {pd.__version__} | numpy {np.__version__} | "
          f"{platform.system()} | {os.cpu_count()} logical CPUs")

    t0 = time.perf_counter()
    df = make_frame(args.rows, np.random.default_rng(0))
    csv_path = os.path.join(tempfile.gettempdir(), "dextra_macro_bench.csv")
    df.to_csv(csv_path, index=False)
    print(f"setup: {len(df):,} rows x {df.shape[1]} cols -> "
          f"{os.path.getsize(csv_path) / 1e6:.0f} MB CSV "
          f"({time.perf_counter() - t0:.1f}s, excluded from timings)\n")

    results = []

    def run(step, n_in, fn):
        gc.collect()
        t = time.perf_counter()
        try:
            out = fn()
            results.append((step, time.perf_counter() - t, n_in, ""))
            return out
        except Exception as e:  # keep timing the remaining steps
            results.append((step, time.perf_counter() - t, n_in,
                            f"FAILED - {type(e).__name__}: {e}"))
            return None

    loaded = run("load (csv, measured inference)", len(df),
                 lambda: dx.load(csv_path, show=False))
    if loaded is None:
        loaded = df
    run("audit (clean_report)", len(loaded),
        lambda: dx.audit(loaded, return_df=True, **KW))
    tidy = run("tidycols (standardize_columns)", len(loaded),
               lambda: dx.tidycols(loaded, **KW))
    if tidy is None:
        tidy = loaded
    cast = run("recast (cast_types, auto)", len(tidy),
               lambda: dx.recast(tidy, **KW))
    if cast is None:
        cast = tidy
    imp = run("impute (handle_missing, auto)", len(cast),
              lambda: dx.impute(cast, **KW))
    if imp is None:
        imp = cast
    ded = run("dedup (dedupe)", len(imp), lambda: dx.dedup(imp, **KW))
    if ded is None:
        ded = imp
    win = run("winsor (clip_outliers, income)", len(ded),
              lambda: dx.winsor(ded, cols=["income"], **KW))
    if win is None:
        win = ded
    run("verify (validate_rules, 2 rules)", len(win), lambda: dx.verify(
        win,
        rules=[{"name": "income_pos", "check": "income >= 0"},
               {"name": "age_range", "check": "age.between(0, 120)"}],
        return_df=True, **KW))
    run("featpipe (scale + one-hot)", len(win), lambda: dx.featpipe(
        win,
        steps=[{"fn": "scale", "cols": ["age", "income", "score"],
                "method": "standard"},
               {"fn": "encode", "cols": ["city"], "method": "onehot"}],
        **KW))

    print("| step | seconds | rows in | rows/s |")
    print("|---|---:|---:|---:|")
    for step, secs, n_in, note in results:
        ok = not note and secs > 0 and n_in
        rate = f"{n_in / secs:,.0f}" if ok else "-"
        label = f"{step} ({note})" if note else step
        print(f"| {label} | {secs:.2f} | {n_in:,} | {rate} |")

    os.remove(csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
