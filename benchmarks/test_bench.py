"""Micro-benchmarks for dextra hot paths (Phase 6.5 performance sprint).

Run locally / in CI with:  pytest benchmarks/ --benchmark-only
These are NOT part of the default test run (testpaths = ["tests"]); they exist
to track performance over time and to catch regressions. Requires
``pytest-benchmark`` and the ``ml`` extra (scikit-learn).
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg", force=True)
pytest.importorskip("sklearn")
pytest.importorskip("pytest_benchmark")

import dextra as dx  # noqa: E402

KW = dict(show=False, plot=False)
NUM = [f"x{i}" for i in range(6)]


@pytest.fixture(scope="module")
def big():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({c: rng.normal(size=n) for c in NUM})
    df["price"] = df[NUM].sum(axis=1) + rng.normal(scale=2, size=n)
    df["churn"] = np.where(df["x0"] > 0, "yes", "no")
    return df


def test_bench_describe_numeric(benchmark, big):
    benchmark(lambda: dx.describe_numeric(big, show=False, return_df=True))


def test_bench_scale(benchmark, big):
    benchmark(lambda: dx.scale(big, cols=NUM, method="standard", **KW))


def test_bench_regress_forest(benchmark, big):
    benchmark(lambda: dx.regress(big, y="price", cols=NUM, method="forest", **KW))


def test_bench_cluster_kmeans(benchmark, big):
    benchmark(lambda: dx.cluster(big, cols=NUM, method="kmeans", k=4, **KW))
