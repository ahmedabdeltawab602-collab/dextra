"""Shared pytest fixtures."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

# Run tests without a GUI backend so plt.show() is a no-op.
matplotlib.use("Agg", force=True)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def numeric_df(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "a": rng.normal(10, 2, 200),
        "b": rng.normal(0, 1, 200),
        "c": rng.integers(1, 100, 200).astype(float),
    })


@pytest.fixture
def mixed_df(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "num": rng.normal(0, 1, 50),
        "cat": rng.choice(["x", "y", "z"], 50),
        "with_nans": np.concatenate([rng.normal(0, 1, 45), [np.nan] * 5]),
    })
