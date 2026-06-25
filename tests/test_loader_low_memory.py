"""m-8: ``low_memory=True`` is a pure performance flag for delimited text.

The streamed header / ragged-row path must return a frame **and** a load plan
that are identical to the default full-parse path; only the transient peak
memory differs (measured separately by ``eval_scratch/eval_39_mem.py``). These
tests lock that equivalence so the low-memory path can never silently diverge
from the loader's measured-inference contract.
"""
import json
import warnings

import pandas as pd
import pytest

import dextra as dx


def _norm_plan(plan):
    """Plan JSON minus the wall-clock stamp, for order-insensitive equality."""
    p = dict(plan)
    p.pop("generated_at", None)
    return json.dumps(p, sort_keys=True, default=str)


def _load(src, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dx.load(src, show=False, return_params=True,
                       on_ambiguous="warn", **kw)


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


_CLEAN = "\n".join(
    ["id,amount,city,when,active"]
    + ["%d,%.1f,%s,2024-01-0%d,%s" % (i, i * 1.5,
       ["Cairo", "Giza", " Alex "][i % 3], i % 9 + 1, ["yes", "no"][i % 2])
       for i in range(60)]
)

# Header not at row 0, plus two ragged rows: exercises the streamed header
# detection and the per-row field-count tally that replaces the full
# list-of-lists in low-memory mode.
_PREAMBLE_RAGGED = (
    "# export 2024-01\n"
    "# region: all\n"
    "id,amount,city\n"
    "1,10,Cairo\n"
    "2,20,Giza,EXTRA\n"
    "3,30\n"
    "4,40,Luxor\n"
)

# Leading-zero identifiers must stay text in both modes (B-1 guard).
_LEADING_ZERO = "code,n\n007,1\n012,2\n034,3\n"


@pytest.mark.parametrize("name,text", [
    ("clean.csv", _CLEAN),
    ("preamble.csv", _PREAMBLE_RAGGED),
    ("zero.csv", _LEADING_ZERO),
])
def test_low_memory_frame_and_plan_identical(tmp_path, name, text):
    src = _write(tmp_path, name, text)
    base_df, base_plan = _load(src)
    low_df, low_plan = _load(src, low_memory=True)
    pd.testing.assert_frame_equal(base_df, low_df)
    assert _norm_plan(base_plan) == _norm_plan(low_plan)


def test_low_memory_replay_roundtrips(tmp_path):
    # A plan captured in low-memory mode replays to the same frame.
    src = _write(tmp_path, "clean.csv", _CLEAN)
    low_df, low_plan = _load(src, low_memory=True)
    replayed = dx.load(src, show=False, params=low_plan)
    pd.testing.assert_frame_equal(replayed, low_df)


def test_low_memory_accepted_and_noop_for_in_memory_frame():
    # In-memory frames take the typed pass-through path; low_memory is accepted
    # (keyword-only) and simply has nothing to optimise there.
    df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})
    out = dx.load(df, show=False, low_memory=True)
    assert list(out.columns) == ["a", "b"]
    assert len(out) == 3
