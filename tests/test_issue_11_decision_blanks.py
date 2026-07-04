"""Regression for issue #11.

The load *decision sentence* must count blank cells that were silently
coerced to NaN (whitespace fields that look like data), instead of hiding
them inside ``null_%`` while reporting ``0 cell(s) failed``.
"""
import dextra as dx
from dextra import _loader


def _write(p, text):
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_issue_11_decision_sentence_counts_coerced_blanks(tmp_path):
    # 'amount' has two whitespace-only cells that coerce to NaN.
    path = _write(
        tmp_path / "blanks.csv",
        "id,amount\n1,10.5\n2, \n3,20.0\n4, \n5,30\n",
    )
    _out, plan = dx.load(path, return_params=True, show=False)
    sentence = _loader._decision_sentence(plan)
    assert "2 blank" in sentence, sentence
    assert "blank cell(s)" in sentence, sentence
    # the two blanks are NOT miscounted as parse failures
    assert "0 cell(s) failed" in sentence, sentence


def test_issue_11_no_blank_clause_when_clean(tmp_path):
    path = _write(tmp_path / "clean.csv", "id,amount\n1,10.5\n2,20.0\n3,30\n")
    _out, plan = dx.load(path, return_params=True, show=False)
    sentence = _loader._decision_sentence(plan)
    assert "blank cell(s)" not in sentence, sentence
