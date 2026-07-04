"""Session-C gate: public docstrings are complete.

Every unique public callable exported via ``dextra.__all__`` must carry a
docstring of at least 4 lines that shows usage (an ``Examples`` section or
a ``>>>`` doctest line). This is the enforceable floor of the 0.6.0
docstring audit -- it keeps future public API from shipping bare.
"""
import inspect

import dextra as dx


def _unique_public_callables():
    seen = {}
    for name in sorted(dx.__all__):
        obj = getattr(dx, name, None)
        if callable(obj) and id(obj) not in seen:
            seen[id(obj)] = (name, obj)
    return list(seen.values())


def test_every_public_callable_has_a_substantial_docstring():
    offenders = []
    for name, obj in _unique_public_callables():
        doc = inspect.getdoc(obj) or ""
        if len(doc.splitlines()) < 4:
            offenders.append(f"{name}: <4 lines")
    assert not offenders, offenders


def test_every_public_callable_shows_usage():
    offenders = []
    for name, obj in _unique_public_callables():
        doc = inspect.getdoc(obj) or ""
        if "Examples" not in doc and ">>>" not in doc:
            offenders.append(name)
    assert not offenders, offenders
