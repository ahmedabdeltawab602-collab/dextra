"""Smart, transparent data loader for dextra - Phase 11 (the entry layer).

See ``LOADER_PHILOSOPHY.md`` and ``LOADER_SPEC_11_1.md``. Stage 11.1 covers
delimited text (csv/tsv) plus a typed pass-through for in-memory frames.

Governing principle: *transparency scales with uncertainty*. Confident parses
load in one line and are fully disclosed; ambiguous decisions are flagged and,
under the chosen policy, warned about / raised / returned-as-a-plan rather than
guessed silently. Every load emits a JSON-safe, replayable **load plan** (the
unified-contract ``params`` artifact). The source is never modified.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ._utils import _ensure_pandas, append_audit, get_variable_name, json_safe, now_iso
from ._version import __version__

# ---------------------------------------------------------------------------
# Exceptions & warning
# ---------------------------------------------------------------------------


class DextraLoaderError(Exception):
    """Base class for loader errors."""


class LoaderSecurityError(DextraLoaderError):
    """Raised when a source is refused for security reasons (e.g. pickle)."""


class LoaderAmbiguityError(DextraLoaderError):
    """Raised under ``on_ambiguous='raise'`` when a parse decision is ambiguous."""


class LoaderReplayError(DextraLoaderError):
    """Raised when a replay plan no longer matches its source."""


class LoaderAbort(DextraLoaderError):
    """Raised when an interactive user aborts the load."""


class DextraLoaderWarning(UserWarning):
    """Warning emitted under ``on_ambiguous='warn'`` listing ambiguous decisions."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PICKLE_EXT = {".pkl", ".pickle"}
_TSV_EXT = {".tsv", ".tab"}
_BOOL_TOKENS = {"true", "false", "yes", "no", "y", "n", "t", "f", "1", "0"}
_TRUE_TOKENS = {"true", "yes", "y", "t", "1"}
_HIGH_RISK_RE = re.compile(r"(^id$|_id$|^key$|key$|target|label|^y$)", re.IGNORECASE)
_DATE_FORMATS = ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%m/%d/%Y",
                 "%Y/%m/%d", "%d-%m-%Y")
_CONFIRMED = "confirmed"
_AMBIGUOUS = "ambiguous"
_HIGH_RISK = "ambiguous-high-risk"
_PARSE_ACCEPT = 0.95  # min parse-rate to accept a non-text dtype


# ---------------------------------------------------------------------------
# Source reading
# ---------------------------------------------------------------------------

def _read_bytes(source):
    """Return ``(raw_bytes, source_meta)`` from a path / file-like object."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        ext = os.path.splitext(path)[1].lower()
        if ext in _PICKLE_EXT:
            raise LoaderSecurityError(
                f"refusing to auto-load pickle {path!r}: pickle can execute "
                "arbitrary code. Pass allow_pickle=True only for trusted files.")
        with open(path, "rb") as fh:
            raw = fh.read()
        try:
            stat = os.stat(path)
            size, mtime = int(stat.st_size), float(stat.st_mtime)
        except OSError:
            size, mtime = None, None
        meta = {"name": os.path.basename(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": size, "mtime": mtime}
        return raw, meta
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        meta = {"name": getattr(source, "name", "<stream>"),
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data), "mtime": None}
        return data, meta
    raise TypeError(
        f"load: source must be a path, file-like object, or DataFrame; "
        f"got {type(source).__name__}.")


# ---------------------------------------------------------------------------
# Detection: encoding / decode / delimiter / header
# ---------------------------------------------------------------------------

def _detect_encoding(sample: bytes, forced: Optional[str]):
    if forced:
        return forced, _CONFIRMED, f"user-specified ({forced})"
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", _CONFIRMED, "BOM: utf-8"
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16", _CONFIRMED, "BOM: utf-16"
    try:
        from charset_normalizer import from_bytes  # lazy `io` extra
        best = from_bytes(sample).best()
        if best is not None and best.encoding:
            enc = "utf-8" if best.encoding.replace("_", "-") == "utf-8" else best.encoding
            return enc, _CONFIRMED, f"charset-normalizer: {enc}"
    except Exception:  # noqa: BLE001 - fall back to stdlib probing
        pass
    try:
        sample.decode("utf-8")
        return "utf-8", _CONFIRMED, "strict utf-8 decode of sample"
    except UnicodeDecodeError:
        pass
    try:
        sample.decode("cp1256")
        return "cp1256", _AMBIGUOUS, "fallback: cp1256 decoded (utf-8 failed)"
    except UnicodeDecodeError:
        return "latin-1", _AMBIGUOUS, "fallback: latin-1 (lossless byte map)"


def _decode_full(raw: bytes, enc: str, conf: str, reason: str):
    """Decode the whole payload; downgrade to latin-1 if strict decode fails."""
    try:
        return raw.decode(enc), conf, reason
    except (UnicodeDecodeError, LookupError):
        return (raw.decode("latin-1"), _AMBIGUOUS,
                f"{reason}; strict decode failed beyond sample -> latin-1")


def _detect_delimiter(sample_text: str, forced: Optional[str], exclude=()):
    if forced:
        return forced, _CONFIRMED, f"user-specified ({forced!r})"
    candidates = [c for c in (",", ";", "\t", "|") if c not in exclude]
    try:
        from clevercsv import Sniffer as _CleverSniffer  # lazy `io` extra
        dialect = _CleverSniffer().sniff(sample_text)
        if (dialect is not None and dialect.delimiter
                and dialect.delimiter not in exclude):
            return dialect.delimiter, _CONFIRMED, "clevercsv sniffer"
    except Exception:  # noqa: BLE001 - fall back to stdlib csv.Sniffer
        pass
    head = "\n".join(sample_text.splitlines()[:10])
    sep = None
    if candidates:
        try:
            sep = csv.Sniffer().sniff(head, delimiters="".join(candidates)).delimiter
        except csv.Error:
            sep = None
    if sep is None:
        present = [c for c in candidates if head.count(c) > 0]
        if not present:
            for cand in ("\t", "|", ";", ",", "\x01"):
                if cand not in exclude and cand not in head:
                    return cand, _CONFIRMED, "single column (no delimiter detected)"
            return "\x01", _CONFIRMED, "single column"
        sep = max(present, key=head.count)
    rows = [r for r in sample_text.splitlines()[:30] if r.strip()]
    counts = [len(r.split(sep)) for r in rows]
    stable = len(set(counts)) == 1 if counts else False
    conf = _CONFIRMED if stable else _AMBIGUOUS
    reason = ("delimiter field count stable" if stable
              else "delimiter field count varies across rows")
    return sep, conf, reason


def _detect_header(rows, forced: Optional[int]):
    """Pick the header row index among parsed ``rows`` (list of field lists)."""
    if forced is not None:
        return int(forced), _CONFIRMED, f"user-specified (row {forced})"
    probe = [r for r in rows[:20]]
    if not probe:
        return 0, _AMBIGUOUS, "empty input -> row 0"
    counts = [len(r) for r in probe]
    modal = max(set(counts), key=counts.count)
    for i, r in enumerate(probe):
        if len(r) != modal:
            continue
        cells = [c for c in r if str(c).strip() != ""]
        if not cells:
            continue
        non_numeric = sum(0 if _looks_numeric(c) else 1 for c in cells)
        if non_numeric >= max(1, len(cells) // 2):
            stable = counts.count(modal) >= max(2, len(counts) - i - 1)
            conf = _CONFIRMED if (i == 0 or stable) else _AMBIGUOUS
            reason = (f"row {i}: {modal} header-like fields"
                      + ("" if conf == _CONFIRMED else "; preamble field counts vary"))
            return i, conf, reason
    return 0, _AMBIGUOUS, "no clear header row -> row 0"


def _looks_numeric(token: str) -> bool:
    t = str(token).strip().replace(",", "").replace(" ", "")
    if t == "":
        return False
    try:
        float(t)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Per-column type inference (measured)
# ---------------------------------------------------------------------------

def _nonnull(s: pd.Series) -> pd.Series:
    return s[s.notna() & (s.astype(str).str.strip() != "")]

def _try_datetime(s: pd.Series):
    base = _nonnull(s)
    if base.empty:
        return None, 0.0
    for fmt in _DATE_FORMATS:
        conv = pd.to_datetime(base, format=fmt, errors="coerce")
        if conv.notna().mean() >= _PARSE_ACCEPT:
            return pd.to_datetime(s, format=fmt, errors="coerce"), float(conv.notna().mean())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        conv = pd.to_datetime(base, errors="coerce")
    rate = float(conv.notna().mean())
    if rate >= _PARSE_ACCEPT:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = pd.to_datetime(s, errors="coerce")
        return full, rate
    return None, rate


def _try_numeric(s: pd.Series, decimal: str, thousands: Optional[str]):
    base = _nonnull(s)
    if base.empty:
        return None, 0.0, None
    cleaned = base.astype(str).str.strip()
    suggest = None
    if (cleaned.str.endswith("%")).mean() > 0.5:
        suggest = "% stripped (value kept as written, not divided by 100)"
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(r"^[\$€£¥]", "", regex=True)
    if thousands:
        cleaned = cleaned.str.replace(thousands, "", regex=False)
    if decimal and decimal != ".":
        cleaned = cleaned.str.replace(decimal, ".", regex=False)
    conv = pd.to_numeric(cleaned, errors="coerce")
    rate = float(conv.notna().mean())
    if rate >= _PARSE_ACCEPT:
        full = s.astype(str).str.strip().str.replace("%", "", regex=False)
        full = full.str.replace(r"^[\$€£¥]", "", regex=True)
        if thousands:
            full = full.str.replace(thousands, "", regex=False)
        if decimal and decimal != ".":
            full = full.str.replace(decimal, ".", regex=False)
        out = pd.to_numeric(full, errors="coerce")
        return out, rate, suggest
    return None, rate, suggest


def _try_bool(s: pd.Series):
    base = _nonnull(s).astype(str).str.strip().str.lower()
    if base.empty:
        return None, 0.0
    if set(base.unique()).issubset(_BOOL_TOKENS):
        mapped = s.astype(str).str.strip().str.lower().map(
            lambda v: True if v in _TRUE_TOKENS else (False if v in _BOOL_TOKENS else np.nan))
        return mapped.astype("boolean"), 1.0
    return None, 0.0


def _infer_column(name: str, s: pd.Series, parse_dates: bool,
                  decimal: str, thousands: Optional[str]):
    """Return ``(typed_series, column_plan_dict)``."""
    base = _nonnull(s)
    n_base = int(base.shape[0])

    typed, dtype, rate, suggest = s, "object", 1.0, None

    if parse_dates:
        conv, r = _try_datetime(s)
        if conv is not None:
            typed, dtype, rate = conv, "datetime64[ns]", r
    if dtype == "object":
        conv, r, sug = _try_numeric(s, decimal, thousands)
        if conv is not None:
            typed, dtype, rate, suggest = conv, "float64", r, sug
    if dtype == "object":
        conv, r = _try_bool(s)
        if conv is not None:
            typed, dtype, rate = conv, "boolean", r
    if dtype == "object" and n_base and (base.nunique() / n_base) <= 0.5:
        suggest = "category"

    if dtype != "object":
        dtype = str(typed.dtype)

    n_failed = 0
    if dtype != "object" and n_base:
        n_failed = int(round((1.0 - rate) * n_base))

    all_nan = bool(dtype != "object" and typed.notna().sum() == 0)
    high_risk = bool(_HIGH_RISK_RE.search(str(name)))
    if dtype == "object" or rate >= 1.0:
        conf = _CONFIRMED
    elif all_nan or (high_risk and n_failed > 0):
        conf = _HIGH_RISK
    else:
        conf = _AMBIGUOUS
    if all_nan:
        conf = _HIGH_RISK

    reason = (f"{dtype} at parse_rate={rate:.2f}" if dtype != "object"
              else "kept as text")
    if all_nan:
        reason = "all values failed coercion -> all-NaN"

    col_plan = {"dtype": dtype, "coerced_from": "object",
                "parse_rate": round(float(rate), 4), "n_failed": n_failed,
                "confidence": conf, "reason": reason, "suggest": suggest}
    return typed, col_plan


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _report_frame(plan: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, cp in plan["columns"].items():
        s = df[col] if col in df.columns else pd.Series(dtype=object)
        flag = "" if cp["confidence"] == _CONFIRMED else "! "
        rows.append({
            "column": f"{flag}{col}",
            "dtype": cp["dtype"],
            "parsed_%": round(cp["parse_rate"] * 100, 2),
            "null_%": round(float(s.isna().mean()) * 100, 2) if len(s) else 0.0,
            "n_distinct": int(s.nunique(dropna=True)) if len(s) else 0,
            "problem": cp["reason"] if cp["confidence"] != _CONFIRMED else "",
            "action": cp["suggest"] or "",
            "confidence": cp["confidence"],
        })
    return pd.DataFrame(rows)


def _banner(plan: dict) -> str:
    d = plan["decisions"]
    src = plan["source"]
    enc, sep, hdr = d["encoding"], d["delimiter"], d["header"]
    encs = "" if enc["confidence"] == _CONFIRMED else " (ambiguous)"
    return (f"source={src['name']} | encoding={enc['value']}{encs} | "
            f"sep={sep['value']!r} | header=row {hdr['value']} | "
            f"{plan['metadata']['n_rows']:,}x{plan['metadata']['n_cols']}")


def _ambiguous_items(plan: dict):
    items = []
    for key, dec in plan["decisions"].items():
        if dec["confidence"] != _CONFIRMED:
            items.append(f"{key}={dec['value']!r} ({dec['reason']})")
    for col, cp in plan["columns"].items():
        if cp["confidence"] != _CONFIRMED:
            items.append(f"column {col!r}: {cp['confidence']} - {cp['reason']}")
    return items


def _decision_sentence(plan: dict) -> str:
    m = plan["metadata"]
    d = plan["decisions"]
    coerced = [(c, cp["dtype"]) for c, cp in plan["columns"].items()
               if cp["dtype"] != "object"]
    types = ", ".join(sorted({t for _, t in coerced})) or "none"
    n_failed = sum(cp["n_failed"] for cp in plan["columns"].values())
    a = m["n_ambiguous"]
    hint = " - re-run with params= to confirm" if a > 0 else ""
    return (f"Loaded {m['n_rows']:,} rows x {m['n_cols']} cols from "
            f"'{plan['source']['name']}' [encoding={d['encoding']['value']}, "
            f"sep={d['delimiter']['value']!r}, header=row {d['header']['value']}]; "
            f"coerced {len(coerced)} column(s) ({types}); {n_failed} cell(s) "
            f"failed -> NaN; {a} ambiguous decision(s){hint}. "
            f"Next: dx.clean_rep(df).")


# ---------------------------------------------------------------------------
# Core build
# ---------------------------------------------------------------------------

def _parse_rows(text: str, sep: str):
    reader = csv.reader(io.StringIO(text), delimiter=sep, quotechar='"')
    return [r for r in reader]


def _build_from_text(text, source_meta, kind, on_ambiguous,
                     encoding_dec, sep_forced, header_forced, parse_dates,
                     decimal, thousands, na_values, max_rows):
    sample_text = "\n".join(text.splitlines()[:200])
    dec0 = "." if decimal is None else decimal
    exclude = tuple(c for c in (dec0, thousands) if c and len(c) == 1)
    sep, sconf, sreason = _detect_delimiter(sample_text, sep_forced, exclude)
    rows = [r for r in _parse_rows(text, sep) if r]
    header_row, hconf, hreason = _detect_header(rows, header_forced)

    dec = "." if decimal is None else decimal
    extra_na = list(na_values) if na_values else []
    df = pd.read_csv(io.StringIO(text), sep=sep, header=header_row,
                     dtype=object, keep_default_na=True, na_values=extra_na,
                     skip_blank_lines=True, nrows=max_rows,
                     engine="python", on_bad_lines="skip")
    df.columns = [str(c) for c in df.columns]

    # ragged-row count (robust to quoted delimiters)
    data_counts = [len(r) for r in rows[header_row + 1:] if r]
    problems = []
    if data_counts:
        modal = max(set(data_counts), key=data_counts.count)
        ragged = sum(1 for c in data_counts if c != modal)
        if ragged:
            problems.append({"scope": "rows", "kind": "ragged",
                             "detail": f"{ragged} row(s) had != {modal} fields",
                             "action": "skipped on read"})

    columns, typed = {}, {}
    for col in df.columns:
        ts, cp = _infer_column(col, df[col], parse_dates, dec, thousands)
        typed[col] = ts
        columns[col] = cp
        if cp["confidence"] == _HIGH_RISK and "all-NaN" in cp["reason"]:
            problems.append({"scope": col, "kind": "all_nan",
                             "detail": cp["reason"], "action": "kept as NaN"})

    out = pd.DataFrame(typed)

    n_amb_cols = sum(1 for cp in columns.values() if cp["confidence"] != _CONFIRMED)
    n_amb_parse = sum(1 for c in (sconf, hconf, encoding_dec[1]) if c != _CONFIRMED)
    plan = {
        "function": "load",
        "source": {**source_meta, "kind": kind},
        "parse": {"encoding": encoding_dec[0], "delimiter": sep, "quotechar": '"',
                  "header_row": header_row, "skiprows": list(range(header_row)),
                  "decimal": dec, "thousands": thousands,
                  "na_values": extra_na},
        "columns": columns,
        "problems": problems,
        "decisions": {
            "encoding": {"value": encoding_dec[0], "confidence": encoding_dec[1],
                         "reason": encoding_dec[2]},
            "delimiter": {"value": sep, "confidence": sconf, "reason": sreason},
            "header": {"value": header_row, "confidence": hconf, "reason": hreason},
            "decimal": {"value": dec, "confidence": _CONFIRMED, "reason": "resolved"},
        },
        "policy": {"on_ambiguous": on_ambiguous, "allow_pickle": False,
                   "max_rows": max_rows},
        "metadata": {"n_rows": int(out.shape[0]), "n_cols": int(out.shape[1]),
                     "n_ambiguous": int(n_amb_cols + n_amb_parse)},
        "version": __version__,
        "generated_at": now_iso(),
    }
    return out, json_safe(plan)


def _apply_plan(text, plan, parse_dates):
    """Replay: apply a stored plan verbatim (no detection)."""
    p = plan["parse"]
    df = pd.read_csv(io.StringIO(text), sep=p["delimiter"], header=p["header_row"],
                     dtype=object, keep_default_na=True,
                     na_values=list(p.get("na_values") or []),
                     skip_blank_lines=True, nrows=plan["policy"].get("max_rows"),
                     engine="python", on_bad_lines="skip")
    df.columns = [str(c) for c in df.columns]
    typed = {}
    for col in df.columns:
        cp = plan["columns"].get(col, {"dtype": "object"})
        dtype = cp.get("dtype", "object")
        s = df[col]
        if dtype.startswith("datetime"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                typed[col] = pd.to_datetime(s, errors="coerce")
        elif dtype in ("float64", "int64"):
            ts, _, _ = _try_numeric(s, p.get("decimal", "."), p.get("thousands"))
            typed[col] = ts if ts is not None else pd.to_numeric(s, errors="coerce")
        elif dtype == "boolean":
            ts, _ = _try_bool(s)
            typed[col] = ts if ts is not None else s
        else:
            typed[col] = s
    return pd.DataFrame(typed)


# ---------------------------------------------------------------------------
# Public API
# ===========================================================================
# 11  load  --  smart, transparent, replayable data loader
# ===========================================================================

def load(
    source,
    *,
    kind: str = "auto",
    params: Optional[dict] = None,
    on_ambiguous: str = "warn",
    encoding: Optional[str] = None,
    sep: Optional[str] = None,
    header_row: Optional[int] = None,
    parse_dates: bool = True,
    decimal: Optional[str] = None,
    thousands: Optional[str] = None,
    na_values=None,
    max_rows: Optional[int] = None,
    sample_bytes: int = 262144,
    allow_pickle: bool = False,
    return_params: bool = False,
    show: bool = True,
    decimals: int = 4,
    df_name: Optional[str] = None,
    interactive: bool = False,
):
    """Load a messy source into a typed DataFrame, disclosing every decision.

    Auto-detects encoding, delimiter and the real header row of a delimited-text
    source, then infers per-column types and **measures** how many cells parsed.
    Transparency scales with uncertainty: confident parses load in one line;
    ambiguous decisions are flagged and handled per ``on_ambiguous``
    (``'warn'`` -> load + warn; ``'raise'`` -> raise; ``'plan'`` -> return the
    plan without loading). Every load emits a JSON-safe, replayable load plan
    (the ``params`` artifact); ``load(source, params=plan)`` reproduces the frame
    exactly. The source is never modified.

    Parameters
    ----------
    source : str | os.PathLike | file-like | DataFrame
        A path, an open binary/text stream, or an in-memory frame (typed
        pass-through). ``.pkl`` is refused unless ``allow_pickle=True``.
    kind : {"auto", "csv", "tsv"}
        Source kind; inferred from the extension when ``"auto"``.
    params : dict, optional
        A previously returned load plan to replay deterministically.
    on_ambiguous : {"warn", "raise", "plan"}
        Policy when a decision is ambiguous (see above).
    encoding, sep, header_row, decimal, thousands : optional
        Force a decision instead of detecting it.
    parse_dates : bool
        Attempt safe datetime inference.
    na_values : list, optional
        Extra NA tokens added to the pandas defaults.
    max_rows : int, optional
        Safety cap on the number of rows read.
    allow_pickle : bool
        Permit loading a pickle source (unsafe; off by default).
    return_params : bool
        Also return the load plan.
    show : bool
        Print the disclosure report + the ``Decision:`` sentence.
    decimals : int
        Numeric precision in the printed report.
    df_name : str, optional
        Name used in the audit / decision (inferred when omitted).
    interactive : bool
        Prompt to confirm before loading (classroom use; never during
        composition and never when ``show=False``).

    Returns
    -------
    pandas.DataFrame, or (DataFrame, plan) when ``return_params=True``; the plan
    alone when ``on_ambiguous='plan'``.

    Examples
    --------
    >>> df = dx.load('messy.csv')
    >>> df, plan = dx.load('messy.csv', return_params=True)
    >>> df = dx.load('messy.csv', params=plan)         # deterministic replay
    """
    if on_ambiguous not in ("warn", "raise", "plan"):
        raise ValueError("on_ambiguous must be 'warn', 'raise' or 'plan'.")

    # In-memory frame -> typed pass-through.
    if isinstance(source, pd.DataFrame) or (
            not isinstance(source, (str, os.PathLike))
            and not hasattr(source, "read") and hasattr(source, "to_pandas")):
        return _load_frame(source, on_ambiguous=on_ambiguous,
                           parse_dates=parse_dates, decimal=decimal,
                           thousands=thousands, return_params=return_params,
                           show=show, df_name=df_name)

    if df_name is None:
        df_name = get_variable_name(source, depth=2)
        if df_name == "DataFrame":
            df_name = None

    raw, source_meta = _read_bytes(source)
    if df_name:
        source_meta = {**source_meta, "name": df_name}

    ext = os.path.splitext(source_meta["name"])[1].lower()
    if kind == "auto":
        kind = "tsv" if ext in _TSV_EXT else "csv"
    if kind == "tsv" and sep is None:
        sep = "\t"

    enc, econf, ereason = _detect_encoding(raw[:sample_bytes], encoding)
    text, econf, ereason = _decode_full(raw, enc, econf, ereason)

    if params is not None:
        out = _apply_plan(text, params, parse_dates)
        src_sha = source_meta.get("sha256")
        plan_sha = params.get("source", {}).get("sha256")
        if src_sha and plan_sha and src_sha != plan_sha:
            entry = {"scope": "source", "kind": "source_changed",
                     "detail": "sha256 differs from the plan", "action": "replayed anyway"}
            params.setdefault("problems", []).append(entry)
            if on_ambiguous == "raise":
                raise LoaderReplayError(
                    "source has changed since the plan was created (sha256 mismatch).")
            warnings.warn("load: source changed since the plan was created.",
                          DextraLoaderWarning, stacklevel=2)
        out = out.copy()
        out.attrs = {}
        append_audit(out, {"stage": "loader", "function": "load",
                           "timestamp": now_iso(), "params": params,
                           "decision": "Replayed a stored load plan."})
        if show:
            print("Decision: Replayed a stored load plan on "
                  f"'{source_meta['name']}'.")
        return (out, params) if return_params else out

    out, plan = _build_from_text(
        text, source_meta, kind, on_ambiguous,
        (enc, econf, ereason), sep, header_row, parse_dates,
        decimal, thousands, na_values, max_rows)

    items = _ambiguous_items(plan)
    if show:
        print(_banner(plan))
        with pd.option_context("display.max_columns", None,
                               "display.width", 0,
                               "display.float_format",
                               lambda v: f"{v:,.{decimals}f}"):
            print(_report_frame(plan, out).to_string(index=False))

    if on_ambiguous == "plan":
        if show:
            print(f"Decision: {_decision_sentence(plan)}")
        return plan

    if items and on_ambiguous == "raise":
        raise LoaderAmbiguityError(
            "ambiguous load decision(s):\n  - " + "\n  - ".join(items)
            + "\nOverride explicitly (encoding=/sep=/header_row=/dtype) or use "
              "on_ambiguous='warn'.")
    if items and on_ambiguous == "warn":
        warnings.warn("load: ambiguous decision(s): " + "; ".join(items),
                      DextraLoaderWarning, stacklevel=2)

    if interactive and show:
        resp = input("Apply this plan? [y]/abort: ").strip().lower()
        if resp == "abort":
            raise LoaderAbort("user aborted the load.")

    out.attrs = {}
    append_audit(out, {"stage": "loader", "function": "load",
                       "timestamp": plan["generated_at"], "params": plan,
                       "decision": _decision_sentence(plan)})
    if show:
        print(f"Decision: {_decision_sentence(plan)}")
    return (out, plan) if return_params else out


def _load_frame(source, *, on_ambiguous, parse_dates, decimal, thousands,
                return_params, show, df_name):
    """Typed pass-through for an in-memory frame: re-infer object columns."""
    df = _ensure_pandas(source)
    name = df_name or get_variable_name(source, depth=3)
    dec = "." if decimal is None else decimal
    columns, typed = {}, {}
    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_object_dtype(dt) or pd.api.types.is_string_dtype(dt):
            ts, cp = _infer_column(col, df[col], parse_dates, dec, thousands)
        else:
            ts = df[col]
            cp = {"dtype": str(df[col].dtype), "coerced_from": str(df[col].dtype),
                  "parse_rate": 1.0, "n_failed": 0, "confidence": _CONFIRMED,
                  "reason": "already typed", "suggest": None}
        typed[str(col)] = ts
        columns[str(col)] = cp
    out = pd.DataFrame(typed)
    out.attrs = dict(df.attrs)
    n_amb = sum(1 for cp in columns.values() if cp["confidence"] != _CONFIRMED)
    plan = json_safe({
        "function": "load", "source": {"name": name, "kind": "frame",
                                        "sha256": None, "size": None, "mtime": None},
        "parse": {}, "columns": columns, "problems": [],
        "decisions": {}, "policy": {"on_ambiguous": on_ambiguous},
        "metadata": {"n_rows": int(out.shape[0]), "n_cols": int(out.shape[1]),
                     "n_ambiguous": int(n_amb)},
        "version": __version__, "generated_at": now_iso(),
    })
    decision = (f"Typed pass-through of in-memory frame '{name}': "
                f"{out.shape[0]:,} rows x {out.shape[1]} cols; "
                f"{n_amb} ambiguous column(s). Next: dx.clean_rep(df).")
    append_audit(out, {"stage": "loader", "function": "load",
                       "timestamp": plan["generated_at"], "params": plan,
                       "decision": decision})
    if show:
        print(f"Decision: {decision}")
    return (out, plan) if return_params else out


def peek(source, *, kind: str = "auto", on_ambiguous: str = "plan",
         show: bool = True, n_preview: int = 10, **load_kwargs):
    """Propose a load plan + preview WITHOUT committing a full load.

    Returns the load plan (dict); loads at most ``n_preview`` rows for the sample.
    The teaching / inspection entry point ("look before you load").
    """
    load_kwargs.pop("return_params", None)
    load_kwargs.pop("max_rows", None)
    plan = load(source, kind=kind, on_ambiguous="plan", show=show,
                max_rows=n_preview, **load_kwargs)
    return plan


# Short aliases, consistent with the underscore-free Phase 8/9/10 naming.
dload = load
dpeek = peek
