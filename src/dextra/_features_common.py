"""dextra features - shared internal helpers (Phase 4)."""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    from IPython.display import display as _ipy_display
except ImportError:  # pragma: no cover
    _ipy_display = None


sns.set_style("whitegrid")


AUDIT_KEY = "dextra_audit"


def _get_scipy_stats(method: str):
    """Lazily import scipy.stats; only boxcox / yeojohnson need it.

    Keeping the import lazy means log / log1p / sqrt and the whole scale()
    family work even when SciPy is not installed.
    """
    try:
        from scipy import stats as _sst
        return _sst
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"transform method '{method}' requires SciPy, which is not "
            f"installed. Install it with `pip install scipy`, or use "
            f"method='log' / 'log1p' / 'sqrt' which need only NumPy."
        ) from exc


def _display(frame: pd.DataFrame) -> None:
    if _ipy_display is not None:
        _ipy_display(frame)
    else:
        print(frame.to_string())


def _print_header(title: str) -> None:
    print(title)
    print("-" * len(title))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finalize_figure(fig, return_fig: bool) -> None:
    """Display the figure when one was created.

    ``show`` and ``plot`` are INDEPENDENT in dextra. A figure is created only
    when ``plot=True``; this helper displays it unless the caller asked for
    the figure object back via ``return_fig``.
    """
    if fig is None:
        return
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if not return_fig:
        plt.show()


def _ret_pack(out, params, fig, return_df, return_params, return_fig):
    """Pack outputs in the fixed order: dataframe, params, figure.

    Only the requested pieces are returned. A single requested piece is
    returned bare; multiple pieces are returned as a tuple.
    """
    results = []
    if return_df:
        results.append(out)
    if return_params:
        results.append(params)
    if return_fig:
        results.append(fig)
    if not results:
        return out
    if len(results) == 1:
        return results[0]
    return tuple(results)


def _append_audit(out: pd.DataFrame, entry: dict) -> None:
    out.attrs.setdefault(AUDIT_KEY, [])
    out.attrs[AUDIT_KEY] = list(out.attrs[AUDIT_KEY])
    out.attrs[AUDIT_KEY].append(entry)


def _fmt_table(frame: pd.DataFrame, decimals: int) -> pd.DataFrame:
    """Format every cell of a summary table for readable display."""
    def _fmt(v):
        if pd.isna(v):
            return "-"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        try:
            return f"{float(v):,.{decimals}f}"
        except (TypeError, ValueError):
            return str(v)
    return frame.map(_fmt)


def _auto_numeric_cols(df: pd.DataFrame) -> list:
    """Return numeric, non-boolean column names."""
    return [c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])]


def _resolve_cols(df: pd.DataFrame, cols, func_name: str) -> list:
    """Validate an explicit cols selector or auto-pick numeric columns."""
    if cols is None:
        chosen = _auto_numeric_cols(df)
        if not chosen:
            raise ValueError(
                f"{func_name}: no numeric columns found. Pass cols= explicitly.")
        return chosen
    chosen = list(cols)
    bad = [c for c in chosen if c not in df.columns]
    if bad:
        raise KeyError(f"{func_name}: cols references columns not in df: {bad}")
    non_num = [c for c in chosen if not pd.api.types.is_numeric_dtype(df[c])]
    if non_num:
        raise TypeError(
            f"{func_name} requires numeric columns; non-numeric passed: {non_num}")
    return chosen


def _hist_bins(n: int) -> int:
    if n <= 1:
        return 10
    return max(10, min(60, int(np.sqrt(n))))
