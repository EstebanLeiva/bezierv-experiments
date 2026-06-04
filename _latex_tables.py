"""Tiny LaTeX helpers for emitting paper-faithful booktabs tables."""
from __future__ import annotations

import math
from pathlib import Path


# ── Numeric formatters ───────────────────────────────────────────────────────
def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '--'
    return f'{int(round(float(x)))}'


def fmt_float(x, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '--'
    return f'{float(x):.{decimals}f}'


def fmt_sci(x, decimals: int = 1) -> str:
    """Scientific notation matching the paper's `1.9e-5` style."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '--'
    s = f'{float(x):.{decimals}e}'
    mantissa, exp = s.split('e')
    return f'{mantissa}e{int(exp):+d}'.replace('e+', 'e').replace('e-0', 'e-').replace('e0', 'e')


def fmt_pct(x, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '--'
    return f'{float(x) * 100:.{decimals}f}\\%'


def fmt_signed_nll(x, decimals: int = 1) -> str:
    """Paper uses `$-$2259.8` style for negative NLLs."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '--'
    v = float(x)
    if v < 0:
        return f'$-${abs(v):.{decimals}f}'
    return f'{v:.{decimals}f}'


def bold(s: str) -> str:
    return f'\\textbf{{{s}}}'


# ── Table wrapper ────────────────────────────────────────────────────────────
def emit_table(
    path: str | Path,
    body: str,
    *,
    caption: str,
    label: str,
    column_spec: str,
    resizebox: str = '\\textwidth',
) -> None:
    """Write a `\\begin{table}...\\end{table}` block ready for `\\input{}`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        '\\begin{table}[H]\n'
        '    \\centering\n'
        f'    \\resizebox{{{resizebox}}}{{!}}{{\n'
        f'    \\begin{{tabular}}{{{column_spec}}}\n'
        f'{body}'
        '    \\end{tabular}}\n'
        f'    \\caption{{{caption}}}\n'
        f'    \\label{{{label}}}\n'
        '\\end{table}\n'
    )
    path.write_text(text)
