#!/usr/bin/env python3
"""Descriptive statistics of the multimodal sample vs. the fitted Bezier RV.

Companion to ``fig_multimodal.py``: tabulates the 10/25/75/90 percent quantiles,
mean, and variance of the 4-cavity mixture sample alongside the same quantities
read off the fitted Bezier random variable, showing the fit reproduces them.

Output: tables/multimodal_stats.tex

Usage:
    python tab_multimodal.py
"""
import argparse
from pathlib import Path

import numpy as np

import _latex_tables as lt
from bezierv.classes.distfit import DistFit
from fig_multimodal import BEZIER_DEGREE, generate_data

# (label, alpha) for each quantile row; mean/variance are handled separately.
QUANTILES = [
    (r'$q_{0.10}$', 0.10),
    (r'$q_{0.25}$', 0.25),
    (r'$q_{0.75}$', 0.75),
    (r'$q_{0.90}$', 0.90),
]


def emit_table(rows: list[tuple[str, float, float]], out_path: Path) -> None:
    lines = ['        \\toprule\n']
    lines.append('        Statistic & \\multicolumn{1}{c}{Empirical} '
                 '& \\multicolumn{1}{c}{B\\\'ezier fit} \\\\\n')
    lines.append('        \\midrule\n')
    for label, emp, fit in rows:
        lines.append(f'        {label} & {lt.fmt_float(emp, 1)} & {lt.fmt_float(fit, 1)} \\\\\n')
    lines.append('        \\bottomrule\n')

    lt.emit_table(
        out_path,
        body=''.join(lines),
        caption=('Descriptive statistics of the multimodal sample and the fitted '
                 'B\\\'ezier random variable.'),
        label='tab:multimodal_stats',
        column_spec='lrr',
        resizebox='0.5\\textwidth',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Descriptive-statistics table for the multimodal fit.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'tables')
    args = parser.parse_args()

    data = generate_data(args.seed)['Diameter'].to_numpy()
    print(f'[tab_multimodal] Fitting 4-cavity mixture (seed={args.seed}, N={data.size}, n={BEZIER_DEGREE})')
    rv, mse = DistFit(data, n=BEZIER_DEGREE).fit(method='mse', algorithm='nonlinear')
    print(f'[tab_multimodal] MSE={mse:.6f}')

    rows = []
    for label, alpha in QUANTILES:
        rows.append((label, float(np.quantile(data, alpha)), rv.quantile(alpha)))
    rows.append(('Mean', float(np.mean(data)), rv.mean()))
    rows.append(('Variance', float(np.var(data, ddof=1)), rv.variance()))

    out_path = args.output_dir / 'multimodal_stats.tex'
    emit_table(rows, out_path)
    print(f'[tab_multimodal] Wrote {out_path}')


if __name__ == '__main__':
    main()
