#!/usr/bin/env python3
"""Reproduce Figure 5 (`fig:restrictioneval`): restriction quality vs Bezier degree.

The counterexample distribution from paper Example 3.6 is a degree-3 Bezier
distribution with non-monotone control points:
    controls_x = (0, 0.6, 0.4, 1)
    controls_z = (0, 1,   0,   1)
We draw 1000 samples from it via inverse-transform sampling (closed form), then
fit IPOPT (nonlinear, restricted formulation) and PGD (further-restricted convex
QP) for each Bezier degree n = 3, ..., 20 and plot the resulting MSE.

Output: figures/restriction.pgf
        figures/restriction_raw.csv

Usage:
    python fig_restriction.py
"""
import argparse
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import _plot_style
from bezierv.classes.distfit import DistFit
from _fit_benchmark import _mse_options


# ── Counterexample distribution (paper Example 3.6) ──────────────────────────
# Bezier cubic with non-monotone control points; the pdf is still ≥ 0.
#     F(t) = sum_{i=0..3} B_{3,i}(t) * z_i, where z = (0, 1, 0, 1)
#          = 3t(1-t)^2 + t^3
#     x(t) = sum_{i=0..3} B_{3,i}(t) * x_i, where x = (0, 0.6, 0.4, 1)
#          = 1.8 t (1-t)^2 + 1.2 t^2 (1-t) + t^3
# F is monotone in t (F'(t) = 3(1-2t)^2 ≥ 0), so inverse-transform sampling works:
#     u ∈ U[0,1] → solve F(t) = u → return x(t)
def _F(t):  # noqa: N802
    return 3.0 * t * (1.0 - t) ** 2 + t ** 3


def _x(t):  # noqa: N802
    return 1.8 * t * (1.0 - t) ** 2 + 1.2 * t ** 2 * (1.0 - t) + t ** 3


def sample_counterexample(n: int, *, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    out = np.empty(n)
    for i, ui in enumerate(u):
        # F is monotone increasing on [0,1] with F(0)=0, F(1)=1.
        t_sol = brentq(lambda t: _F(t) - ui, 0.0, 1.0)
        out[i] = _x(t_sol)
    return np.sort(out)


def run_sweep(samples: np.ndarray, n_values: range) -> pd.DataFrame:
    rows = []
    for n in n_values:
        row = {'n': n}
        for method, key in [('nonlinear', 'IPOPT'), ('projgrad', 'PGD')]:
            try:
                try:
                    df = DistFit(samples, n=n, method_init_x='uniform')
                except ValueError:
                    df = DistFit(samples, n=n, method_init_x='uniform')
                t0 = _time.perf_counter()
                _, mse = df.fit(method='mse', algorithm=method, options=_mse_options(method))
                row[f'{key}_MSE'] = float(mse)
                row[f'{key}_time_ms'] = (_time.perf_counter() - t0) * 1e3
            except Exception as exc:
                print(f'  n={n:2d}  {key} failed: {exc}')
                row[f'{key}_MSE'] = float('nan')
                row[f'{key}_time_ms'] = float('nan')
        print(f"  n={n:2d}  IPOPT MSE={row['IPOPT_MSE']:.2e} ({row['IPOPT_time_ms']:6.0f} ms)   "
              f"PGD MSE={row['PGD_MSE']:.2e} ({row['PGD_time_ms']:5.0f} ms)")
        rows.append(row)
    return pd.DataFrame(rows).set_index('n')


def emit_figure(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    mm = _plot_style.mm
    fig, ax = plt.subplots(figsize=(130 * mm, 90 * mm), layout='constrained')
    ax.plot(df.index, df['IPOPT_MSE'], marker='o', label='IPOPT')
    ax.plot(df.index, df['PGD_MSE'],   marker='s', label='PGD')
    ax.set_yscale('log')
    ax.set_xlabel(r'Degree $n$')
    ax.set_ylabel(r'MSE (log scale)')
    ax.set_xticks(df.index)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_style.save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce paper Figure 5 (restriction quality vs n).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-samples', type=int, default=1000)
    parser.add_argument('--n-min', type=int, default=3)
    parser.add_argument('--n-max', type=int, default=20)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'figures')
    args = parser.parse_args()

    _plot_style.apply()

    print(f'[fig_restriction] Drawing {args.n_samples} samples from counterexample (seed={args.seed})')
    samples = sample_counterexample(args.n_samples, seed=args.seed)
    print(f'[fig_restriction] sample stats: min={samples.min():.3f}, max={samples.max():.3f}, '
          f'mean={samples.mean():.3f}')

    print(f'[fig_restriction] Sweeping degree n = {args.n_min}..{args.n_max}')
    df = run_sweep(samples, range(args.n_min, args.n_max + 1))

    csv_path = args.output_dir / 'restriction_raw.csv'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)
    print(f'[fig_restriction] Wrote {csv_path}')

    out = args.output_dir / 'restriction.pgf'
    emit_figure(df, out)
    print(f'[fig_restriction] Wrote {out}')


if __name__ == '__main__':
    main()
