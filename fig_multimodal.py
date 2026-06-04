#!/usr/bin/env python3
"""Reproduce Figure 4: Bézier fit of a 4-component Gaussian mixture.

Output: figures/multimodal.pgf
Usage:  python fig_multimodal.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _plot_style
from bezierv.classes.distfit import DistFit


CAVITY_CONFIGS = [
    {'id': 1, 'mean': 9.80,  'std': 0.05, 'count': 250},
    {'id': 2, 'mean': 9.85,  'std': 0.06, 'count': 250},
    {'id': 3, 'mean': 10.10, 'std': 0.08, 'count': 250},
    {'id': 4, 'mean': 10.45, 'std': 0.05, 'count': 250},
]
BEZIER_DEGREE = 15


def generate_data(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for cfg in CAVITY_CONFIGS:
        data = rng.normal(loc=cfg['mean'], scale=cfg['std'], size=cfg['count'])
        frames.append(pd.DataFrame({'Diameter': data}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce paper Figure 4 (multimodal Bezier fit).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'figures' / 'multimodal.pgf')
    args = parser.parse_args()

    _plot_style.apply()
    import matplotlib.pyplot as plt

    print(f'[fig_multimodal] Generating 4-cavity mixture (seed={args.seed}, n={BEZIER_DEGREE})')
    df = generate_data(args.seed)

    fitter = DistFit(df['Diameter'], n=BEZIER_DEGREE)
    rv, mse = fitter.fit(method='mse', algorithm='nonlinear')
    print(f'[fig_multimodal] MSE={mse:.6f}  mean={rv.get_mean():.4f}  std={rv.get_variance() ** 0.5:.4f}')

    x_vals = np.linspace(df['Diameter'].min(), df['Diameter'].max(), 400)
    pdf_vals = np.array([rv.pdf_x(x) for x in x_vals])
    cdf_vals = np.array([rv.cdf_x(x) for x in x_vals])

    mm = _plot_style.mm
    fig, ax1 = plt.subplots(figsize=(140 * mm, 95 * mm), layout='constrained')
    ax1.hist(df['Diameter'], bins=25, color='#BBBBBB', edgecolor='white',
             density=True, alpha=0.85, label='Data', zorder=1)
    ax1.plot(x_vals, pdf_vals, color='#0072B2', label=r'B\'ezier PDF', zorder=2)
    ax1.set_xlabel(r'Value')
    ax1.set_ylabel(r'Probability density')
    ax1.spines[['top']].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(x_vals, cdf_vals, color='#D55E00', linestyle='--',
             label=r'B\'ezier CDF', zorder=2)
    ax2.set_ylabel(r'Cumulative probability')
    ax2.set_ylim(-0.02, 1.08)
    ax2.spines[['top']].set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='outside lower center', ncol=3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_style.save(fig, args.output)
    print(f'[fig_multimodal] Wrote {args.output} (+ .png)')


if __name__ == '__main__':
    main()
