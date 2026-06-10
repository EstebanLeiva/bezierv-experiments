#!/usr/bin/env python3
"""Plot a Bézier PDF/CDF from hard-coded control points.

Same dual-axis layout as fig_multimodal.py, but the control points are fixed
global constants (CONTROLS_X / CONTROLS_Z) instead of being fitted to data.

Outputs: figures/multimodal_manual.pgf      (Bezier PDF/CDF + control points)
         figures/multimodal_manual_hist.pgf (histogram of N_SIMS Bezier draws)
Usage:   python fig_multimodal_manual.py
"""
import argparse
from pathlib import Path

import numpy as np

import _plot_style
from bezierv.classes.bezierv import Bezierv


# Control points of the Bézier random variable (degree = len - 1).
CONTROLS_X = [9.442635548513067, 9.589313786242428, 9.768114864316782, 9.768130416271076,
              9.970292391164527, 9.970313480943467, 9.970322676000000, 10.034634693354150,
              10.034646185315097, 10.034673127690063, 10.423673394402751, 10.423842653151324,
              10.423855526861590, 10.423860867001675, 10.423865659949564, 10.608942683968376]
CONTROLS_Z = [0.0, 0.027559993444077512, 0.15624064767833581, 0.15624554336745830,
              0.26291583355898210, 0.26292118493303984, 0.26293727667182960, 0.7557663976572169,
              0.7557705230264974, 0.7557761121147117, 0.7591283006340018, 0.7690466559442081,
              0.7690860211361442, 0.7690948987248934, 0.9999880233174911, 1.0]
HIST_BINS = 40
N_SIMS = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot a Bezier PDF/CDF from hard-coded control points.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path,
                        default=Path(__file__).parent / 'figures' / 'multimodal_manual.pgf')
    parser.add_argument('--output-hist', type=Path,
                        default=Path(__file__).parent / 'figures' / 'multimodal_manual_hist.pgf')
    args = parser.parse_args()

    _plot_style.apply()
    import matplotlib.pyplot as plt

    controls_x = np.asarray(CONTROLS_X, dtype=float)
    controls_z = np.asarray(CONTROLS_Z, dtype=float)
    n = len(controls_x) - 1
    print(f'[fig_multimodal_manual] Building Bezier RV (n={n}) from hard-coded control points')
    rv = Bezierv(n, controls_x, controls_z)

    print(f'[fig_multimodal_manual] Simulating N_sim={N_SIMS} draws from the Bezier RV')
    sim = rv.random(N_SIMS, rng=args.seed)

    x_vals = np.linspace(controls_x[0], controls_x[-1], 400)
    pdf_vals = np.array([rv.pdf_x(x) for x in x_vals])
    cdf_vals = np.array([rv.cdf_x(x) for x in x_vals])

    mm = _plot_style.mm
    fig, ax1 = plt.subplots(figsize=(140 * mm, 95 * mm), layout='constrained')
    ax1.plot(x_vals, pdf_vals, color='#0072B2', label=r'B\'ezier PDF', zorder=2)
    ax1.set_xlabel(r'Value')
    ax1.set_ylabel(r'Probability density')
    ax1.spines[['top']].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(x_vals, cdf_vals, color='#D55E00', linestyle='--',
             label=r'B\'ezier CDF', zorder=2)
    ax2.plot(rv.controls_x, rv.controls_z, color='#009E73', marker='o',
             markersize=4, linestyle='None',
             label=r'Control points', zorder=3)
    ax2.set_ylabel(r'Cumulative probability')
    ax2.set_ylim(-0.02, 1.08)
    ax2.spines[['top']].set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='outside lower center', ncol=3)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_style.save(fig, args.output)
    print(f'[fig_multimodal_manual] Wrote {args.output} (+ .png)')

    # Standalone histogram of the simulated draws, matching fig_multimodal's style.
    fig_h, axh = plt.subplots(figsize=(140 * mm, 95 * mm), layout='constrained')
    axh.hist(sim, bins=HIST_BINS, color='#BBBBBB', edgecolor='white',
             density=True, alpha=0.85, label='Data', zorder=1)
    axh.set_xlabel(r'Value')
    axh.set_ylabel(r'Probability density')
    axh.spines[['top', 'right']].set_visible(False)
    fig_h.legend(loc='outside lower center')

    args.output_hist.parent.mkdir(parents=True, exist_ok=True)
    _plot_style.save(fig_h, args.output_hist)
    print(f'[fig_multimodal_manual] Wrote {args.output_hist} (+ .png)')


if __name__ == '__main__':
    main()
