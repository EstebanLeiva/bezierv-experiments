#!/usr/bin/env python3
"""Reproduce the data-driven reliable shortest-path application (paper §8).

Steps:
    1. Load Chicago_main.json, preprocess for the chosen regime.
    2. Fit a Bezier RV (PGD) on every arc.
    3. Solve the S-aRP via the pulse algorithm with Monte-Carlo convolution
       (matching node 136 -> 306, alpha=0.9, gamma=0.4).
    4. Plot the Chicago network with the optimal path overlaid (paper
       Figure 6a) and the path's travel-time distribution with the budget and
       reliability lines (paper Figure 6b).

Outputs:
    figures/sarp_path_grid.pgf
    figures/sarp_path_dist.pgf
    + console summary: path nodes, total length (miles), achieved reliability

Usage:
    python exp_sarp.py                            # paper defaults (full Chicago, rush_hour)
    python exp_sarp.py --regime full_day --max-arcs 200    # smoke run on a subset
"""
import argparse
from pathlib import Path

import numpy as np

import _chicago_graph as cg
import _plot_style
import _sarp
from _fit_benchmark import _mse_options
from bezierv.classes.distfit import DistFit


def make_pgd_fitter(n_bezier: int):
    """Return a callable that fits a Bezier RV via PGD for use with build_bezier_graph."""
    def fit(samples: np.ndarray):
        try:
            df = DistFit(samples, n=n_bezier)
        except ValueError:
            df = DistFit(samples, n=n_bezier, method_init_x='uniform')
        rv, _ = df.fit(method='mse', algorithm='projgrad',
                       options=_mse_options('projgrad'))
        return rv
    return fit


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce paper §8 (S-aRP on Chicago network).')
    parser.add_argument('--regime', default='rush_hour', choices=list(cg.REGIMES))
    parser.add_argument('--start', type=int, default=136)
    parser.add_argument('--end', type=int, default=306)
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--gamma', type=float, default=0.4)
    parser.add_argument('--n-sims', type=int, default=1000)
    parser.add_argument('--n-bezier', type=int, default=None,
                        help='Bezier degree per arc. Defaults to paper value per regime.')
    parser.add_argument('--max-arcs', type=int, default=0,
                        help='Subsample arcs (smoke mode); 0 = use all.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'figures')
    args = parser.parse_args()

    n_bezier = args.n_bezier or cg.REGIME_BEZIER_DEGREE[args.regime]
    rng = np.random.default_rng(args.seed)

    print(f'[exp_sarp] regime={args.regime}  n_bezier={n_bezier}  '
          f'alpha={args.alpha}  gamma={args.gamma}  n_sims={args.n_sims}')

    print('[exp_sarp] Loading Chicago_main.json …')
    raw = cg.load_raw()
    proc = cg.preprocess_regime(raw, args.regime, seed=args.seed)
    proc = cg.filter_multi_obs(proc, min_obs=2)
    print(f'[exp_sarp] {len(proc)} arcs after filtering')
    if args.max_arcs > 0:
        proc = cg.sample_arcs(proc, args.max_arcs, seed=args.seed)
        print(f'[exp_sarp] subsampled to {len(proc)} arcs (smoke mode)')

    print('[exp_sarp] Fitting Bezier RV per arc (PGD) …')
    fit_func = make_pgd_fitter(n_bezier)
    graph = _sarp.build_bezier_graph(proc, fit_func=fit_func)

    print(f'[exp_sarp] Running pulse algorithm  ({args.start} -> {args.end}) …')
    pulse, elapsed, cal = _sarp.run_sarp(
        graph, args.start, args.end,
        alpha=args.alpha, gamma=args.gamma,
        n_sims=args.n_sims, n_bezier=n_bezier, rng=rng,
    )

    path = [int(p) for p in pulse.optimal_path]
    total_miles = sum(
        graph.nodes[path[i]].links[path[i + 1]].deterministic['length_miles']
        for i in range(len(path) - 1)
    )

    rng_eval = np.random.default_rng(args.seed + 1)
    bz_path = _sarp.path_distribution(graph, path, n_sims=args.n_sims,
                                      n_bezier=n_bezier, rng=rng_eval)
    achieved = float(bz_path.cdf_x(cal['T_max']))

    print('─' * 60)
    print(f'  Optimal path        : {path}')
    print(f'  # arcs              : {len(path) - 1}')
    print(f'  Total length (mi)   : {total_miles:.2f}')
    print(f'  Time budget T_max(h): {cal["T_max"]:.3f}')
    print(f'  Target reliability  : {args.alpha:.2f}')
    print(f'  Achieved reliability: {achieved:.3f}')
    print(f'  Pulse runtime (s)   : {elapsed:.2f}')
    print('─' * 60)

    _plot_style.apply()
    import matplotlib.pyplot as plt
    mm = _plot_style.mm

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig_grid, _ = _sarp.plot_path_on_grid(graph, path)
    out_grid = args.output_dir / 'sarp_path_grid.pgf'
    _plot_style.save(fig_grid, out_grid)
    print(f'[exp_sarp] Wrote {out_grid} (+ .png)')

    fig, ax = plt.subplots(figsize=(140 * mm, 95 * mm), layout='constrained')
    x_vals = np.linspace(bz_path.controls_x[0], bz_path.controls_x[-1], 400)
    cdf_vals = np.array([bz_path.cdf_x(x) for x in x_vals])
    ax.plot(x_vals, cdf_vals, color='#0072B2', label=r'B\'ezier CDF')
    ax.axhline(args.alpha, color='#009E73', linestyle='--',
               label=f'${int(args.alpha * 100)}\\%$ reliability')
    ax.axvline(cal['T_max'], color='#D55E00', linestyle='--', label=r'Time budget')
    ax.set_xlabel(r'Travel time (hours)')
    ax.set_ylabel(r'Cumulative probability')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
              ncol=3, frameon=False, borderaxespad=0.)

    out = args.output_dir / 'sarp_path_dist.pgf'
    _plot_style.save(fig, out)
    print(f'[exp_sarp] Wrote {out} (+ .png)')


if __name__ == '__main__':
    main()
