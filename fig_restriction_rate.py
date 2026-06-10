#!/usr/bin/env python3
"""Fitted-vs-true CDF error of the restricted (monotone) Bezier fit -> 0.

Companion to ``fig_restriction.py``: confirms empirically that the monotone
restriction is asymptotically lossless -- as the degree N grows, the restricted
PGD fit's sup-norm distance to the *true* CDF (the quantity Theorem 3.8 bounds)
goes to zero.

m must grow with N: the fitted-vs-true error splits as approx_error(N) ~ C/N
(Thm 3.8) + estimation_error(m) ~ sqrt(ln2/2m) (DKW). At fixed m the second term
is a floor (~0.02 at m=2000), so we schedule m(N) = round(c*N^2) (default c=20,
m=2000 at N=10) to shrink it like 1/N in step with the restriction gap. The fit
only ever sees samples; the true CDF is used solely to score it.

Each degree is averaged over ``n_reps`` independent realizations (per-(N, rep)
seeds) and plotted as the mean sup error with a +/-1 std band. PGD only: at
m ~ 1e5-1e6 IPOPT is infeasible and its problem size is decoupled from m. PGD
iterations scale with N (max_iter ~ 12*N, threshold 1e-5) since the Bernstein
Gram matrix is ill-conditioned at large N.

Output: figures/restriction_rate.pgf  (+ .png)
        figures/restriction_rate_raw.csv          (per-degree mean/std)
        figures/restriction_rate_checkpoint.csv   (append-only, per (N, rep))

Usage:
    python fig_restriction_rate.py               # paper-faithful (defaults reproduce the figure)
    python fig_restriction_rate.py --no-resume   # ignore any existing checkpoint and start fresh
"""
import argparse
import time as _time
from pathlib import Path

import numpy as np
import pandas as pd

import _plot_style
from bezierv.classes.distfit import DistFit, ProjGradOptions
# Reuse the exact closed-form counterexample curve (paper Example 3.6).
from fig_restriction import _F, _x


CHECKPOINT_COLUMNS = ['n', 'rep', 'm', 'status', 'MSE', 'sup', 'l2', 'time_ms']
VALUE_COLUMNS = ['m', 'MSE', 'sup', 'l2', 'time_ms']

# Precomputed grid for vectorised inverse-transform sampling and true-CDF
# evaluation: _F = CDF(t), _x = support point x(t).
_T_GRID = np.linspace(0.0, 1.0, 200_001)
_F_GRID = _F(_T_GRID)
_X_GRID = _x(_T_GRID)


# ── Sampling and true CDF (vectorised; scales to m ~ 1e6) ─────────────────────
def sample_counterexample(m: int, seed) -> np.ndarray:
    """m sorted draws via vectorised inverse-transform sampling: u~U[0,1] -> t=F^{-1}(u) -> x(t).

    ``seed`` may be an int or a SeedSequence (for independent per-realization seeds).
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=m)
    t = np.interp(u, _F_GRID, _T_GRID)
    return np.sort(_x(t))


def true_cdf_on_grid(x_eval: np.ndarray) -> np.ndarray:
    """True CDF at support points x: invert x = x(t) then read F(t)."""
    out = np.empty_like(x_eval)
    interior = (x_eval > 0.0) & (x_eval < 1.0)
    t = np.interp(x_eval[interior], _X_GRID, _T_GRID)
    out[interior] = _F(t)
    out[x_eval <= 0.0] = 0.0
    out[x_eval >= 1.0] = 1.0
    return out


def cdf_errors(bezierv, x_eval: np.ndarray, true_F: np.ndarray) -> tuple[float, float]:
    """Sup-norm and L2-norm distance between the fitted and true CDF."""
    fitted = np.array([bezierv.cdf_x(float(x)) for x in x_eval])
    diff = np.abs(fitted - true_F)
    return float(diff.max()), float(np.sqrt(np.trapezoid(diff ** 2, x_eval)))


# ── Degree grid, sample-size schedule, PGD iteration budget, seeds ────────────
def log_spaced_degrees(n_min: int, n_max: int, num: int) -> np.ndarray:
    grid = np.geomspace(n_min, n_max, num=num)
    return np.unique(np.clip(np.round(grid).astype(int), n_min, n_max))


def samples_for_degree(n: int, *, per_n2: float, m_min: int, m_cap: int) -> int:
    """m(N) = clip(round(per_n2 * N^2), m_min, m_cap)."""
    return int(np.clip(round(per_n2 * n * n), m_min, m_cap))


def pgd_max_iter(n: int, *, iter_per_n: float, min_iter: int, max_iter_cap: int) -> int:
    return int(np.clip(round(iter_per_n * n), min_iter, max_iter_cap))


def rep_seed(base_seed: int, n: int, rep: int) -> np.random.SeedSequence:
    """Independent, reproducible seed for realization ``rep`` at degree ``n``."""
    return np.random.SeedSequence([int(base_seed), int(n), int(rep)])


# ── Checkpoint I/O ───────────────────────────────────────────────────────────
def _append_checkpoint(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    frame = pd.DataFrame([{c: row.get(c) for c in CHECKPOINT_COLUMNS}],
                         columns=CHECKPOINT_COLUMNS)
    with open(path, 'a', newline='') as f:
        frame.to_csv(f, header=write_header, index=False)
        f.flush()


def _resolve_checkpoint(path: Path) -> dict:
    """Collapse the append-only log into a final status per (n, rep).

    A lone ``attempt`` (no ``ok``/``failed`` follow-up) means the process died
    mid-fit -> ``crashed``. ``ok`` beats ``failed`` beats a bare ``attempt``.
    """
    resolved: dict = {}
    if not path.exists():
        return resolved
    df = pd.read_csv(path)
    if df.empty:
        return resolved
    rank = {'attempt': 0, 'failed': 1, 'ok': 2}
    for (n, rep), grp in df.groupby(['n', 'rep']):
        best = grp.loc[grp['status'].map(rank).fillna(-1).idxmax()]
        status = str(best['status'])
        if status == 'attempt':
            status = 'crashed'
        rec = {'status': status}
        for c in VALUE_COLUMNS:
            rec[c] = float(best[c]) if status == 'ok' else float('nan')
        resolved[(int(n), int(rep))] = rec
    return resolved


def run_sweep(n_values: np.ndarray, x_eval: np.ndarray, true_F: np.ndarray, *,
              seed: int, n_reps: int, per_n2: float, m_min: int, m_cap: int,
              budget_s: float, pgd_cap_s: float, checkpoint_path: Path,
              pgd_iter_per_n: float, pgd_min_iter: int, pgd_max_iter_cap: int,
              pgd_threshold: float) -> dict:
    resolved = _resolve_checkpoint(checkpoint_path)
    if resolved:
        print(f'[resume] loaded {len(resolved)} (degree, rep) fits from checkpoint')

    t_start = _time.perf_counter()
    stop = False
    for n in n_values:
        if stop:
            break
        n = int(n)
        m = samples_for_degree(n, per_n2=per_n2, m_min=m_min, m_cap=m_cap)
        max_iter = pgd_max_iter(n, iter_per_n=pgd_iter_per_n,
                                min_iter=pgd_min_iter, max_iter_cap=pgd_max_iter_cap)

        sups = []
        for rep in range(n_reps):
            if _time.perf_counter() - t_start > budget_s:
                print(f'[budget] {budget_s:.0f}s spent; stopping at n={n}, rep={rep}')
                stop = True
                break
            prior = resolved.get((n, rep))
            if prior is not None and prior['status'] in ('ok', 'failed', 'crashed'):
                if prior['status'] == 'ok':
                    sups.append(prior['sup'])
                continue

            _append_checkpoint(checkpoint_path,
                               {'n': n, 'rep': rep, 'm': m, 'status': 'attempt'})
            t0 = _time.perf_counter()
            try:
                samples = sample_counterexample(m, rep_seed(seed, n, rep))
                df = DistFit(samples, n=n, method_init_x='uniform')
                bezierv, mse = df.fit(method='mse', algorithm='projgrad',
                                      options=ProjGradOptions(max_iter=max_iter,
                                                              threshold=pgd_threshold))
                elapsed = _time.perf_counter() - t0
                sup, l2 = cdf_errors(bezierv, x_eval, true_F)
                rec = {'status': 'ok', 'm': m, 'MSE': float(mse), 'sup': sup,
                       'l2': l2, 'time_ms': elapsed * 1e3}
            except Exception as exc:
                elapsed = _time.perf_counter() - t0
                print(f'  n={n:4d} rep={rep:2d}  PGD failed after {elapsed:.1f}s: {exc}')
                _append_checkpoint(checkpoint_path,
                                   {'n': n, 'rep': rep, 'm': m, 'status': 'failed'})
                resolved[(n, rep)] = {'status': 'failed',
                                      **{c: float('nan') for c in VALUE_COLUMNS}}
                continue

            _append_checkpoint(checkpoint_path, {'n': n, 'rep': rep, **rec})
            resolved[(n, rep)] = rec
            sups.append(sup)
            if elapsed > pgd_cap_s:
                print(f'  n={n:4d}  PGD fit took {elapsed:.1f}s > cap {pgd_cap_s:.0f}s; '
                      f'stopping the sweep (larger n only gets slower)')
                stop = True
                break

        if sups:
            arr = np.array(sups)
            print(f'  n={n:4d}  m={m:>8d}  iters={max_iter:5d}  reps={len(arr):2d}  '
                  f'sup mean={arr.mean():.3e}  std={arr.std(ddof=1) if len(arr) > 1 else 0:.1e}',
                  flush=True)

    return resolved


def aggregate_to_wide(resolved: dict, n_reps: int) -> pd.DataFrame:
    """Per-degree mean/std over the ``ok`` realizations."""
    ns = sorted({n for (n, _) in resolved})
    rows = []
    for n in ns:
        recs = [resolved[(n, r)] for r in range(n_reps)
                if (n, r) in resolved and resolved[(n, r)]['status'] == 'ok']
        if not recs:
            continue
        sup = np.array([r['sup'] for r in recs])
        l2 = np.array([r['l2'] for r in recs])
        mse = np.array([r['MSE'] for r in recs])
        rows.append({
            'n': n, 'm': recs[0]['m'], 'n_ok': len(recs),
            'PGD_sup_mean': sup.mean(),
            'PGD_sup_std': sup.std(ddof=1) if len(sup) > 1 else 0.0,
            'PGD_l2_mean': l2.mean(),
            'PGD_l2_std': l2.std(ddof=1) if len(l2) > 1 else 0.0,
            'PGD_MSE_mean': mse.mean(),
        })
    return pd.DataFrame(rows).set_index('n')


# ── Figure ───────────────────────────────────────────────────────────────────
def emit_figure(df: pd.DataFrame, out_path: Path, *, per_n2: float) -> None:
    import matplotlib.pyplot as plt
    mm = _plot_style.mm
    fig, ax = plt.subplots(figsize=(130 * mm, 100 * mm), layout='constrained')

    n = df.index.to_numpy(dtype=float)
    mean = df['PGD_sup_mean'].to_numpy(dtype=float)
    std = df['PGD_sup_std'].to_numpy(dtype=float)
    mask = np.isfinite(mean)

    # Mean sup-norm fitted-vs-true CDF distance, with a +/-1 std band.
    if mask.any():
        lower = np.clip(mean - std, 1e-12, None)
        upper = mean + std
        ax.fill_between(n[mask], lower[mask], upper[mask],
                        color='#0072B2', alpha=0.18, linewidth=0,
                        label=r'$\pm 1$ std over realizations')
        ax.plot(n[mask], mean[mask], marker='o', color='#0072B2',
                label=r'PGD fit: mean discrepancy')

    # 1/N reference anchored to the first finite mean point.
    ref = df['PGD_sup_mean'].dropna()
    if not ref.empty:
        n0, y0 = float(ref.index[0]), float(ref.iloc[0])
        ax.plot(n, (y0 * n0) / n, 'k--', linewidth=1.0,
                label=r'$\mathcal{O}(1/n)$ reference')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$n$-degree (log scale)')
    ax.set_ylabel(r'Discrepancy (log scale)')
    ax.spines[['top', 'right']].set_visible(False)

    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _plot_style.save(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Fitted-vs-true CDF sup error of the restricted PGD fit -> 0 as N grows.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-reps', type=int, default=10,
                        help='Independent realizations averaged at each degree.')
    parser.add_argument('--n-min', type=int, default=3)
    parser.add_argument('--n-max', type=int, default=100)
    parser.add_argument('--num-degrees', type=int, default=18,
                        help='Number of log-spaced degrees between n-min and n-max.')
    parser.add_argument('--samples-per-n2', type=float, default=20.0,
                        help='m(N) = round(this * N^2). Default 20 -> m=2000 at N=10.')
    parser.add_argument('--m-min', type=int, default=200,
                        help='Lower clamp on m(N).')
    parser.add_argument('--m-cap', type=int, default=300_000,
                        help='Upper clamp on m(N) (memory/runtime ceiling).')
    parser.add_argument('--n-eval', type=int, default=1000,
                        help='Number of x-points for the true-CDF error grid.')
    parser.add_argument('--time-budget-min', type=float, default=55.0)
    parser.add_argument('--pgd-cap-s', type=float, default=240.0,
                        help='Per-fit time cap for PGD; the sweep stops past it.')
    parser.add_argument('--pgd-iter-per-n', type=float, default=12.0)
    parser.add_argument('--pgd-min-iter', type=int, default=500)
    parser.add_argument('--pgd-max-iter-cap', type=int, default=10000)
    parser.add_argument('--pgd-threshold', type=float, default=1e-5)
    parser.add_argument('--no-resume', action='store_true',
                        help='Ignore and overwrite any existing checkpoint.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'figures')
    args = parser.parse_args()

    _plot_style.apply()

    checkpoint_path = args.output_dir / 'restriction_rate_checkpoint.csv'
    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f'[fig_restriction_rate] removed existing checkpoint {checkpoint_path}')

    x_eval = np.linspace(1e-6, 1 - 1e-6, args.n_eval)
    true_F = true_cdf_on_grid(x_eval)

    n_values = log_spaced_degrees(args.n_min, args.n_max, args.num_degrees)
    m_values = [samples_for_degree(int(n), per_n2=args.samples_per_n2,
                                   m_min=args.m_min, m_cap=args.m_cap)
                for n in n_values]
    print(f'[fig_restriction_rate] {args.n_reps} reps x {len(n_values)} degrees')
    print(f'[fig_restriction_rate] degrees:           {list(n_values)}')
    print(f'[fig_restriction_rate] sample sizes m(N): {m_values}')

    resolved = run_sweep(n_values, x_eval, true_F,
                         seed=args.seed, n_reps=args.n_reps,
                         per_n2=args.samples_per_n2,
                         m_min=args.m_min, m_cap=args.m_cap,
                         budget_s=args.time_budget_min * 60.0,
                         pgd_cap_s=args.pgd_cap_s,
                         checkpoint_path=checkpoint_path,
                         pgd_iter_per_n=args.pgd_iter_per_n,
                         pgd_min_iter=args.pgd_min_iter,
                         pgd_max_iter_cap=args.pgd_max_iter_cap,
                         pgd_threshold=args.pgd_threshold)

    df = aggregate_to_wide(resolved, args.n_reps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'restriction_rate_raw.csv'
    df.to_csv(csv_path)
    print(f'[fig_restriction_rate] Wrote {csv_path}')

    out = args.output_dir / 'restriction_rate.pgf'
    emit_figure(df, out, per_n2=args.samples_per_n2)
    print(f'[fig_restriction_rate] Wrote {out}')


if __name__ == '__main__':
    main()
