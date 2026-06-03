#!/usr/bin/env python3
"""Reproduce Table 5 (`tab:MDEtab`): MSE comparison on the Chicago travel-time
dataset across the three time regimes (PGD vs IPOPT vs Nelder-Mead).

Output: tables/MDEtab.tex
        tables/realmse_raw.csv                       (per-arc long-format results)
        tables/realmse_partial_{regime}.csv          (incremental checkpoint per regime)

The script is resumable: each (arc, method) fit is appended to a per-regime
checkpoint CSV as soon as it completes. If the run is interrupted (Ctrl-C, crash,
timeout), simply re-invoke with the same arguments and it will pick up where it
left off. To start fresh, delete the partial CSVs (or pass --fresh).

Usage:
    python tab_real_mse.py                       # smoke run, 20 arcs per regime
    python tab_real_mse.py --max-arcs 0          # paper-faithful (all arcs, very slow!)
    python tab_real_mse.py --regime rush_hour    # run only one regime (skips final table)
    python tab_real_mse.py --fresh               # ignore and remove existing partials
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _chicago_graph as cg
import _fit_benchmark as fb
import _latex_tables as lt

REGIMES_ORDER = ['rush_hour', 'daytime', 'full_day']


def _agg(df: pd.DataFrame, method: str) -> dict:
    sub = df[df['method'] == method]
    if len(sub) == 0:
        return {'time_mean': float('nan'), 'time_std': float('nan'),
                'metric_mean': float('nan'), 'metric_std': float('nan'),
                'fail_rate': float('nan')}
    valid = sub[~sub['failed']]
    fail_rate = float(sub['failed'].mean()) if len(sub) else float('nan')
    if len(valid) == 0:
        return {'time_mean': float('nan'), 'time_std': float('nan'),
                'metric_mean': float('nan'), 'metric_std': float('nan'),
                'fail_rate': fail_rate}
    return {
        'time_mean':   valid['time_s'].mean() * 1000.0,
        'time_std':    valid['time_s'].std()  * 1000.0,
        'metric_mean': valid['metric'].mean(),
        'metric_std':  valid['metric'].std(),
        'fail_rate':   fail_rate,
    }


def emit_table(regime_dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    lines = []
    lines.append('        \\toprule\n')
    lines.append('        & & \\multicolumn{5}{c}{PGD} & \\multicolumn{5}{c}{IPOPT} & \\multicolumn{5}{c}{NM} \\\\\n')
    lines.append('        \\cmidrule(lr){3-7} \\cmidrule(lr){8-12} \\cmidrule(lr){13-17}\n')
    lines.append('        & & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{MSE} & & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{MSE} & & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{MSE} & \\\\\n')
    lines.append('        \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11} \\cmidrule(lr){13-14} \\cmidrule(lr){15-16}\n')
    lines.append('        time frame & $n$ & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{fail} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{fail} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{fail} \\\\\n')
    lines.append('        \\midrule\n')

    avg = {m: {'tm': [], 'ts': [], 'mm': [], 'ms': [], 'fr': []}
           for m in ('bezier_pgd', 'bezier_ipopt', 'bezier_nm')}

    for regime in REGIMES_ORDER:
        df = regime_dfs[regime]
        n = cg.REGIME_BEZIER_DEGREE[regime]
        p = _agg(df, 'bezier_pgd')
        i = _agg(df, 'bezier_ipopt')
        nm = _agg(df, 'bezier_nm')
        lines.append(
            f"        {cg.REGIME_DISPLAY[regime]:12s} & {n} "
            f"& {lt.fmt_float(p['time_mean'])} & {lt.fmt_float(p['time_std'])} "
            f"& {lt.fmt_float(p['metric_mean'], 4)} & {lt.fmt_float(p['metric_std'], 4)} "
            f"& {lt.fmt_pct(p['fail_rate'], 1)} "
            f"& {lt.fmt_float(i['time_mean'])} & {lt.fmt_float(i['time_std'])} "
            f"& {lt.fmt_float(i['metric_mean'], 4)} & {lt.fmt_float(i['metric_std'], 4)} "
            f"& {lt.fmt_pct(i['fail_rate'], 1)} "
            f"& {lt.fmt_float(nm['time_mean'])} & {lt.fmt_float(nm['time_std'])} "
            f"& {lt.fmt_float(nm['metric_mean'], 4)} & {lt.fmt_float(nm['metric_std'], 4)} "
            f"& {lt.fmt_pct(nm['fail_rate'], 1)} "
            "\\\\\n"
        )
        for d, m in ((p, 'bezier_pgd'), (i, 'bezier_ipopt'), (nm, 'bezier_nm')):
            avg[m]['tm'].append(d['time_mean']); avg[m]['ts'].append(d['time_std'])
            avg[m]['mm'].append(d['metric_mean']); avg[m]['ms'].append(d['metric_std'])
            avg[m]['fr'].append(d['fail_rate'])

    lines.append('        \\midrule\n')
    am = {m: {k: float(np.nanmean(v)) for k, v in vals.items()} for m, vals in avg.items()}
    # Paper highlights: PGD time mean/std (best speed) and IPOPT MSE mean/std (best MSE).
    lines.append(
        f"        \\multicolumn{{2}}{{l}}{{average}} "
        f"& {lt.bold(lt.fmt_float(am['bezier_pgd']['tm']))} & {lt.bold(lt.fmt_float(am['bezier_pgd']['ts']))} "
        f"& {lt.fmt_float(am['bezier_pgd']['mm'], 4)} & {lt.fmt_float(am['bezier_pgd']['ms'], 4)} "
        f"& {lt.bold(lt.fmt_pct(am['bezier_pgd']['fr'], 1))} "
        f"& {lt.fmt_float(am['bezier_ipopt']['tm'])} & {lt.fmt_float(am['bezier_ipopt']['ts'])} "
        f"& {lt.bold(lt.fmt_float(am['bezier_ipopt']['mm'], 4))} & {lt.bold(lt.fmt_float(am['bezier_ipopt']['ms'], 4))} "
        f"& {lt.fmt_pct(am['bezier_ipopt']['fr'], 1)} "
        f"& {lt.fmt_float(am['bezier_nm']['tm'])} & {lt.fmt_float(am['bezier_nm']['ts'])} "
        f"& {lt.fmt_float(am['bezier_nm']['mm'], 4)} & {lt.fmt_float(am['bezier_nm']['ms'], 4)} "
        f"& {lt.fmt_pct(am['bezier_nm']['fr'], 1)} "
        "\\\\\n"
    )
    lines.append('        \\bottomrule\n')

    lt.emit_table(
        out_path,
        body=''.join(lines),
        caption='MSE comparison (transportation data).',
        label='tab:MDEtab',
        column_spec='lc rr rr c rr rr c rr rr c',
    )


def _partial_path(output_dir: Path, regime: str) -> Path:
    return output_dir / f'realmse_partial_{regime}.csv'


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce paper Table 5 (Chicago MSE).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-arcs', type=int, default=20,
                        help='Max arcs per regime; 0 = all (paper-faithful).')
    parser.add_argument('--regime', choices=REGIMES_ORDER, default=None,
                        help='Run only this regime; omit to run all three '
                             '(table+CSV are only emitted on a full run).')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'tables')
    parser.add_argument('--fresh', action='store_true',
                        help='Delete any existing partial checkpoints before running.')
    args = parser.parse_args()

    regimes = [args.regime] if args.regime else REGIMES_ORDER

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for regime in regimes:
            p = _partial_path(args.output_dir, regime)
            if p.exists():
                p.unlink()
                print(f'[tab_real_mse] removed stale checkpoint {p}')

    print('[tab_real_mse] Loading Chicago_main.json …')
    raw = cg.load_raw()

    regime_dfs: dict[str, pd.DataFrame] = {}
    all_rows = []
    for regime in regimes:
        n = cg.REGIME_BEZIER_DEGREE[regime]
        print(f'[tab_real_mse] {regime} (n={n}) — preprocessing')
        proc = cg.preprocess_regime(raw, regime, seed=args.seed)
        proc = cg.filter_multi_obs(proc, min_obs=2)
        if args.max_arcs > 0:
            proc = cg.sample_arcs(proc, args.max_arcs, seed=args.seed)
        checkpoint = _partial_path(args.output_dir, regime)
        print(f'[tab_real_mse] {regime}: fitting on {len(proc)} arcs '
              f'(checkpoint: {checkpoint})')
        df = fb.run_arc_mse_benchmark_resumable(
            proc, n_bezier=n, regime=regime, checkpoint_path=checkpoint,
            seed=args.seed,
        )
        # regime/n_bezier are already in the checkpoint rows, but coerce in case
        # an older partial file is missing them.
        if 'regime' not in df.columns:
            df['regime'] = regime
        if 'n_bezier' not in df.columns:
            df['n_bezier'] = n
        regime_dfs[regime] = df
        all_rows.append(df)

    if args.regime is not None:
        print(f"[tab_real_mse] Single-regime run ({args.regime}) complete; "
              f"skipping MDEtab.tex and realmse_raw.csv. "
              f"Re-run without --regime once all three checkpoints exist to emit them.")
        return

    full = pd.concat(all_rows, ignore_index=True)
    csv_path = args.output_dir / 'realmse_raw.csv'
    full.to_csv(csv_path, index=False)
    print(f'[tab_real_mse] Wrote {csv_path}  ({len(full)} rows)')

    emit_table(regime_dfs, args.output_dir / 'MDEtab.tex')
    print(f"[tab_real_mse] Wrote {args.output_dir / 'MDEtab.tex'}")

    fail = full.groupby(['regime', 'method'])['failed'].mean().unstack()
    print(f'[tab_real_mse] Per-regime fail rates:\n{fail}')


if __name__ == '__main__':
    main()
