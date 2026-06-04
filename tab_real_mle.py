#!/usr/bin/env python3
"""Reproduce Table 4 (`tab:realmle`): MLE comparison on the Chicago travel-time
dataset across the three time regimes.

Output: tables/realmle.tex
        tables/realmle_raw.csv  (per-arc long-format results)

Usage:
    python tab_real_mle.py                       # smoke run, 20 arcs per regime
    python tab_real_mle.py --max-arcs 0          # paper-faithful (all ~1086 arcs/regime, slow!)
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
    sub = df[(df['method'] == method) & ~df['failed']]
    if len(sub) == 0:
        return {'time_mean': float('nan'), 'time_std': float('nan'),
                'metric_mean': float('nan'), 'metric_std': float('nan')}
    return {
        'time_mean':   sub['time_s'].mean() * 1000.0,
        'time_std':    sub['time_s'].std()  * 1000.0,
        'metric_mean': sub['metric'].mean(),
        'metric_std':  sub['metric'].std(),
    }


def emit_table(regime_dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    lines = []
    lines.append('        \\toprule\n')
    lines.append('        & & \\multicolumn{4}{c}{B\\\'ezier} & \\multicolumn{4}{c}{Beta} & \\multicolumn{4}{c}{Johnson} \\\\\n')
    lines.append('        \\cmidrule(lr){3-6} \\cmidrule(lr){7-10} \\cmidrule(lr){11-14}\n')
    lines.append('        & & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{NLL} & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{NLL} & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{NLL} \\\\\n')
    lines.append('        \\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10} \\cmidrule(lr){11-12} \\cmidrule(lr){13-14}\n')
    lines.append('        time frame & $n$ & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} & \\multicolumn{1}{c}{mean} & \\multicolumn{1}{c}{std} \\\\\n')
    lines.append('        \\midrule\n')

    # Per-method standard deviations averaged across regimes for the final row
    avg = {m: {'tm': [], 'ts': [], 'mm': [], 'ms': []}
           for m in ('bezier_mle', 'gen_beta_mle', 'johnson_mle')}

    for regime in REGIMES_ORDER:
        df = regime_dfs[regime]
        n = cg.REGIME_BEZIER_DEGREE[regime]
        b = _agg(df, 'bezier_mle')
        β = _agg(df, 'gen_beta_mle')
        j = _agg(df, 'johnson_mle')
        lines.append(
            f"        {cg.REGIME_DISPLAY[regime]:12s} & {n} "
            f"& {lt.fmt_float(b['time_mean'])} & {lt.fmt_float(b['time_std'])} "
            f"& {lt.fmt_signed_nll(b['metric_mean'])} & {lt.fmt_float(b['metric_std'])} "
            f"& {lt.fmt_float(β['time_mean'])} & {lt.fmt_float(β['time_std'])} "
            f"& {lt.fmt_signed_nll(β['metric_mean'])} & {lt.fmt_float(β['metric_std'])} "
            f"& {lt.fmt_float(j['time_mean'])} & {lt.fmt_float(j['time_std'])} "
            f"& {lt.fmt_signed_nll(j['metric_mean'])} & {lt.fmt_float(j['metric_std'])} "
            "\\\\\n"
        )
        for d, m in ((b, 'bezier_mle'), (β, 'gen_beta_mle'), (j, 'johnson_mle')):
            avg[m]['tm'].append(d['time_mean']); avg[m]['ts'].append(d['time_std'])
            avg[m]['mm'].append(d['metric_mean']); avg[m]['ms'].append(d['metric_std'])

    lines.append('        \\midrule\n')
    am = {m: {k: float(np.nanmean(v)) for k, v in vals.items()} for m, vals in avg.items()}
    # Paper highlights smallest std NLL across the three methods on the average row
    stds_nll = {m: am[m]['ms'] for m in am}
    best_method = min(stds_nll, key=lambda k: stds_nll[k]) if any(np.isfinite(v) for v in stds_nll.values()) else None

    def _nll_std_cell(method: str) -> str:
        s = lt.fmt_float(am[method]['ms'])
        return lt.bold(s) if method == best_method else s

    lines.append(
        f"        \\multicolumn{{2}}{{l}}{{average}} "
        f"& {lt.fmt_float(am['bezier_mle']['tm'])} & {lt.fmt_float(am['bezier_mle']['ts'])} "
        f"& {lt.fmt_signed_nll(am['bezier_mle']['mm'])} & {_nll_std_cell('bezier_mle')} "
        f"& {lt.fmt_float(am['gen_beta_mle']['tm'])} & {lt.fmt_float(am['gen_beta_mle']['ts'])} "
        f"& {lt.fmt_signed_nll(am['gen_beta_mle']['mm'])} & {_nll_std_cell('gen_beta_mle')} "
        f"& {lt.fmt_float(am['johnson_mle']['tm'])} & {lt.fmt_float(am['johnson_mle']['ts'])} "
        f"& {lt.fmt_signed_nll(am['johnson_mle']['mm'])} & {_nll_std_cell('johnson_mle')} "
        "\\\\\n"
    )
    lines.append('        \\bottomrule\n')

    lt.emit_table(
        out_path,
        body=''.join(lines),
        caption='MLE comparison (transportation data).',
        label='tab:realmle',
        column_spec='lc rr rr rr rr rr rr',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce paper Table 4 (Chicago MLE).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-arcs', type=int, default=20,
                        help='Max arcs per regime; 0 = all (paper-faithful).')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'tables')
    args = parser.parse_args()

    print('[tab_real_mle] Loading Chicago_main.json …')
    raw = cg.load_raw()

    regime_dfs: dict[str, pd.DataFrame] = {}
    all_rows = []
    for regime in REGIMES_ORDER:
        n = cg.REGIME_BEZIER_DEGREE[regime]
        print(f'[tab_real_mle] {regime} (n={n}) — preprocessing')
        proc = cg.preprocess_regime(raw, regime, seed=args.seed)
        proc = cg.filter_multi_obs(proc, min_obs=2)
        if args.max_arcs > 0:
            proc = cg.sample_arcs(proc, args.max_arcs, seed=args.seed)
        print(f'[tab_real_mle] {regime}: fitting on {len(proc)} arcs')
        df = fb.run_arc_mle_benchmark(proc, n_bezier=n, seed=args.seed)
        df['regime'] = regime
        df['n_bezier'] = n
        regime_dfs[regime] = df
        all_rows.append(df)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    full = pd.concat(all_rows, ignore_index=True)
    csv_path = args.output_dir / 'realmle_raw.csv'
    full.to_csv(csv_path, index=False)
    print(f'[tab_real_mle] Wrote {csv_path}  ({len(full)} rows)')

    emit_table(regime_dfs, args.output_dir / 'realmle.tex')
    print(f"[tab_real_mle] Wrote {args.output_dir / 'realmle.tex'}")


if __name__ == '__main__':
    main()
