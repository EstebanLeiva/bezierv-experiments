#!/usr/bin/env python3
"""Reproduce Table 3 (`tab:realdist`): descriptive statistics for the Chicago
travel-time dataset across the three time regimes.

Output: tables/realdist.tex

Usage:
    python tab_real_stats.py
"""
import argparse
from pathlib import Path

import numpy as np

import _chicago_graph as cg
import _latex_tables as lt

REGIMES_ORDER = ['rush_hour', 'daytime', 'full_day']


def _arc_stats(processed: dict) -> dict:
    """Per-regime aggregate stats matching the paper Table 3 layout."""
    counts = np.array([len(v['travel_time']) for v in processed.values()])
    means_min = np.array([float(np.mean(v['travel_time'])) for v in processed.values()]) * 60.0       # → minutes
    stds_min  = np.array([float(np.std(v['travel_time'], ddof=1)) for v in processed.values()]) * 60.0
    ranges_min = np.array([float(max(v['travel_time']) - min(v['travel_time'])) for v in processed.values()]) * 60.0
    return {
        'n_obs_mean': float(counts.mean()),
        'n_obs_std':  float(counts.std(ddof=1)),
        'n_obs_min':  int(counts.min()),
        'n_obs_max':  int(counts.max()),
        'tt_mean':    float(means_min.mean()),
        'tt_std':     float(stds_min.mean()),
        'tt_range':   float(ranges_min.mean()),
    }


def emit_table(stats: dict[str, dict], out_path: Path) -> None:
    rh, day, fd = stats['rush_hour'], stats['daytime'], stats['full_day']
    lines = []
    lines.append('        \\hline\n')
    lines.append('        & statistic & \\multicolumn{1}{c}{Rush hour} & \\multicolumn{1}{c}{Daytime} & \\multicolumn{1}{c}{Full-day} \\\\\n')
    lines.append('        \\hline\n')
    lines.append(f"        \\multirow{{4}}{{*}}{{\\# obs.}} & mean  & {lt.fmt_float(rh['n_obs_mean'])} & {lt.fmt_float(day['n_obs_mean'])} & {lt.fmt_float(fd['n_obs_mean'])} \\\\\n")
    lines.append(f"                                 & std   & {lt.fmt_float(rh['n_obs_std'])} & {lt.fmt_float(day['n_obs_std'])} & {lt.fmt_float(fd['n_obs_std'])} \\\\\n")
    lines.append(f"                                 & min   & {lt.fmt_int(rh['n_obs_min'])} & {lt.fmt_int(day['n_obs_min'])} & {lt.fmt_int(fd['n_obs_min'])} \\\\\n")
    lines.append(f"                                 & max   & {lt.fmt_int(rh['n_obs_max'])} & {lt.fmt_int(day['n_obs_max'])} & {lt.fmt_int(fd['n_obs_max'])} \\\\\n")
    lines.append('        \\hline\n')
    lines.append(f"        \\multirow{{3}}{{*}}{{travel time}} & mean  & {lt.fmt_float(rh['tt_mean'], 2)} & {lt.fmt_float(day['tt_mean'], 2)} & {lt.fmt_float(fd['tt_mean'], 2)} \\\\\n")
    lines.append(f"                                     & std   & {lt.fmt_float(rh['tt_std'], 2)} & {lt.fmt_float(day['tt_std'], 2)} & {lt.fmt_float(fd['tt_std'], 2)} \\\\\n")
    lines.append(f"                                     & range & {lt.fmt_float(rh['tt_range'], 2)} & {lt.fmt_float(day['tt_range'], 2)} & {lt.fmt_float(fd['tt_range'], 2)} \\\\\n")
    lines.append('        \\hline\n')

    lt.emit_table(
        out_path,
        body=''.join(lines),
        caption=('Distribution of the number of observations per arc and mean values of '
                 'arc-level travel time means, standard deviations, and ranges across '
                 'different time regimes.'),
        label='tab:realdist',
        column_spec='ll|rrr',
        resizebox='0.5\\textwidth',
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce paper Table 3 (Chicago descriptive stats).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'tables')
    args = parser.parse_args()

    print('[tab_real_stats] Loading Chicago_main.json …')
    raw = cg.load_raw()
    print(f'[tab_real_stats] {len(raw)} raw records')

    regime_stats = {}
    for regime in REGIMES_ORDER:
        print(f'[tab_real_stats] Preprocessing {regime} …')
        proc = cg.preprocess_regime(raw, regime, seed=args.seed)
        proc = cg.filter_multi_obs(proc, min_obs=2)
        print(f'                 {regime}: {len(proc)} arcs')
        regime_stats[regime] = _arc_stats(proc)

    out_path = args.output_dir / 'realdist.tex'
    emit_table(regime_stats, out_path)
    print(f'[tab_real_stats] Wrote {out_path}')


if __name__ == '__main__':
    main()
