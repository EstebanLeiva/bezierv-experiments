#!/usr/bin/env python3
"""Reproduce Tables 1 (MLE) and 2 (MSE) of the paper from synthetic data.

Outputs:
    tables/syntheticmle.tex
    tables/syntheticmse.tex
    tables/synthetic_raw.csv      (per-instance long-format results, for traceability)

Usage:
    python tab_synthetic.py                          # smoke run, 3 instances per family
    python tab_synthetic.py --n-per-family 200       # paper-faithful (slow!)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _fit_benchmark as fb
import _latex_tables as lt


FAMILY_DISPLAY = {
    'beta':              'Beta',
    'bimodal_gaussian':  'Bimodal Gaussian',
    'exponential':       'Exponential',
    'gamma':             'Gamma',
    'log_logistic':      'Log Logistic',
    'lognormal':         'Lognormal',
    'normal':            'Normal',
    'triangular':        'Triangular',
    'trimodal_gaussian': 'Trimodal Gaussian',
    'uniform':           'Uniform',
    'weibull':           'Weibull',
}

# Multimodal Bezier NLL is bolded in paper Table 1.
BOLD_BEZIER_NLL = {'bimodal_gaussian', 'trimodal_gaussian'}


def _agg(df: pd.DataFrame, method: str, metric_col: str = 'metric') -> dict:
    """Return {family: {'time_mean':, 'time_std':, 'metric_mean':, 'metric_std':, 'fail_rate':}}."""
    sub = df[df['method'] == method]
    out: dict[str, dict] = {}
    for fam, g in sub.groupby('family'):
        valid = g.dropna(subset=['time_s', metric_col])
        out[fam] = {
            'time_mean':   valid['time_s'].mean() * 1000.0 if len(valid) else float('nan'),
            'time_std':    valid['time_s'].std()  * 1000.0 if len(valid) else float('nan'),
            'metric_mean': valid[metric_col].mean()         if len(valid) else float('nan'),
            'metric_std':  valid[metric_col].std()          if len(valid) else float('nan'),
            'fail_rate':   float(g['failed'].mean())        if len(g) else float('nan'),
        }
    # overall average across all instances (paper's "Average" row)
    all_valid = sub.dropna(subset=['time_s', metric_col])
    out['__average__'] = {
        'time_mean':   all_valid['time_s'].mean() * 1000.0 if len(all_valid) else float('nan'),
        'time_std':    all_valid['time_s'].std()  * 1000.0 if len(all_valid) else float('nan'),
        'metric_mean': all_valid[metric_col].mean()         if len(all_valid) else float('nan'),
        'metric_std':  all_valid[metric_col].std()          if len(all_valid) else float('nan'),
        'fail_rate':   float(sub['failed'].mean())          if len(sub) else float('nan'),
    }
    return out


# ── Table 1: MLE comparison ──────────────────────────────────────────────────
def emit_table_mle(df: pd.DataFrame, out_path: Path) -> None:
    bez = _agg(df, 'bezier_mle')
    bet = _agg(df, 'gen_beta_mle')
    joh = _agg(df, 'johnson_mle')

    lines = []
    lines.append('        \\toprule\n')
    lines.append('        & \\multicolumn{4}{c}{B\\\'ezier} & \\multicolumn{4}{c}{Beta} & \\multicolumn{4}{c}{Johnson} \\\\\n')
    lines.append('        \\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13}\n')
    lines.append('        & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{NLL} & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{NLL} & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{NLL} \\\\\n')
    lines.append('        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-13}\n')
    lines.append('        dist & mean & std & mean & std & mean & std & mean & std & mean & std & mean & std \\\\\n')
    lines.append('        \\midrule\n')

    for fam in fb.SOURCE_FAMILIES:
        b, β, j = bez[fam], bet[fam], joh[fam]
        nll_b = lt.fmt_float(b['metric_mean'])
        if fam in BOLD_BEZIER_NLL:
            nll_b = lt.bold(nll_b)
        row = (
            f"        {FAMILY_DISPLAY[fam]:<18s} "
            f"& {lt.fmt_float(b['time_mean'])} & {lt.fmt_float(b['time_std'])} "
            f"& {nll_b} & {lt.fmt_float(b['metric_std'])} "
            f"& {lt.fmt_float(β['time_mean'])} & {lt.fmt_float(β['time_std'])} "
            f"& {lt.fmt_float(β['metric_mean'])} & {lt.fmt_float(β['metric_std'])} "
            f"& {lt.fmt_float(j['time_mean'])} & {lt.fmt_float(j['time_std'])} "
            f"& {lt.fmt_float(j['metric_mean'])} & {lt.fmt_float(j['metric_std'])} "
            "\\\\\n"
        )
        lines.append(row)

    lines.append('        \\midrule\n')
    a_b, a_β, a_j = bez['__average__'], bet['__average__'], joh['__average__']
    lines.append(
        f"        Average            "
        f"& {lt.fmt_float(a_b['time_mean'])} & {lt.fmt_float(a_b['time_std'])} "
        f"& {lt.fmt_float(a_b['metric_mean'])} & {lt.fmt_float(a_b['metric_std'])} "
        f"& {lt.fmt_float(a_β['time_mean'])} & {lt.fmt_float(a_β['time_std'])} "
        f"& {lt.fmt_float(a_β['metric_mean'])} & {lt.fmt_float(a_β['metric_std'])} "
        f"& {lt.fmt_float(a_j['time_mean'])} & {lt.fmt_float(a_j['time_std'])} "
        f"& {lt.fmt_float(a_j['metric_mean'])} & {lt.fmt_float(a_j['metric_std'])} "
        "\\\\\n"
    )
    lines.append('        \\bottomrule\n')

    lt.emit_table(
        out_path,
        body=''.join(lines),
        caption='MLE comparison (synthetic data).',
        label='tab:syntheticmle',
        column_spec='l rr rr rr rr rr rr',
    )


# ── Table 2: MSE comparison ──────────────────────────────────────────────────
def emit_table_mse(df: pd.DataFrame, out_path: Path) -> None:
    pgd = _agg(df, 'bezier_pgd')
    ipt = _agg(df, 'bezier_ipopt')
    nm  = _agg(df, 'bezier_nm')

    show_fail = bool(
        df[df['method'].isin(['bezier_pgd', 'bezier_ipopt', 'bezier_nm'])]['failed'].any()
    )

    lines = []
    lines.append('        \\toprule\n')
    if show_fail:
        lines.append('        & \\multicolumn{5}{c}{PGD} & \\multicolumn{5}{c}{IPOPT} & \\multicolumn{5}{c}{NM} \\\\\n')
        lines.append('        \\cmidrule(lr){2-6} \\cmidrule(lr){7-11} \\cmidrule(lr){12-16}\n')
        lines.append('        & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{MSE} & & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{MSE} & & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{MSE} & \\\\\n')
        lines.append('        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){7-8} \\cmidrule(lr){9-10} \\cmidrule(lr){12-13} \\cmidrule(lr){14-15}\n')
        lines.append('        dist & mean & std & mean & std & fail & mean & std & mean & std & fail & mean & std & mean & std & fail \\\\\n')
    else:
        lines.append('        & \\multicolumn{4}{c}{PGD} & \\multicolumn{4}{c}{IPOPT} & \\multicolumn{4}{c}{NM} \\\\\n')
        lines.append('        \\cmidrule(lr){2-5} \\cmidrule(lr){6-9} \\cmidrule(lr){10-13}\n')
        lines.append('        & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{MSE} & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{MSE} & \\multicolumn{2}{c}{time (ms)} & \\multicolumn{2}{c}{MSE} \\\\\n')
        lines.append('        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9} \\cmidrule(lr){10-11} \\cmidrule(lr){12-13}\n')
        lines.append('        dist & mean & std & mean & std & mean & std & mean & std & mean & std & mean & std \\\\\n')
    lines.append('        \\midrule\n')

    for fam in fb.SOURCE_FAMILIES:
        p, i, n = pgd[fam], ipt[fam], nm[fam]
        if show_fail:
            row = (
                f"        {FAMILY_DISPLAY[fam]:<18s} "
                f"& {lt.fmt_float(p['time_mean'])} & {lt.fmt_float(p['time_std'])} "
                f"& {lt.fmt_sci(p['metric_mean'])} & {lt.fmt_sci(p['metric_std'])} "
                f"& {lt.fmt_pct(p['fail_rate'], 1)} "
                f"& {lt.fmt_float(i['time_mean'])} & {lt.fmt_float(i['time_std'])} "
                f"& {lt.fmt_sci(i['metric_mean'])} & {lt.fmt_sci(i['metric_std'])} "
                f"& {lt.fmt_pct(i['fail_rate'], 1)} "
                f"& {lt.fmt_float(n['time_mean'])} & {lt.fmt_float(n['time_std'])} "
                f"& {lt.fmt_sci(n['metric_mean'])} & {lt.fmt_sci(n['metric_std'])} "
                f"& {lt.fmt_pct(n['fail_rate'], 1)} "
                "\\\\\n"
            )
        else:
            row = (
                f"        {FAMILY_DISPLAY[fam]:<18s} "
                f"& {lt.fmt_float(p['time_mean'])} & {lt.fmt_float(p['time_std'])} "
                f"& {lt.fmt_sci(p['metric_mean'])} & {lt.fmt_sci(p['metric_std'])} "
                f"& {lt.fmt_float(i['time_mean'])} & {lt.fmt_float(i['time_std'])} "
                f"& {lt.fmt_sci(i['metric_mean'])} & {lt.fmt_sci(i['metric_std'])} "
                f"& {lt.fmt_float(n['time_mean'])} & {lt.fmt_float(n['time_std'])} "
                f"& {lt.fmt_sci(n['metric_mean'])} & {lt.fmt_sci(n['metric_std'])} "
                "\\\\\n"
            )
        lines.append(row)

    lines.append('        \\midrule\n')
    a_p, a_i, a_n = pgd['__average__'], ipt['__average__'], nm['__average__']
    if show_fail:
        lines.append(
            f"        Average            "
            f"& {lt.bold(lt.fmt_float(a_p['time_mean']))} & {lt.bold(lt.fmt_float(a_p['time_std']))} "
            f"& {lt.fmt_sci(a_p['metric_mean'])} & {lt.fmt_sci(a_p['metric_std'])} "
            f"& {lt.fmt_pct(a_p['fail_rate'], 1)} "
            f"& {lt.fmt_float(a_i['time_mean'])} & {lt.fmt_float(a_i['time_std'])} "
            f"& {lt.bold(lt.fmt_sci(a_i['metric_mean']))} & {lt.bold(lt.fmt_sci(a_i['metric_std']))} "
            f"& {lt.fmt_pct(a_i['fail_rate'], 1)} "
            f"& {lt.fmt_float(a_n['time_mean'])} & {lt.fmt_float(a_n['time_std'])} "
            f"& {lt.fmt_sci(a_n['metric_mean'])} & {lt.fmt_sci(a_n['metric_std'])} "
            f"& {lt.fmt_pct(a_n['fail_rate'], 1)} "
            "\\\\\n"
        )
    else:
        lines.append(
            f"        Average            "
            f"& {lt.bold(lt.fmt_float(a_p['time_mean']))} & {lt.bold(lt.fmt_float(a_p['time_std']))} "
            f"& {lt.fmt_sci(a_p['metric_mean'])} & {lt.fmt_sci(a_p['metric_std'])} "
            f"& {lt.fmt_float(a_i['time_mean'])} & {lt.fmt_float(a_i['time_std'])} "
            f"& {lt.bold(lt.fmt_sci(a_i['metric_mean']))} & {lt.bold(lt.fmt_sci(a_i['metric_std']))} "
            f"& {lt.fmt_float(a_n['time_mean'])} & {lt.fmt_float(a_n['time_std'])} "
            f"& {lt.fmt_sci(a_n['metric_mean'])} & {lt.fmt_sci(a_n['metric_std'])} "
            "\\\\\n"
        )
    lines.append('        \\bottomrule\n')

    column_spec = 'l rr rr c rr rr c rr rr c' if show_fail else 'l rr rr rr rr rr rr'
    lt.emit_table(
        out_path,
        body=''.join(lines),
        caption='MSE comparison (synthetic data).',
        label='tab:syntheticmse',
        column_spec=column_spec,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Reproduce paper Tables 1 (MLE) and 2 (MSE).')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-per-family', type=int, default=3,
                        help='Number of random instances per source family. Paper uses 200.')
    parser.add_argument('--n-samples', type=int, default=1000,
                        help='Sample size per instance. Paper uses 1000.')
    parser.add_argument('--n-bezier', type=int, default=10,
                        help='Bezier degree. Paper uses 10.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).parent / 'tables')
    args = parser.parse_args()

    print(f'[tab_synthetic] seed={args.seed}  n_per_family={args.n_per_family}  '
          f'n_samples={args.n_samples}  n_bezier={args.n_bezier}')

    df = fb.run_synthetic_benchmark(
        n_per_family=args.n_per_family,
        n_samples=args.n_samples,
        n_bezier=args.n_bezier,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / 'synthetic_raw.csv'
    df.to_csv(csv_path, index=False)
    print(f'[tab_synthetic] Wrote {csv_path}  ({len(df)} rows)')

    emit_table_mle(df, args.output_dir / 'syntheticmle.tex')
    print(f"[tab_synthetic] Wrote {args.output_dir / 'syntheticmle.tex'}")
    emit_table_mse(df, args.output_dir / 'syntheticmse.tex')
    print(f"[tab_synthetic] Wrote {args.output_dir / 'syntheticmse.tex'}")

    fail = df.groupby('method')['failed'].sum()
    if fail.any():
        print(f'[tab_synthetic] Failures by method:\n{fail[fail > 0]}')


if __name__ == '__main__':
    main()
