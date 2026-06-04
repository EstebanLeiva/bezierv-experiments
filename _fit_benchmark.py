"""Benchmark helpers for the synthetic and Chicago-arc fitting experiments."""
from __future__ import annotations

import time as _time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from bezierv.classes.distfit import (
    DistFit,
    MLEOptions,
    NelderMeadOptions,
    NonLinearOptions,
    ProjGradOptions,
)

ARC_MSE_COLUMNS = [
    'regime', 'arc', 'n_samples', 'n_bezier',
    'method', 'time_s', 'metric_name', 'metric', 'failed',
]

IPOPT_MAX_CPU_TIME_S = 60.0


def _mle_options() -> MLEOptions:
    return MLEOptions(max_iter=1000, tol=1e-3, tol_res_root=1e-5,
                      tol_lambda_root=1e-5, max_iters_root=100)


def _mse_options(algorithm: str):
    if algorithm == 'projgrad':
        return ProjGradOptions(max_iter=200, threshold=1e-3)
    if algorithm == 'nonlinear':
        # IPOPT options must be nested under 'options' so Pyomo forwards them
        # to the solver instead of treating them as writer io_options.
        return NonLinearOptions(
            solver_options={'options': {'max_cpu_time': IPOPT_MAX_CPU_TIME_S}}
        )
    if algorithm == 'neldermead':
        return NelderMeadOptions(max_iter=200)
    raise ValueError(f'Unknown MSE algorithm: {algorithm}')

# Paper-labelled families (matches Table 1 / Table 2 row order).
SOURCE_FAMILIES = [
    'beta',
    'bimodal_gaussian',
    'exponential',
    'gamma',
    'log_logistic',
    'lognormal',
    'normal',
    'triangular',
    'trimodal_gaussian',
    'uniform',
    'weibull',
]


# ── Synthetic data generation ────────────────────────────────────────────────
def _sample_family(name: str, rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """Draw *n_samples* sorted observations from a random instance of *name*."""
    rs = int(rng.integers(0, 2 ** 31))

    if name == 'uniform':
        loc, scale = rng.uniform(1, 10), rng.uniform(1, 20)
        data = stats.uniform.rvs(loc=loc, scale=scale, size=n_samples, random_state=rs)
    elif name == 'exponential':
        loc, scale = rng.uniform(0.5, 5), rng.uniform(0.5, 10)
        data = stats.expon.rvs(loc=loc, scale=scale, size=n_samples, random_state=rs)
    elif name == 'gamma':
        a, loc, scale = rng.uniform(0.5, 8), rng.uniform(0.1, 3), rng.uniform(0.5, 5)
        data = stats.gamma.rvs(a, loc=loc, scale=scale, size=n_samples, random_state=rs)
    elif name == 'weibull':
        c, loc, scale = rng.uniform(0.5, 5), rng.uniform(0.1, 3), rng.uniform(0.5, 5)
        data = stats.weibull_min.rvs(c, loc=loc, scale=scale, size=n_samples, random_state=rs)
    elif name == 'normal':
        loc, scale = rng.uniform(5, 20), rng.uniform(0.5, 4)
        data = stats.norm.rvs(loc=loc, scale=scale, size=n_samples, random_state=rs)
    elif name == 'lognormal':
        s, loc, scale = rng.uniform(0.1, 1.2), rng.uniform(0.0, 2), rng.uniform(0.5, 5)
        data = stats.lognorm.rvs(s, loc=loc, scale=scale, size=n_samples, random_state=rs)
    elif name == 'beta':
        a, b = rng.uniform(0.5, 5), rng.uniform(0.5, 5)
        loc, scale = rng.uniform(1, 5), rng.uniform(1, 20)
        data = stats.beta.rvs(a, b, loc=loc, scale=scale, size=n_samples, random_state=rs)
    elif name == 'log_logistic':
        c, loc, scale = rng.uniform(1.5, 8), rng.uniform(0.1, 3), rng.uniform(0.5, 5)
        data = stats.fisk.rvs(c, loc=loc, scale=scale, size=n_samples, random_state=rs)
    elif name == 'triangular':
        c, loc, scale = rng.uniform(0.1, 0.9), rng.uniform(1, 5), rng.uniform(2, 20)
        data = stats.triang.rvs(c, loc=loc, scale=scale, size=n_samples, random_state=rs)
    elif name == 'bimodal_gaussian':
        mu1 = rng.uniform(3, 10)
        mu2 = rng.uniform(mu1 + 3, mu1 + 15)
        sig1, sig2 = rng.uniform(0.5, 3), rng.uniform(0.5, 3)
        w = rng.uniform(0.3, 0.7)
        n1 = int(round(w * n_samples))
        n2 = n_samples - n1
        data = np.concatenate([rng.standard_normal(n1) * sig1 + mu1,
                               rng.standard_normal(n2) * sig2 + mu2])
    elif name == 'trimodal_gaussian':
        mu1 = rng.uniform(2, 6)
        mu2 = rng.uniform(mu1 + 3, mu1 + 10)
        mu3 = rng.uniform(mu2 + 3, mu2 + 10)
        sig1 = rng.uniform(0.3, 2); sig2 = rng.uniform(0.3, 2); sig3 = rng.uniform(0.3, 2)
        w = rng.dirichlet([1.0, 1.0, 1.0])
        n1 = int(round(w[0] * n_samples))
        n2 = int(round(w[1] * n_samples))
        n3 = n_samples - n1 - n2
        data = np.concatenate([rng.standard_normal(n1) * sig1 + mu1,
                               rng.standard_normal(n2) * sig2 + mu2,
                               rng.standard_normal(n3) * sig3 + mu3])
    else:
        raise ValueError(f'Unknown family: {name}')

    return np.sort(data.astype(np.float64))


# ── Fit wrappers ─────────────────────────────────────────────────────────────
def _fit_bezier_mle(data: np.ndarray, n: int) -> tuple[float, float, bool]:
    """Returns (elapsed_seconds, nll, failed)."""
    try:
        try:
            df = DistFit(data, n=n)
        except ValueError:
            df = DistFit(data, n=n, method_init_x='uniform')
        t0 = _time.perf_counter()
        _, nll = df.fit(method='mle', options=_mle_options())
        return _time.perf_counter() - t0, float(nll), False
    except Exception:
        return float('nan'), float('nan'), True


def _fit_bezier_mse(data: np.ndarray, algorithm: str, n: int) -> tuple[float, float, bool]:
    """Returns (elapsed_seconds, mse, failed). algorithm ∈ {nonlinear, projgrad, neldermead}."""
    try:
        try:
            df = DistFit(data, n=n)
        except ValueError:
            df = DistFit(data, n=n, method_init_x='uniform')
        t0 = _time.perf_counter()
        _, mse = df.fit(method='mse', algorithm=algorithm, options=_mse_options(algorithm))
        return _time.perf_counter() - t0, float(mse), False
    except Exception:
        return float('nan'), float('nan'), True


def _fit_scipy_mle(scipy_dist, data: np.ndarray) -> tuple[float, float, bool]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t0 = _time.perf_counter()
            params = scipy_dist.fit(data)
            elapsed = _time.perf_counter() - t0
        logpdf = scipy_dist.logpdf(data, *params)
        if not np.all(np.isfinite(logpdf)):
            return elapsed, float('nan'), True
        nll = -float(np.sum(logpdf))
        if not np.isfinite(nll):
            return elapsed, float('nan'), True
        return elapsed, nll, False
    except Exception:
        return float('nan'), float('nan'), True


# ── Synthetic benchmark (paper Tables 1 + 2) ─────────────────────────────────
def run_synthetic_benchmark(
    *,
    n_per_family: int,
    n_samples: int = 1000,
    n_bezier: int = 10,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run all 6 fits on each of n_per_family×len(SOURCE_FAMILIES) instances.

    Returns a long-format DataFrame with columns:
        family, instance, method, time_s, metric_name, metric, failed
    where method ∈ {bezier_mle, gen_beta_mle, johnson_mle,
                    bezier_pgd, bezier_ipopt, bezier_nm}
    and metric_name ∈ {nll, mse}.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    total = n_per_family * len(SOURCE_FAMILIES)
    pbar = tqdm(total=total, desc='Synthetic benchmark', unit='instance', disable=not verbose)

    for family in SOURCE_FAMILIES:
        for inst in range(n_per_family):
            data = _sample_family(family, rng, n_samples)

            # --- MLE ---------------------------------------------------------
            elapsed, nll, failed = _fit_bezier_mle(data, n=n_bezier)
            records.append({'family': family, 'instance': inst, 'method': 'bezier_mle',
                            'time_s': elapsed, 'metric_name': 'nll', 'metric': nll, 'failed': failed})

            elapsed, nll, failed = _fit_scipy_mle(stats.beta, data)
            records.append({'family': family, 'instance': inst, 'method': 'gen_beta_mle',
                            'time_s': elapsed, 'metric_name': 'nll', 'metric': nll, 'failed': failed})

            elapsed, nll, failed = _fit_scipy_mle(stats.johnsonsu, data)
            records.append({'family': family, 'instance': inst, 'method': 'johnson_mle',
                            'time_s': elapsed, 'metric_name': 'nll', 'metric': nll, 'failed': failed})

            # --- MSE ---------------------------------------------------------
            for algo, method_name in [('projgrad', 'bezier_pgd'),
                                      ('nonlinear', 'bezier_ipopt'),
                                      ('neldermead', 'bezier_nm')]:
                elapsed, mse, failed = _fit_bezier_mse(data, algorithm=algo, n=n_bezier)
                records.append({'family': family, 'instance': inst, 'method': method_name,
                                'time_s': elapsed, 'metric_name': 'mse', 'metric': mse, 'failed': failed})

            pbar.update(1)

    pbar.close()
    return pd.DataFrame.from_records(records)


# ── Chicago per-arc benchmarks (paper Tables 4 + 5) ──────────────────────────
def run_arc_mle_benchmark(
    arcs: dict,
    *,
    n_bezier: int,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit Bezier MLE, generalised Beta, and Johnson SU on every arc.

    Returns a long-format DataFrame: arc, method, time_s, metric_name='nll',
    metric, failed.
    """
    records: list[dict] = []
    items = sorted(arcs.items(), key=lambda kv: str(kv[0]))  # deterministic
    pbar = tqdm(items, desc=f'Arc MLE (n={n_bezier})', unit='arc', disable=not verbose)

    for key, value in pbar:
        sample = np.sort(np.asarray(value['travel_time'], dtype=np.float64))
        arc_label = f"{value.get('start_node', key[0])}->{value.get('end_node', key[1])}"

        elapsed, nll, failed = _fit_bezier_mle(sample, n=n_bezier)
        records.append({'arc': arc_label, 'n_samples': len(sample), 'method': 'bezier_mle',
                        'time_s': elapsed, 'metric_name': 'nll', 'metric': nll, 'failed': failed})

        elapsed, nll, failed = _fit_scipy_mle(stats.beta, sample)
        records.append({'arc': arc_label, 'n_samples': len(sample), 'method': 'gen_beta_mle',
                        'time_s': elapsed, 'metric_name': 'nll', 'metric': nll, 'failed': failed})

        elapsed, nll, failed = _fit_scipy_mle(stats.johnsonsu, sample)
        records.append({'arc': arc_label, 'n_samples': len(sample), 'method': 'johnson_mle',
                        'time_s': elapsed, 'metric_name': 'nll', 'metric': nll, 'failed': failed})

    return pd.DataFrame.from_records(records)


def run_arc_mse_benchmark(
    arcs: dict,
    *,
    n_bezier: int,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Fit Bezier MSE via PGD, IPOPT, and NM on every arc.

    Returns a long-format DataFrame: arc, method, time_s, metric_name='mse',
    metric, failed.
    """
    records: list[dict] = []
    items = sorted(arcs.items(), key=lambda kv: str(kv[0]))
    pbar = tqdm(items, desc=f'Arc MSE (n={n_bezier})', unit='arc', disable=not verbose)

    for key, value in pbar:
        sample = np.sort(np.asarray(value['travel_time'], dtype=np.float64))
        arc_label = f"{value.get('start_node', key[0])}->{value.get('end_node', key[1])}"

        for algo, method_name in [('projgrad', 'bezier_pgd'),
                                  ('nonlinear', 'bezier_ipopt'),
                                  ('neldermead', 'bezier_nm')]:
            elapsed, mse, failed = _fit_bezier_mse(sample, algorithm=algo, n=n_bezier)
            records.append({'arc': arc_label, 'n_samples': len(sample), 'method': method_name,
                            'time_s': elapsed, 'metric_name': 'mse', 'metric': mse, 'failed': failed})

    return pd.DataFrame.from_records(records)


def _append_arc_mse_row(checkpoint_path: Path, row: dict) -> None:
    """Append a single result row to *checkpoint_path*, writing the header on first write.

    Uses a fixed column order (ARC_MSE_COLUMNS) so reruns produce a consistent CSV.
    Writes are flushed immediately so a Ctrl-C does not lose the latest result.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not checkpoint_path.exists()
    frame = pd.DataFrame([{c: row.get(c) for c in ARC_MSE_COLUMNS}], columns=ARC_MSE_COLUMNS)
    with open(checkpoint_path, 'a', newline='') as f:
        frame.to_csv(f, header=write_header, index=False)
        f.flush()


def _load_done_arc_methods(checkpoint_path: Path, regime: str) -> set[tuple[str, str]]:
    """Return the set of (arc, method) pairs already recorded for *regime*."""
    if not checkpoint_path.exists():
        return set()
    df = pd.read_csv(checkpoint_path)
    if df.empty or 'arc' not in df.columns or 'method' not in df.columns:
        return set()
    if 'regime' in df.columns:
        df = df[df['regime'] == regime]
    return set(zip(df['arc'].astype(str), df['method'].astype(str)))


def run_arc_mse_benchmark_resumable(
    arcs: dict,
    *,
    n_bezier: int,
    regime: str,
    checkpoint_path: Path,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Resumable variant of :func:`run_arc_mse_benchmark`.

    Each (arc, method) fit is written to *checkpoint_path* as it completes, so a
    partial run resumes by re-invoking with the same arcs and checkpoint; pairs
    already recorded for *regime* are skipped. Returns the rows for *regime*
    loaded back from the checkpoint (fresh results + any prior partial work).
    """
    checkpoint_path = Path(checkpoint_path)
    done = _load_done_arc_methods(checkpoint_path, regime)

    items = sorted(arcs.items(), key=lambda kv: str(kv[0]))
    methods = [('projgrad', 'bezier_pgd'),
               ('nonlinear', 'bezier_ipopt'),
               ('neldermead', 'bezier_nm')]
    total = len(items) * len(methods)
    already = sum(1 for key, value in items
                  for _, m in methods
                  if (f"{value.get('start_node', key[0])}->{value.get('end_node', key[1])}", m) in done)

    pbar = tqdm(total=total, initial=already,
                desc=f'Arc MSE (n={n_bezier}, {regime})', unit='fit',
                disable=not verbose)

    for key, value in items:
        sample = np.sort(np.asarray(value['travel_time'], dtype=np.float64))
        arc_label = f"{value.get('start_node', key[0])}->{value.get('end_node', key[1])}"
        for algo, method_name in methods:
            if (arc_label, method_name) in done:
                continue
            elapsed, mse, failed = _fit_bezier_mse(sample, algorithm=algo, n=n_bezier)
            _append_arc_mse_row(checkpoint_path, {
                'regime': regime,
                'arc': arc_label,
                'n_samples': len(sample),
                'n_bezier': n_bezier,
                'method': method_name,
                'time_s': elapsed,
                'metric_name': 'mse',
                'metric': mse,
                'failed': failed,
            })
            done.add((arc_label, method_name))
            pbar.update(1)

    pbar.close()

    df = pd.read_csv(checkpoint_path)
    if 'regime' in df.columns:
        df = df[df['regime'] == regime]
    return df.reset_index(drop=True)
