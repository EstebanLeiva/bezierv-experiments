import time as _time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from bezierv.classes.distfit import DistFit

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MLE_FIT_KWARGS: dict = {
    "max_iter":        1000,
    "tol":             1e-3,
    "tol_res_root":    1e-5,
    "tol_lambda_root": 1e-5,
    "max_iters_root":  100,
}

MLE_FIT_KWARGS_REAL: dict = {
    "max_iter":        1000,
    "tol":             1e-3,
    "tol_res_root":    1e-5,
    "tol_lambda_root": 1e-5,
    "max_iters_root":  100,
}

MSE_METHODS = ["nonlinear", "projgrad", "neldermead"]

MSE_METHOD_KWARGS: dict[str, dict] = {
    "nonlinear":  {},                       # uses default solver (ipopt)
    "projgrad":   {"step_size_PG": 1e-2, "max_iter_PG": 200, "threshold_PG": 0},
    "neldermead": {"max_iter_NM": 200},
}

GRAPH_COMPETITOR_DISTRIBUTIONS: dict = {
    "johnson":  stats.johnsonsu,   
    "gamma":    stats.gamma,       
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sample_source_distribution(dist_name: str, 
                                rng: np.random.Generator,
                                n_samples: int) -> np.ndarray:
    """
    Draw *n_samples* observations from a randomly-parameterised instance of
    *dist_name*.  All distributions are parameterised so that samples are
    positive (important for MLE with bounded-support competitors).

    Returns a sorted float64 array.
    """
    rs = int(rng.integers(0, 2**31))   # seed for scipy RVS

    if dist_name == "uniform":
        loc   = rng.uniform(1, 10)
        scale = rng.uniform(1, 20)
        data  = stats.uniform.rvs(loc=loc, scale=scale,
                                  size=n_samples, random_state=rs)

    elif dist_name == "exponential":
        loc   = rng.uniform(0.5, 5)
        scale = rng.uniform(0.5, 10)
        data  = stats.expon.rvs(loc=loc, scale=scale,
                                size=n_samples, random_state=rs)

    elif dist_name == "gamma":
        a     = rng.uniform(0.5, 8)
        loc   = rng.uniform(0.1, 3)
        scale = rng.uniform(0.5, 5)
        data  = stats.gamma.rvs(a, loc=loc, scale=scale,
                                size=n_samples, random_state=rs)

    elif dist_name == "weibull":
        c     = rng.uniform(0.5, 5)
        loc   = rng.uniform(0.1, 3)
        scale = rng.uniform(0.5, 5)
        data  = stats.weibull_min.rvs(c, loc=loc, scale=scale,
                                      size=n_samples, random_state=rs)

    elif dist_name == "normal":
        loc   = rng.uniform(5, 20) 
        scale = rng.uniform(0.5, 4)
        data  = stats.norm.rvs(loc=loc, scale=scale,
                               size=n_samples, random_state=rs)

    elif dist_name == "lognormal":
        s     = rng.uniform(0.1, 1.2)
        loc   = rng.uniform(0.0, 2)
        scale = rng.uniform(0.5, 5)
        data  = stats.lognorm.rvs(s, loc=loc, scale=scale,
                                  size=n_samples, random_state=rs)

    elif dist_name == "beta":
        a     = rng.uniform(0.5, 5)
        b     = rng.uniform(0.5, 5)
        loc   = rng.uniform(1, 5)
        scale = rng.uniform(1, 20)
        data  = stats.beta.rvs(a, b, loc=loc, scale=scale,
                               size=n_samples, random_state=rs)

    elif dist_name == "log_logistic":
        c     = rng.uniform(1.5, 8)
        loc   = rng.uniform(0.1, 3)
        scale = rng.uniform(0.5, 5)
        data  = stats.fisk.rvs(c, loc=loc, scale=scale,
                               size=n_samples, random_state=rs)

    elif dist_name == "triangular":
        c     = rng.uniform(0.1, 0.9)
        loc   = rng.uniform(1, 5)
        scale = rng.uniform(2, 20)
        data  = stats.triang.rvs(c, loc=loc, scale=scale,
                                 size=n_samples, random_state=rs)

    elif dist_name == "bimodal_gaussian":
        # Two-component Gaussian mixture; both peaks are strictly positive
        mu1   = rng.uniform(3, 10)
        mu2   = rng.uniform(mu1 + 3, mu1 + 15)
        sig1  = rng.uniform(0.5, 3)
        sig2  = rng.uniform(0.5, 3)
        w     = rng.uniform(0.3, 0.7)          # weight of component 1
        n1    = int(np.round(w * n_samples))
        n2    = n_samples - n1
        part1 = rng.standard_normal(n1) * sig1 + mu1
        part2 = rng.standard_normal(n2) * sig2 + mu2
        data  = np.concatenate([part1, part2])

    elif dist_name == "trimodal_gaussian":
        # Three-component Gaussian mixture; all peaks are strictly positive
        mu1   = rng.uniform(2, 6)
        mu2   = rng.uniform(mu1 + 3, mu1 + 10)
        mu3   = rng.uniform(mu2 + 3, mu2 + 10)
        sig1  = rng.uniform(0.3, 2)
        sig2  = rng.uniform(0.3, 2)
        sig3  = rng.uniform(0.3, 2)
        w     = rng.dirichlet([1.0, 1.0, 1.0])  # random weights summing to 1
        n1    = int(np.round(w[0] * n_samples))
        n2    = int(np.round(w[1] * n_samples))
        n3    = n_samples - n1 - n2
        part1 = rng.standard_normal(n1) * sig1 + mu1
        part2 = rng.standard_normal(n2) * sig2 + mu2
        part3 = rng.standard_normal(n3) * sig3 + mu3
        data  = np.concatenate([part1, part2, part3])

    else:
        raise ValueError(f"Unknown distribution: {dist_name}")

    return np.sort(data.astype(np.float64))


def _fit_scipy_mle(scipy_dist, data: np.ndarray) -> tuple[float, float]:
    """
    Fit *scipy_dist* to *data* by MLE and return (elapsed_seconds, nll).

    Implementation notes
    --------------------
    - Uses scipy_dist.fit(data) to obtain fitted parameters (shape..., loc, scale).
    - Computes negative log-likelihood from those fitted parameters via scipy_dist.logpdf.
    - Returns (nan, nan) on failure.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = _time.perf_counter()
            # Use the distribution's own fit method (works across SciPy versions)
            params = scipy_dist.fit(data)
            elapsed = _time.perf_counter() - t0

        # params is a tuple of (shape1, shape2, ..., loc, scale) or (loc, scale)
        # Compute the negative log-likelihood from the fitted parameters.
        # Use logpdf which exists on rv_continuous; sum of logpdf across data is the log-likelihood.
        logpdf_vals = scipy_dist.logpdf(data, *params)
        # If logpdf produced non-finite values (e.g. zeros), treat as failure
        if not np.all(np.isfinite(logpdf_vals)):
            return elapsed, np.nan

        nll = -float(np.sum(logpdf_vals))
        return elapsed, nll if np.isfinite(nll) else (elapsed, np.nan)
    except Exception:
        # Keep contract: on any failure return NaNs for (time, nll)
        return np.nan, np.nan


def _fit_bezierv_mle(data: np.ndarray,
                     n: int = 7,
                     **kwargs) -> tuple[float, float]:
    """
    Fit bezierv MLE to *data* and return (elapsed_seconds, nll).
    Returns (nan, nan) on failure.

    The metric returned by primal_grad.fit is already the negative
    log-likelihood: ``-sum(log(A @ w))``.

    Parameters
    ----------
    data : np.ndarray
        Sorted sample data.
    n : int, default 7
        Number of bezierv control points.
    **kwargs
        Extra keyword arguments forwarded to :meth:`DistFit.fit`
        (e.g. ``max_iter``, ``tol_res_root``, ``tol_lambda_root``,
        ``max_iters_root``).  Defaults from ``MLE_FIT_KWARGS`` are used
        by the benchmark callers when no kwargs are supplied.
    """
    try:
        try:
            df = DistFit(data, n=n)
        except ValueError:
            df = DistFit(data, n=n, method_init_x='uniform')
        t0 = _time.perf_counter()
        _, nll = df.fit(method="mle", **kwargs)
        elapsed = _time.perf_counter() - t0
        return elapsed, float(nll)
    except Exception:
        return np.nan, np.nan

def _fit_bezierv_mse(data: np.ndarray,
                     method: str,
                     n: int = 7,
                     **kwargs) -> tuple[float, float]:
    """
    Fit bezierv using an MSE-based method and return (elapsed_seconds, mse).
    Raises on failure (caller is responsible for catching).

    Parameters
    ----------
    data : np.ndarray
        Sorted sample data.
    method : str
        One of ``'nonlinear'``, ``'projgrad'``, ``'neldermead'``.
    n : int, default 7
        Number of bezierv control points.
    **kwargs
        Extra keyword arguments forwarded to :meth:`DistFit.fit`
        (e.g. ``step_size_PG``, ``max_iter_NM``, ``solver_NL``).
    """
    try:
        df = DistFit(data, n=n)
    except ValueError:
        df = DistFit(data, n=n, method_init_x='uniform')
    t0 = _time.perf_counter()
    _, mse = df.fit(method='mse', algorithm=method, **kwargs)
    elapsed = _time.perf_counter() - t0
    return elapsed, float(mse)


SOURCE_DISTRIBUTIONS = [
    "uniform",
    "exponential",
    "gamma",
    "weibull",
    "normal",
    "lognormal",
    "beta",
    "log_logistic",
    "triangular",
    "bimodal_gaussian",
    "trimodal_gaussian",
]

COMPETITOR_DISTRIBUTIONS = {
    "generalized_beta": stats.beta,       # 4 params: a, b, loc, scale
    "johnson":          stats.johnsonsu,  # 4 params: a, b, loc, scale (Johnson SU)
}


def benchmark_distribution_fits(
    k: int,
    n_samples: int = 500,
    n_bezierv: int = 7,    seed: int | None = None,
    output_csv: str = "benchmark_results.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Benchmark bezierv MLE, Generalised Beta, Johnson SU, and Burr XII (4-param)
    against *k* randomly-generated distribution instances.

    For each instance a source distribution is chosen at random from
    ``SOURCE_DISTRIBUTIONS``, parameters are sampled uniformly within
    plausible ranges, and *n_samples* observations are drawn.  All four
    fitting methods are then applied and their elapsed time and negative
    log-likelihood are recorded.

    Parameters
    ----------
    k : int
        Number of random instances (each instance = one source distribution
        randomly parameterised).
    n_samples : int, default 500
        Number of samples drawn from each instance.
    n_bezierv : int, default 7
        Number of bezierv control points (*n* argument of :class:`DistFit`).
    seed : int | None, default None
        Seed for the NumPy random generator (for reproducibility).
    output_csv : str, default "benchmark_results.csv"
        File path for the output CSV.
    verbose : bool, default True
        Whether to show a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Dist``, ``Method``, ``Time``, ``NegLogLik``.
        One row per (instance, method) combination, so 4 * k rows in total
        (minus any rows where fitting failed, which still appear with NaN).
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    iterator = range(k)
    if verbose:
        iterator = tqdm(iterator, desc="Benchmarking", unit="instance")

    for _ in iterator:
        # ------------------------------------------------------------------
        # 1.  Draw a random source distribution and generate data
        # ------------------------------------------------------------------
        dist_name = SOURCE_DISTRIBUTIONS[
            int(rng.integers(0, len(SOURCE_DISTRIBUTIONS)))
        ]
        data = _sample_source_distribution(dist_name, rng, n_samples)

        # ------------------------------------------------------------------
        # 2.  Fit bezierv MLE
        # ------------------------------------------------------------------
        elapsed, nll = _fit_bezierv_mle(data, n=n_bezierv, **MLE_FIT_KWARGS)
        #print(f"Instance {dist_name}: bezierv MLE took {elapsed:.3f}s, NLL={nll:.2f}")
        records.append(
            {"Dist": dist_name, "Method": "bezierv_mle",
             "Time": elapsed, "NegLogLik": nll}
        )

        # ------------------------------------------------------------------
        # 3.  Fit competitor distributions (scipy MLE)
        # ------------------------------------------------------------------
        for method_name, scipy_dist in COMPETITOR_DISTRIBUTIONS.items():
            elapsed, nll = _fit_scipy_mle(scipy_dist, data)
            records.append(
                {"Dist": dist_name, "Method": method_name,
                 "Time": elapsed, "NegLogLik": nll}
            )

    df = pd.DataFrame(records, columns=["Dist", "Method", "Time", "NegLogLik"])
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"\nResults saved to '{output_csv}'  ({len(df)} rows, {k} instances).")

    return df

def benchmark_bezierv_mse_methods(
    k: int,
    n_samples: int = 1000,
    n_bezierv: int = 7,
    seed: int | None = None,
    output_csv: str = "data/benchmark_mse_methods.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Benchmark the three bezierv MSE fitting methods (nonlinear, projgrad,
    neldermead) against *k* randomly-generated distribution instances.

    For each instance a source distribution is chosen at random from
    ``SOURCE_DISTRIBUTIONS``, parameters are sampled uniformly within
    plausible ranges, and *n_samples* observations are drawn.  All three MSE
    fitting methods are applied to the *same* dataset and their elapsed time
    and mean-squared error (CDF distance) are recorded.

    Parameters
    ----------
    k : int
        Number of random instances.
    n_samples : int, default 500
        Number of samples drawn from each instance.
    n_bezierv : int, default 7
        Number of bezierv control points (*n* argument of :class:`DistFit`).
    mse_method_kwargs : dict[str, dict] | None, default None
        Per-method keyword arguments forwarded to :func:`_fit_bezierv_mse`.
        When ``None``, ``MSE_METHOD_KWARGS`` is used.
    seed : int | None, default None
        Seed for the NumPy random generator (for reproducibility).
    output_csv : str, default "data/benchmark_mse_methods.csv"
        File path for the output CSV.
    verbose : bool, default True
        Whether to show a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Dist``, ``Method``, ``Time``, ``MSE``.
        One row per (instance, method) combination — ``3 * k`` rows in total
        (rows with fitting failures appear with NaN values).
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    iterator = range(k)
    if verbose:
        iterator = tqdm(iterator, desc="MSE benchmark", unit="instance")

    for _ in iterator:
        dist_name = SOURCE_DISTRIBUTIONS[
            int(rng.integers(0, len(SOURCE_DISTRIBUTIONS)))
        ]
        data = _sample_source_distribution(dist_name, rng, n_samples)

        for method in MSE_METHODS:
            failed = False
            try:
                elapsed, mse = _fit_bezierv_mse(
                    data, method=method, n=n_bezierv, **MSE_METHOD_KWARGS[method]
                )
            except Exception:
                failed = True
                elapsed, mse = np.nan, np.nan
            records.append(
                {"Dist": dist_name, "Method": method,
                 "Time": elapsed, "MSE": mse, "Failed": failed}
            )

    df = pd.DataFrame(records, columns=["Dist", "Method", "Time", "MSE", "Failed"])
    df.to_csv(output_csv, index=False)

    if verbose:
        print(f"\nResults saved to '{output_csv}'  ({len(df)} rows, {k} instances).")
        failure_summary = (
            df[df["Failed"]]
            .groupby(["Dist", "Method"])
            .size()
            .reset_index(name="FailCount")
        )
        if not failure_summary.empty:
            print("\nFitting failures (dist, method) -> count:")
            for _, row in failure_summary.iterrows():
                print(f"  {row['Dist']:22s}  {row['Method']:12s}  {row['FailCount']}")
        else:
            print("No fitting failures.")

    return df


def benchmark_graph_fits_mse(
    data: dict,
    k: int,
    n_bezierv: int = 7,
    seed: int | None = None,
    output_csv: str = "data/benchmark_graph_fits_mse.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Benchmark the three Bezierv MSE fitting methods (nonlinear, projgrad,
    neldermead) against the travel-time samples in the preprocessed graph
    *data* dict (output of ``preprocess_data``).

    For each arc, up to *k* observations are drawn (all samples are used when
    fewer than *k* are available).  All three MSE methods are applied to the
    same dataset and their elapsed time and MSE (CDF distance) are recorded.

    Parameters
    ----------
    data : dict
        Preprocessed arc data as returned by ``preprocess_data``.  Each value
        must contain a ``'travel_time'`` list of float observations.
    k : int
        Maximum number of samples to draw from each arc's travel-time list.
    n_bezierv : int, default 7
        Number of Bezierv control points.
    seed : int | None, default None
        Seed for the NumPy random generator (for reproducibility).
    output_csv : str, default "data/benchmark_graph_fits_mse.csv"
        File path for the output CSV.
    verbose : bool, default True
        Whether to show a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns
        ``Arc``, ``n_samples``, ``Method``, ``Time``, ``MSE``, ``Failed``.
        Three rows per arc (one per MSE method).
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    items = list(data.items())
    iterator = tqdm(items, desc="MSE benchmark – graph arcs", unit="arc") if verbose else items

    for key, value in iterator:
        travel_times = value.get("travel_time", [])
        if not travel_times:
            continue

        travel_arr = np.asarray(travel_times, dtype=np.float64)

        n_available = len(travel_arr)
        n_draw = min(k, n_available)
        idx = rng.choice(n_available, size=n_draw, replace=False)
        sample = np.sort(travel_arr[idx])

        arc_label = f"{value.get('start_node', key[0])}->{value.get('end_node', key[1])}"

        for method in MSE_METHODS:
            failed = False
            try:
                elapsed, mse = _fit_bezierv_mse(
                    sample, method=method, n=n_bezierv, **MSE_METHOD_KWARGS[method]
                )
            except Exception:
                failed = True
                elapsed, mse = np.nan, np.nan
            records.append({
                "Arc":      arc_label,
                "n_samples": n_draw,
                "Method":   method,
                "Time":     elapsed,
                "MSE":      mse,
                "Failed":   failed,
            })

    df = pd.DataFrame(records, columns=["Arc", "n_samples", "Method", "Time", "MSE", "Failed"])
    df.to_csv(output_csv, index=False)

    if verbose:
        n_arcs = len(df["Arc"].unique())
        print(f"\nResults saved to '{output_csv}'  ({len(df)} rows, {n_arcs} arcs).")
        failure_summary = (
            df[df["Failed"]]
            .groupby("Method")
            .size()
        )
        if not failure_summary.empty:
            print("\nFitting failures per method:")
            for method, cnt in failure_summary.items():
                print(f"  {method:12s}  {cnt}")
        else:
            print("No fitting failures.")

    return df

def benchmark_graph_fits(
    data: dict,
    k: int,
    n_bezierv: int = 7,
    seed: int | None = None,
    output_csv: str = "data/benchmark_graph_fits.csv",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Benchmark Bezierv MLE, Johnson SU, and Gamma against the travel-time
    samples found in the preprocessed graph *data* dict (output of
    ``preprocess_data``).

    For each arc in *data*, up to *k* travel-time observations are drawn
    (without replacement when the arc has more than *k* samples, all samples
    are used when fewer are available).  All three methods are fitted to the
    same dataset and their elapsed time and negative log-likelihood (NLL) are
    recorded.

    Parameters
    ----------
    data : dict
        Preprocessed arc data as returned by ``preprocess_data``.  Each value
        must contain a ``'travel_time'`` list of float observations.
    k : int
        Maximum number of samples to draw from each arc's travel-time list.
    n_bezierv : int, default 7
        Number of Bezierv control points.
    seed : int | None, default None
        Seed for the NumPy random generator (for reproducibility).
    output_csv : str, default "data/benchmark_graph_fits.csv"
        File path for the output CSV.
    verbose : bool, default True
        Whether to show a tqdm progress bar.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns
        ``Arc``, ``n_samples``, ``Method``, ``Time``, ``NegLogLik``.
        Three rows per arc (one per method), with NaN for failed fits.
    """
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    items = list(data.items())
    iterator = tqdm(items, desc="Benchmarking graph arcs", unit="arc") if verbose else items

    for key, value in iterator:
        travel_times = value.get("travel_time", [])
        if not travel_times:
            continue

        travel_arr = np.asarray(travel_times, dtype=np.float64)

        # Sample up to k observations (without replacement)
        n_available = len(travel_arr)
        n_draw = min(k, n_available)
        idx = rng.choice(n_available, size=n_draw, replace=False)
        sample = np.sort(travel_arr[idx])

        arc_label = f"{value.get('start_node', key[0])}->{value.get('end_node', key[1])}"

        # --- Bezierv MLE --------------------------------------------------
        elapsed, nll = _fit_bezierv_mle(sample, n=n_bezierv, **MLE_FIT_KWARGS_REAL)
        records.append({
            "Arc": arc_label,
            "n_samples": n_draw,
            "Method": "bezierv_mle",
            "Time": elapsed,
            "NegLogLik": nll,
        })

        # --- Competitor distributions (scipy MLE) -------------------------
        for method_name, scipy_dist in GRAPH_COMPETITOR_DISTRIBUTIONS.items():
            elapsed, nll = _fit_scipy_mle(scipy_dist, sample)
            records.append({
                "Arc": arc_label,
                "n_samples": n_draw,
                "Method": method_name,
                "Time": elapsed,
                "NegLogLik": nll,
            })

    df = pd.DataFrame(records, columns=["Arc", "n_samples", "Method", "Time", "NegLogLik"])
    df.to_csv(output_csv, index=False)

    if verbose:
        n_arcs = len(df["Arc"].unique())
        print(f"\nResults saved to '{output_csv}'  ({len(df)} rows, {n_arcs} arcs).")
        failed = df[df["NegLogLik"].isna()]
        if not failed.empty:
            print("\nFitting failures (arc, method) -> count:")
            summary = failed.groupby("Method").size()
            for method, cnt in summary.items():
                print(f"  {method:20s}  {cnt}")
        else:
            print("No fitting failures.")

    return df