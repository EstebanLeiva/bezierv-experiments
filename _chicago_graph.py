"""Load and preprocess the Chicago road-network travel-time dataset.

The paper splits the dataset into three time regimes:

    rush_hour : 06:00 - 10:00
    daytime   : 06:00 - 18:00
    full_day  : 00:00 - 23:59

Weekend observations are dropped.
For arcs whose observations are all identical, lognormal samples are synthesised
using the across-arc mean variance.
"""
from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ── Time regimes (paper §7.2) ────────────────────────────────────────────────
REGIMES: dict[str, tuple[time, time]] = {
    'rush_hour': (time(6, 0,  0), time(10, 0, 0)),
    'daytime':   (time(6, 0,  0), time(18, 0, 0)),
    'full_day':  (time(0, 0,  0), time(23, 59, 59)),
}

# Bezier degree used per regime in paper Table 4.
REGIME_BEZIER_DEGREE = {
    'rush_hour': 10,
    'daytime':   15,
    'full_day':  20,
}

REGIME_DISPLAY = {
    'rush_hour': 'Rush hour',
    'daytime':   'Daytime',
    'full_day':  'Full-day',
}

DATA_PATH = Path(__file__).parent / 'data' / 'Chicago_main.json'
_STATIC_KEYS = {
    'start_node', 'end_node',
    'start_longitude', 'start_latitude',
    'end_longitude',   'end_latitude',
    'length_miles',
}


def load_raw(path: Path = DATA_PATH) -> dict:
    """Return the raw {edge_id: {...}} dict from disk."""
    with open(path, 'r') as f:
        return json.load(f)


def preprocess(raw: dict,
               start_t: time,
               end_t: time,
               *,
               seed: int = 42,
               progress: bool = False) -> dict:
    """Group observations within the regime window (weekdays only).

      1. Bucket timestamps by (start_node, end_node) inside the [start_t, end_t] window
         on weekdays only.
      2. For arcs whose observations are all identical, synthesise their travel_time
         lognormally using the dataset-wide mean variance.
      3. Set 'min_travel_time' on each arc.
    """
    rng = np.random.default_rng(seed)
    organized: dict = {}

    iterator = raw.items()
    if progress:
        iterator = tqdm(iterator, desc='Preprocessing observations', unit='record')

    for _, value in iterator:
        edge = (value['start_node'], value['end_node'])
        bucket = organized.setdefault(edge, {})
        for key, val in value.items():
            if key in _STATIC_KEYS:
                bucket[key] = val
            else:
                dt = datetime.strptime(key, '%Y-%m-%d %H:%M:%S')
                if start_t <= dt.time() <= end_t and dt.weekday() < 5:
                    bucket.setdefault('travel_time', []).append(val)

    # Global mean variance across arcs with > 1 unique observation
    variances = [np.var(v['travel_time'])
                 for v in organized.values()
                 if 'travel_time' in v and len(set(v['travel_time'])) > 1]
    global_var = float(np.mean(variances)) if variances else 0.0

    for v in organized.values():
        tt = v.get('travel_time', [])
        if not tt:
            v['artificial'] = False
            continue
        if len(set(tt)) == 1:
            x0 = tt[0]
            sigma2 = float(np.log(global_var / x0 ** 2 + 1)) if global_var > 0 else 1e-6
            sigma = float(np.sqrt(sigma2))
            mu = float(np.log(x0)) - 0.5 * sigma2
            v['travel_time'] = rng.lognormal(mean=mu, sigma=sigma, size=len(tt)).tolist()
            v['artificial'] = True
        else:
            v['artificial'] = False
        v['min_travel_time'] = float(min(v['travel_time']))

    return organized


def preprocess_regime(raw: dict, regime: str, *, seed: int = 42, progress: bool = False) -> dict:
    if regime not in REGIMES:
        raise KeyError(f'Unknown regime {regime!r}; choose from {list(REGIMES)}')
    start_t, end_t = REGIMES[regime]
    return preprocess(raw, start_t, end_t, seed=seed, progress=progress)


def filter_multi_obs(processed: dict, min_obs: int = 2) -> dict:
    """Drop arcs with fewer than *min_obs* travel-time observations.

    The paper reports the dataset reduces to 1,086 arcs after excluding arcs with
    a single observation. This filter applies the same criterion.
    """
    return {k: v for k, v in processed.items()
            if len(v.get('travel_time', [])) >= min_obs}


def sample_arcs(processed: dict, n: int, *, seed: int = 42) -> dict:
    """Deterministically subsample *n* arcs (for smoke tests)."""
    keys = sorted(processed.keys())
    if n >= len(keys):
        return dict(processed)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(keys), size=n, replace=False)
    return {keys[i]: processed[keys[i]] for i in idx}
