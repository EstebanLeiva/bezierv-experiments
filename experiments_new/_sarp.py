"""Pulse-algorithm callbacks for the stochastic shortest-path experiment (paper §8).

`length_miles` is the minimised deterministic cost; `prune_feasibility` enforces
the chance constraint P[t(P) ≤ T_max] ≥ alpha via Monte-Carlo convolution of the
arc-wise Bezier RVs.
"""
from __future__ import annotations

import time as _time

import numpy as np

from pulsekit.dijkstra import dijkstra_between_nodes
from pulsekit.graph import Graph
from pulsekit.pulse import Parameters, Pulse


# ── Pulse callbacks ──────────────────────────────────────────────────────────
def info_update(graph, current_node, reachable_node, path, deterministic_info, random_info):
    link = graph.nodes[current_node].links[reachable_node]
    deterministic_info['length_miles'] += link.deterministic['length_miles']
    return deterministic_info, random_info


def _montecarlo_prob(list_bezierv, value: float, n_sims: int, rng: np.random.Generator) -> float:
    sims = np.zeros(n_sims)
    for bz in list_bezierv:
        sims += bz.random(n_sims, rng=rng)
    return float(np.sum(np.sort(sims) <= value) / n_sims)


def prune_feasibility(pulse_alg, current_node, current_path_info) -> bool:
    if len(current_path_info.path) < 1:
        return False
    path = current_path_info.path.copy()
    path.append(current_node)
    arc_bezierv = [
        pulse_alg.parameters.graph.nodes[path[i]].links[path[i + 1]].random['time']['bezierv']
        for i in range(len(path) - 1)
    ]
    T_max  = pulse_alg.parameters.constants['T_max']
    alpha  = pulse_alg.parameters.constants['alpha']
    n_sims = pulse_alg.parameters.constants['n_sims']
    rng    = pulse_alg.parameters.constants['rng']
    remaining = T_max - pulse_alg.preprocessing.deterministic['min_travel_time'][current_node]
    return _montecarlo_prob(arc_bezierv, remaining, n_sims, rng) <= alpha


def prune_bounds(pulse_alg, current_node, current_path_info) -> bool:
    return (current_path_info.deterministic['length_miles']
            + pulse_alg.preprocessing.deterministic['length_miles'][current_node]
            > pulse_alg.current_optimal_objective)


def exploration_order(pulse_alg, node):
    return pulse_alg.preprocessing.deterministic['length_miles'][node]


# Pulsekit's heap pushes `(score, PathInformation)`; PathInformation has no
# __lt__, so ties on `score` raise TypeError. Append a monotone counter as a
# deterministic tie-breaker.
_score_counter = 0


def pulse_score(pulse_alg, current_path_info):
    global _score_counter
    _score_counter += 1
    return (pulse_alg.preprocessing.deterministic['length_miles'][current_path_info.path[-1]],
            _score_counter)


# ── Build a Bezier-weighted graph from preprocessed arc data ─────────────────
def build_bezier_graph(arcs: dict, *, fit_func, verbose: bool = True) -> Graph:
    """Construct a `pulsekit.Graph` with a fitted Bezier RV on each arc.

    `fit_func(samples) -> bezierv.classes.bezierv.Bezierv` performs the fit.
    """
    from tqdm import tqdm

    graph = Graph()
    items = sorted(arcs.items(), key=lambda kv: str(kv[0]))
    iterator = tqdm(items, desc='Fitting arc Beziers', unit='arc') if verbose else items

    n_fail = 0
    for key, value in iterator:
        samples = np.sort(np.asarray(value['travel_time'], dtype=np.float64))
        try:
            bz = fit_func(samples)
        except Exception:
            n_fail += 1
            continue

        deterministic = {
            'start_longitude': value['start_longitude'],
            'start_latitude':  value['start_latitude'],
            'end_longitude':   value['end_longitude'],
            'end_latitude':    value['end_latitude'],
            'length_miles':    value['length_miles'],
            'min_travel_time': value['min_travel_time'],
        }
        random_part = {'time': {'bezierv': bz, 'mean_bezierv': bz.get_mean()}}
        graph.add_link(str(value['start_node']), str(value['end_node']),
                       deterministic, random_part)

    if verbose:
        print(f'[build_bezier_graph] {n_fail} arcs failed to fit')
    return graph


# ── Time budget calibration (paper §8) ──────────────────────────────────────
def calibrate_time_budget(graph: Graph,
                          start_node: int,
                          target_node: int,
                          *,
                          alpha: float,
                          gamma: float,
                          n_sims: int,
                          n_bezier: int,
                          rng: np.random.Generator):
    """T = T_t_alpha + (T_c_alpha - T_t_alpha) * (1 - gamma) + 1e-2

    where T_t_alpha is the alpha-quantile of the min-mean-time path and
    T_c_alpha is the alpha-quantile of the min-cost path.
    """
    from bezierv.classes.convolver import Convolver

    shortest_mean_path, _ = dijkstra_between_nodes(
        graph, start_node, target_node, cost_key='mean_bezierv', rand_var='time'
    )
    shortest_cost_path, cost_min_cost = dijkstra_between_nodes(
        graph, start_node, target_node, cost_key='length_miles'
    )

    cost_min_mean = sum(
        graph.nodes[shortest_mean_path[i]].links[shortest_mean_path[i + 1]].deterministic['length_miles']
        for i in range(len(shortest_mean_path) - 1)
    )

    def _path_quantile(path: list[int], q: float) -> float:
        bz_list = [graph.nodes[path[i]].links[path[i + 1]].random['time']['bezierv']
                   for i in range(len(path) - 1)]
        bz_path, _ = Convolver(bz_list).convolve(n_sims=n_sims, rng=rng, n=n_bezier,
                                                 method='mse', algorithm='projgrad')
        return float(bz_path.quantile(q))

    T_t_alpha = _path_quantile(shortest_mean_path, alpha)
    T_c_alpha = _path_quantile(shortest_cost_path, alpha)
    T = T_t_alpha + (T_c_alpha - T_t_alpha) * (1.0 - gamma) + 1e-2

    return {
        'T_max': T,
        'shortest_mean_path': shortest_mean_path,
        'cost_min_mean': cost_min_mean,
        'shortest_cost_path': shortest_cost_path,
        'cost_min_cost': cost_min_cost,
        'T_t_alpha': T_t_alpha,
        'T_c_alpha': T_c_alpha,
    }


# ── Run the pulse algorithm ──────────────────────────────────────────────────
def run_sarp(graph: Graph,
             start_node: int,
             target_node: int,
             *,
             alpha: float,
             gamma: float,
             n_sims: int,
             n_bezier: int,
             rng: np.random.Generator):
    cal = calibrate_time_budget(graph, start_node, target_node,
                                alpha=alpha, gamma=gamma,
                                n_sims=n_sims, n_bezier=n_bezier, rng=rng)
    params = Parameters(
        graph=graph,
        source_node=start_node,
        target_node=target_node,
        constants={'T_max': cal['T_max'], 'alpha': alpha, 'n_sims': n_sims, 'rng': rng},
        max_pulse_depth=len(cal['shortest_mean_path']) + 1,
        deterministic_weights=['length_miles', 'min_travel_time'],
        random_weights={'time': ['bezierv']},
        prep_deterministic_weights=['length_miles', 'min_travel_time'],
        prep_random_weights={},
        info_update=info_update,
        pulse_score=pulse_score,
        exploration_order=exploration_order,
        pruning_functions=[prune_feasibility, prune_bounds],
        initial_path=cal['shortest_mean_path'],
        initial_objective=cal['cost_min_mean'],
    )
    pulse = Pulse(params)
    pulse.preprocess()
    t0 = _time.perf_counter()
    pulse.run(init_optimal_path=cal['shortest_mean_path'], init_objective=cal['cost_min_mean'])
    elapsed = _time.perf_counter() - t0
    return pulse, elapsed, cal


# ── Convolution of arc Beziers along a path ──────────────────────────────────
def path_distribution(graph: Graph,
                      path: list[int],
                      *,
                      n_sims: int,
                      n_bezier: int,
                      rng: np.random.Generator):
    from bezierv.classes.convolver import Convolver
    bz_list = [graph.nodes[path[i]].links[path[i + 1]].random['time']['bezierv']
               for i in range(len(path) - 1)]
    bz_path, _ = Convolver(bz_list).convolve(n_sims=n_sims, rng=rng, n=n_bezier,
                                             method='mse', algorithm='projgrad')
    return bz_path


# ── Chicago network + optimal-path map (paper Figure 6a) ─────────────────────
def plot_path_on_grid(graph: Graph,
                      path: list[int],
                      *,
                      edge_color: str = '#BBBBBB',
                      path_color: str = '#D55E00',
                      edge_linewidth: float = 0.5,
                      path_linewidth: float = 2.0,
                      node_size: float = 14.0,
                      figsize_mm: tuple = (120, 120)):
    """Plot the full Chicago network with the optimal path overlaid.

    Coordinates are read from each link's deterministic dict (start/end lon/lat,
    populated in `build_bezier_graph`). Returns ``(fig, ax)``; the caller is
    responsible for saving and for applying ``_plot_style.apply()`` beforehand.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    positions: dict = {}
    edges: list = []
    for u, node in graph.nodes.items():
        for v, link in node.links.items():
            d = link.deterministic
            positions[u] = (d['start_longitude'], d['start_latitude'])
            positions[v] = (d['end_longitude'],   d['end_latitude'])
            edges.append((u, v))

    mm = 1 / 25.4
    fig, ax = plt.subplots(figsize=(figsize_mm[0] * mm, figsize_mm[1] * mm),
                           layout='constrained')

    ax.add_collection(LineCollection(
        [[positions[u], positions[v]] for u, v in edges],
        colors=edge_color, linewidths=edge_linewidth, alpha=0.7, zorder=1,
    ))

    path_edges = list(zip(path[:-1], path[1:]))
    ax.add_collection(LineCollection(
        [[positions[u], positions[v]] for u, v in path_edges],
        colors=path_color, linewidths=path_linewidth, zorder=3,
    ))

    xs, ys = zip(*(positions[i] for i in path))
    ax.scatter(xs, ys, s=node_size, color=path_color, edgecolor='white',
               linewidth=0.4, zorder=4)

    ax.autoscale_view()
    ax.set_xlabel(r'Longitude')
    ax.set_ylabel(r'Latitude')
    ax.set_aspect('equal', adjustable='datalim')
    ax.spines[['top', 'right']].set_visible(False)
    return fig, ax
