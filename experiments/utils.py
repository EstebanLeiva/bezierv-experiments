import json
import random
import pickle
import numpy as np
from datetime import datetime, time
import time
from scipy import stats

from distfit import distfit

import sarp_bezierv

from tqdm import tqdm
from typing import Tuple
from scipy.stats import norm, lognorm

from bezierv.classes.bezierv import Bezierv
from bezierv.classes.distfit import DistFit
from bezierv.classes.convolver import Convolver

from pulsekit.graph import Graph
from pulsekit.dijkstra import dijkstra, dijkstra_between_nodes
from pulsekit.pulse import Parameters



def load_json(file_path):
    """
    Load a JSON file and return its content.
    
    Args:
        file_path (str): The path to the JSON file.
        
    Returns:
        dict: The content of the JSON file as a dictionary.
    """
    with open(file_path, 'r') as file:
        return json.load(file)
    
def get_sample(values, sample_size):
  if not values:
    return []
  population_size = len(values)
  if sample_size > population_size:
    raise ValueError("Sample size cannot be larger than the population size for sampling without replacement.")
  sample = random.sample(values, k=sample_size)
  return sample

def preprocess_data(data, start_time, end_time):
    """
    Preprocess the data by converting it to lowercase and stripping whitespace.
    
    Args:
        data (dict): The data to preprocess.
        start_time (time): The start time for filtering.
        end_time (time): The end time for filtering.
        
    Returns:
        dict: The preprocessed data.
    """
    data_organized = {}
    subkeys = ['start_node', 'end_node', 'start_longitude', 'start_latitude', 'end_longitude', 'end_latitude', 'length_miles']
    for key, value in data.items():
        start = value['start_node']
        end = value['end_node']
        key = (start, end)
        if key not in data_organized:
            data_organized[key] = {}
        for sub_key, sub_value in value.items():
            if sub_key not in subkeys:
                dt = datetime.strptime(sub_key, "%Y-%m-%d %H:%M:%S")
                if start_time <= dt.time() <= end_time and dt.weekday() < 5:
                    if 'travel_time' not in data_organized[key]:
                        data_organized[key]['travel_time'] = []
                    else:
                        data_organized[key]['travel_time'].append(sub_value)
            else:
                data_organized[key][sub_key] = sub_value
    
    variances = []
    for key, value in data_organized.items():
        uniques = len(set(value['travel_time']))
        if uniques > 1:
            variances.append(np.var(value['travel_time']))
    var = np.mean(variances)

    for key, value in tqdm(data_organized.items(), desc="Generating travel times"):
        uniques = len(set(value['travel_time']))
        if uniques == 1:
            samples = len(value['travel_time'])
            sigma2 = np.log(var / value['travel_time'][0]**2 + 1)
            sigma = np.sqrt(sigma2)
            mu = np.log(value['travel_time']) - 0.5 * sigma2  
            value['travel_time'] = np.random.lognormal(mean=mu, sigma=sigma, size=samples).tolist()
            value['artificial'] = True
        else:
            value['artificial'] = False
    for key, value in data_organized.items():
        value['min_travel_time'] = min(value['travel_time'])
    return data_organized

def fit_graph(data, 
              n, 
              k,
              method='projgrad',
              step_size_PG: float=0.001,
              max_iter_PG: float=1000,
              threshold_PG: float=1e-3,
              step_size_PS: float=0.001,
              max_iter_PS: int=1000,
              solver_NL: str='ipopt',
              max_iter_NM: int=1000
              ):
    """
    Build the graph for the data and fit a bezier distribution to the travel times.
    
    Args:
        data (dict): The data to fit the graph to.
        n (int): The number of control points for the bezier fit for all arcs.
        k (int): The number of datapoints to sample for each arc.
        
    Returns:
        graph: The graph with fitted distributions.
        errors: The list of mean squared errors for the fitted bezier curves.
    """
    graph = Graph()
    errors = []
    times = []
    for key, value in tqdm(data.items(), desc="Fitting graph links"):
        start_node_name = str(value['start_node'])
        end_node_name = str(value['end_node'])
        deterministic = {
        'start_longitude': value['start_longitude'],
        'start_latitude': value['start_latitude'],
        'end_longitude': value['end_longitude'],
        'end_latitude': value['end_latitude'],
        'length_miles': value['length_miles'],
        'min_travel_time': value['min_travel_time']
        }

        x_sample = get_sample(value['travel_time'], min(k, len(value['travel_time'])))
        distfit = DistFit(x_sample, n=n)
        try:
            t0 = time.perf_counter()
            bezierv_fitted, mse = distfit.fit(method=method, 
                                          step_size_PG=step_size_PG, 
                                          max_iter_PG=max_iter_PG, 
                                          threshold_PG=threshold_PG,
                                          step_size_PS=step_size_PS,
                                          max_iter_PS=max_iter_PS,
                                          solver_NL=solver_NL,
                                          max_iter_NM=max_iter_NM)
            elapsed_time = time.perf_counter() - t0
            errors.append(mse)
            times.append(elapsed_time)
        except Exception as e:
            print(f"Error fitting Bezierv for link {start_node_name} to {end_node_name}: {e} with method {method}")
            print('--------------------')
            bezierv_fitted, _ = distfit.fit(method='projgrad', 
                                          step_size_PG=step_size_PG, 
                                          max_iter_PG=max_iter_PG, 
                                          threshold_PG=threshold_PG,
                                          step_size_PS=step_size_PS,
                                          max_iter_PS=max_iter_PS,
                                          solver_NL=solver_NL)
        
        
        random = {'time':
                      {'bezierv': bezierv_fitted,
                       'mean_bezierv': bezierv_fitted.get_mean()}}
        graph.add_link(start_node_name, end_node_name, deterministic, random)
     
    pkl_file = 'data/fitted_graph_' + method + '.pkl'
    with open(pkl_file, 'wb') as f:
        pickle.dump(graph, f)
    pkl_errors = 'data/fitted_errors_' + method + '.pkl'
    with open(pkl_errors, 'wb') as f:
        pickle.dump(errors, f)
    pkl_times = 'data/fitted_times_' + method + '.pkl'
    with open(pkl_times, 'wb') as f:
        pickle.dump(times, f)

    return graph, errors, times

def get_empirical_graph(data):
    """
    Build the graph for the data with empirical distributions.
    
    Args:
        data (dict): The data to build the graph from.
        
    Returns:
        Graph: The graph with empirical samples.
    """
    graph = Graph()
    for key, value in tqdm(data.items(), desc="Building empirical graph links"):
        start_node_name = str(value['start_node'])
        end_node_name = str(value['end_node'])
        deterministic = {
            'start_longitude': value['start_longitude'],
            'start_latitude': value['start_latitude'],
            'end_longitude': value['end_longitude'],
            'end_latitude': value['end_latitude'],
            'length_miles': value['length_miles'],
            'min_travel_time': value['min_travel_time'], 
            'weight': 1
        }
        random = {'time': {'samples': value['travel_time'],
                           'mean': np.mean(value['travel_time']),
                           'variance': np.var(value['travel_time'])}}
        graph.add_link(start_node_name, end_node_name, deterministic, random)
    return graph

"""
def get_phase_graph(data, file_dir):
    params = json.load(open(file_dir))
    graph = Graph()
    for key, value in tqdm(data.items(), desc="Building empirical graph links"):
        start_node_name = str(value['start_node'])
        end_node_name = str(value['end_node'])
        deterministic = {
            'start_longitude': value['start_longitude'],
            'start_latitude': value['start_latitude'],
            'end_longitude': value['end_longitude'],
            'end_latitude': value['end_latitude'],
            'length_miles': value['length_miles'],
            'min_travel_time': value['min_travel_time']
        }
        random = {'time': {
                            'ph': (np.array(params[str(key)]['alpha']), np.array(params[str(key)]['A'])),
                            'mean': sarp_phase.ph_mean(np.array(params[str(key)]['alpha']), np.array(params[str(key)]['A']))
                           }}
        graph.add_link(start_node_name, end_node_name, deterministic, random)
    return graph
"""

def get_path_distribution(graph: Graph, 
                          path: list, 
                          n_sims: int,
                          rng: np.random.Generator) -> Bezierv:
    """
    Get the distribution of a path in the graph.
    
    Args:
        graph (Graph): The graph instance.
        path (list): The list of node indices representing the path.
        
    Returns:
        Bezierv: The Bezierv distribution of the path.
    """
    bezierv_list = [graph.nodes[path[0]].links[path[1]].random["time"]["bezierv"]]
    for i in range(1, len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        new_bezierv = graph.nodes[start_node].links[end_node].random["time"]["bezierv"]
        bezierv_list.append(new_bezierv)
    convolver = Convolver(bezierv_list)
    bezierv = convolver.convolve(method="projgrad", n_sims=n_sims, rng=rng, n=7)
    return bezierv

def convolution_experiment(
    graph_bezierv: Graph,
    graph_empirical: Graph,
    path: list,
    n_sims: int,
    rng: np.random.Generator,
    distr: str | list = "full",   # distfit candidates ('popular', 'full', or list of scipy names)
):

    bezierv_path = get_path_distribution(graph_bezierv, path, n_sims=n_sims, rng=rng)
    x_grid = np.linspace(bezierv_path.controls_x[0], bezierv_path.controls_x[-1], 1000)
    cdf_bezierv = [bezierv_path.cdf_x(x) for x in x_grid]

    fitted_models = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        samples_uv = np.asarray(graph_empirical.nodes[u].links[v].random["time"]["samples"])
        df = distfit(distr=distr, random_state=rng)
        df.fit_transform(samples_uv, verbose=0)
        fitted_models.append(df)

    totals = np.zeros(n_sims, dtype=float)
    for df in fitted_models:
        name = df.model["name"]
        params = df.model["params"]
        scipy_dist = getattr(stats, name)
        totals += scipy_dist.rvs(*params, size=n_sims, random_state=rng)

    totals_sorted = np.sort(totals)
    idx = np.searchsorted(totals_sorted, x_grid, side="right")
    cdf_mc = idx / n_sims

    return {
        "x": x_grid,
        "cdf_bezierv": cdf_bezierv,
        "cdf_montecarlo": cdf_mc,
        "mc_samples": totals,
        "fitted_models": fitted_models,
    }

def path_cdf_distfit(
    graph_empirical,
    path,
    x: float,
    *,
    n_sims: int = 100_000,
    rng: np.random.Generator | None = None,
    distr = "full",
):
    """
    Estimate `P(Total ≤ x)` for the travel-time along `path` using parametric
    fits obtained with *distfit*.

    Parameters
    ----------
    graph_empirical : Graph
        Graph whose edges contain empirical travel-time samples under
        `graph_empirical.nodes[u].links[v].random["time"]["samples"]`.
    path : list-like
        Ordered list of node IDs (v₀, v₁, …, v_k).  Consecutive pairs define
        the arcs whose times are summed.
    x : float
        Travel-time value at which the CDF is evaluated.
    n_sims : int, default 50 000
        Monte-Carlo sample size for the convolution.
    rng : np.random.Generator, optional
        Random-number generator.  `np.random.default_rng()` is used if *None*.
    distr : {"popular", "full"} | list[str], default "full"
        Candidate distributions to pass to `distfit`.

    Returns
    -------
    dict
        {
            "cdf": float,                  # P(Total ≤ x)
            "fitted_models": list[distfit] # one per arc (in path order)
        }
    """
    rng = rng or np.random.default_rng()

    fitted_models = []
    for u, v in zip(path[:-1], path[1:]):
        samples = np.asarray(
            graph_empirical.nodes[u].links[v].random["time"]["samples"]
        )
        df = distfit(distr=distr, random_state=rng)
        df.fit_transform(samples, verbose=0)
        fitted_models.append(df)

    totals = np.zeros(n_sims, dtype=float)
    for df in fitted_models:
        name = df.model["name"]
        params = df.model["params"]
        scipy_dist = getattr(stats, name)
        totals += scipy_dist.rvs(
            *params, size=n_sims, random_state=rng
        )

    cdf_value = np.mean(totals <= x)

    return float(cdf_value)

"""
def get_path_distribution_phase(graph: Graph, path: list):
    alpha = graph.nodes[path[0]].links[path[1]].random["time"]["ph"][0]
    Z = graph.nodes[path[0]].links[path[1]].random["time"]["ph"][1]
    
    for i in range(1, len(path) - 1):
        start_node = path[i]
        end_node = path[i + 1]
        link_ph = graph.nodes[start_node].links[end_node].random["time"]["ph"]
        alpha, Z = ph_convolve((alpha, Z), link_ph)
    
    return alpha, Z
"""
def get_time_budget_bezierv(graph: Graph, 
                            start_node: int, 
                            target_node: int, 
                            alpha: float, 
                            gamma: float,
                            n_sims: int,
                            rng: np.random.Generator) -> Tuple:
    """
    Calculates the time budget T based on shortest paths for mean time and cost.
    """
    
    
    mean_key = "mean_bezierv"
    shortest_mean_path, _ = dijkstra_between_nodes(graph, start_node, target_node, mean_key, "time")
    shortest_cost_path, cost_min_cost = dijkstra_between_nodes(graph, start_node, target_node, "length_miles")

    cost_min_mean = 0.0
    for i in range(len(shortest_mean_path) - 1):
        u = shortest_mean_path[i]
        v = shortest_mean_path[i + 1]
        link = graph.nodes[u].links[v]
        cost_min_mean += link.deterministic["length_miles"]
    
    bezierv_mean = get_path_distribution(graph, shortest_mean_path, n_sims=n_sims, rng=rng)
    T_t_alpha = bezierv_mean.quantile(alpha)
    bezierv_cost = get_path_distribution(graph, shortest_cost_path, n_sims=n_sims, rng=rng)
    T_c_alpha = bezierv_cost.quantile(alpha)

    T = T_t_alpha + (T_c_alpha - T_t_alpha) * (1 - gamma) + 1e-2

    return T, shortest_mean_path, cost_min_mean, bezierv_mean, shortest_cost_path, cost_min_cost, bezierv_cost

def experiments_convolution(graph_projgrad: Graph,
                            graph_phase: Graph,
                            graph_empirical: Graph,
                            n: int):
    nodes = list(graph_projgrad.nodes.keys())
    history = []
    path_lenghts = []
    time_bezierv = []
    time_phase = []
    time_empirical = []
    bezierv_mses = []
    phase_mses = []
    bezierv_diff_pos = []
    phase_diff_pos = []
    bezierv_diff_neg = []
    phase_diff_neg = []
    for i in range(n):
        start_node = random.choice(nodes)
        end_node = random.choice(nodes)
        if start_node == end_node or (start_node, end_node) in history:
            continue
        history.append((start_node, end_node))
        path = dijkstra_between_nodes(graph_empirical, start_node, end_node, 'weight')[0]       
        path_lenghts.append(len(path))

        t0 = time.perf_counter()
        bezierv_path = get_path_distribution(graph_projgrad, path, grid_size=100)
        xs = np.linspace(bezierv_path.controls_x[0], bezierv_path.controls_x[-1], 100)
        cdf_bezierv = np.array([bezierv_path.cdf_x(x) for x in xs])
        time_bezierv.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        xs = np.linspace(bezierv_path.controls_x[0], bezierv_path.controls_x[-1], 100)
        a, Z = get_path_distribution_phase(graph_phase, path)
        cdf_phase = np.array([ph_cdf(a, Z, x) for x in xs])
        time_phase.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        xs = np.linspace(bezierv_path.controls_x[0], bezierv_path.controls_x[-1], 100)
        F, samples = monte_carlo_cdf(graph_empirical, path, n_sim=10000)
        cdf_monte_carlo = np.array([F(x) for x in xs])
        time_empirical.append(time.perf_counter() - t0)

        bezierv_mses.append(np.average((cdf_bezierv - cdf_monte_carlo)**2))
        phase_mses.append(np.average((cdf_phase - cdf_monte_carlo)**2))

        list_bezierv = [cdf_bezierv[i] - cdf_monte_carlo[i] for i in range(len(xs))]
        list_phase = [cdf_phase[i] - cdf_monte_carlo[i] for i in range(len(xs))]

        bezierv_diff_pos.append(np.sum([x for x in list_bezierv if x > 0]))
        bezierv_diff_neg.append(np.sum([x for x in list_bezierv if x < 0]))
        phase_diff_pos.append(np.sum([x for x in list_phase if x > 0]))
        phase_diff_neg.append(np.sum([x for x in list_phase if x < 0]))

    return path_lenghts, bezierv_mses, phase_mses, bezierv_diff_pos, bezierv_diff_neg, phase_diff_pos, phase_diff_neg, time_bezierv, time_phase, time_empirical

def get_od_pairs(graph_empirical: Graph,
              n: int):
    nodes = list(graph_empirical.nodes.keys())
    pairs = []
    while True:
        start_node = random.choice(nodes)
        end_node = random.choice(nodes)
        if start_node != end_node:
            pairs.append((start_node, end_node))
        if len(pairs) == n:
            break
    return pairs

def run_sarp_bezier(graph: Graph, 
          start_node: int, 
          target_node: int, 
          alpha: float, 
          gamma: float,
          n_sims: int,
          rng: np.random.Generator):
    
    T_max, shortest_mean_path, cost_min_mean, bezierv_min_mean, shortest_cost_path, cost_min_cost, bezierv_min_cost = get_time_budget_bezierv(graph, start_node, target_node, alpha, gamma, n_sims=1000, rng=rng)

    max_pulse_depth = len(shortest_mean_path) + 1
    
    parameters = Parameters(
        graph=graph,
        cost_key='length_miles',
        source_node=start_node,
        target_node=target_node,
        constants={
            "T_max": T_max,
            "alpha": alpha,
            "n_sims": n_sims,
            "rng": rng
        },
        max_pulse_depth=max_pulse_depth,
        deterministic_weights=["length_miles", "min_travel_time"],
        random_weights={"time": ["bezierv"]},
        prep_deterministic_weights=["length_miles", "min_travel_time"],
        prep_random_weights={},
        info_update=sarp_bezierv.info_update,
        pulse_score=sarp_bezierv.pulse_score,
        exploration_order=sarp_bezierv.exploration_order,
        pruning_functions=[
            sarp_bezierv.prune_feasibility,
            sarp_bezierv.prune_bounds
        ]
    )
    
    t0 = time.perf_counter()
    sarp_model = sarp_bezierv.run_sarp(parameters, init_optimal_path=shortest_mean_path, init_optimal_objective=cost_min_mean)
    elapsed_time = time.perf_counter() - t0

    return sarp_model, elapsed_time, shortest_mean_path
"""
def get_time_budget_phase(graph: Graph, 
                            start_node: int, 
                            target_node: int, 
                            alpha: float, 
                            gamma: float) -> Tuple:
    shortest_mean_path, _ = dijkstra_between_nodes(graph, start_node, target_node, 'mean', "time")
    shortest_cost_path, cost_min_cost = dijkstra_between_nodes(graph, start_node, target_node, "length_miles")

    cost_min_mean = 0.0
    for i in range(len(shortest_mean_path) - 1):
        u = shortest_mean_path[i]
        v = shortest_mean_path[i + 1]
        link = graph.nodes[u].links[v]
        cost_min_mean += link.deterministic["length_miles"]
    
        phase_mean = get_path_distribution_phase(graph, shortest_mean_path)
        T_t_alpha = phase_mean.get_quantile(alpha)

        phase_cost = get_path_distribution_phase(graph, shortest_cost_path)
        T_c_alpha = phase_cost.get_quantile(alpha)

        T = T_t_alpha + (T_c_alpha - T_t_alpha) * (1 - gamma) + 1e-2

        return T, shortest_mean_path, cost_min_mean, phase_mean, shortest_cost_path, cost_min_cost, phase_cost


def run_sarp_phase(graph: Graph,
            start_node: int, 
            target_node: int, 
            alpha: float, 
            gamma: float):
        
        T_max, shortest_mean_path, cost_min_mean, phase_min_mean, shortest_cost_path, cost_min_cost, phase_min_cost = get_time_budget_phase(graph, start_node, target_node, alpha, gamma)
    
        max_pulse_depth = len(shortest_mean_path) + 1
        
        parameters = Parameters(
            graph=graph,
            source_node=start_node,
            target_node=target_node,
            constants={
                "T_max": T_max,
                "alpha": alpha
            },
            max_pulse_depth=max_pulse_depth,
            deterministic_weights=["length_miles", "min_travel_time"],
            random_weights={"time": ["ph"]},
            prep_deterministic_weights=["length_miles", "min_travel_time"],
            prep_random_weights={},
            info_update=sarp_phase.info_update,
            pulse_score=sarp_phase.pulse_score,
            exploration_order=sarp_phase.exploration_order,
            pruning_functions=[
                sarp_phase.prune_feasibility,
                sarp_phase.prune_bounds
            ]
        )

        t0 = time.perf_counter()
        sarp_model = sarp_phase.run_sarp(parameters, init_optimal_path=shortest_mean_path, init_optimal_objective=cost_min_mean)
        elapsed_time = time.perf_counter() - t0

        return sarp_model, elapsed_time, shortest_mean_path

def get_time_budget_empirical(graph: Graph, 
                            start_node: int, 
                            target_node: int, 
                            alpha: float, 
                            gamma: float) -> Tuple:
    shortest_mean_path, _ = dijkstra_between_nodes(graph, start_node, target_node, 'mean', "time")
    shortest_cost_path, cost_min_cost = dijkstra_between_nodes(graph, start_node, target_node, "length_miles")

    cost_min_mean = 0.0
    for i in range(len(shortest_mean_path) - 1):
        u = shortest_mean_path[i]
        v = shortest_mean_path[i + 1]
        link = graph.nodes[u].links[v]
        cost_min_mean += link.deterministic["length_miles"]

    T_t_alpha = sarp_montecarlo.get_quantile(graph, shortest_mean_path, alpha, n_sim=10000)

    T_c_alpha = sarp_montecarlo.get_quantile(graph, shortest_cost_path, alpha, n_sim=10000)

    T = T_t_alpha + (T_c_alpha - T_t_alpha) * (1 - gamma) + 1e-2

    return T, shortest_mean_path, cost_min_mean, shortest_cost_path, cost_min_cost


def run_sarp_montecarlo(graph: Graph,
            start_node: int, 
            target_node: int, 
            alpha: float, 
            gamma: float,
            n_sim: int=1000):
    
    T_max, shortest_mean_path, cost_min_mean, shortest_cost_path, cost_min_cost = get_time_budget_empirical(graph, start_node, target_node, alpha, gamma)

    max_pulse_depth = len(shortest_mean_path) + 1
    
    parameters = Parameters(
        graph=graph,
        source_node=start_node,
        target_node=target_node,
        constants={
            "T_max": T_max,
            "alpha": alpha,
            "n_sim": n_sim
        },
        max_pulse_depth=max_pulse_depth,
        deterministic_weights=["length_miles", "min_travel_time"],
        random_weights={"time": ["samples"]},
        prep_deterministic_weights=["length_miles", "min_travel_time"],
        prep_random_weights={},
        info_update=sarp_montecarlo.info_update_mc,
        pulse_score=sarp_montecarlo.pulse_score,
        exploration_order=sarp_montecarlo.exploration_order,
        pruning_functions=[
            sarp_montecarlo.prune_feasibility_mc,
            sarp_montecarlo.prune_bounds
        ]    
    )

    t0 = time.perf_counter()
    sarp_model = sarp_montecarlo.run_sarp(parameters, init_optimal_path=shortest_mean_path, init_optimal_objective=cost_min_mean)
    elapsed_time = time.perf_counter() - t0

    return sarp_model, elapsed_time

"""