from pulsekit.pulse import Pulse
from bezierv.classes.convolver import Convolver
import numpy as np


def info_update(graph, current_node, reachable_node, path, deterministic_info, random_info):
    """
    Updates path information when traversing a link.

    Args:
        graph: The graph instance.
        current_node: The starting node of the link.
        reachable_node: The ending node of the link.
        path: The current path (list of node indices).
        deterministic_info: Dictionary of deterministic path properties.
        random_info: Dictionary of random path properties.

    Returns:
        A tuple containing the updated deterministic and random info dictionaries.
    """
    link = graph.nodes[current_node].links[reachable_node]
    deterministic_info["length_miles"] += link.deterministic["length_miles"]

    return deterministic_info, random_info


def get_montecarlo_prob(list_bezierv: list, value: float, n_sims: int, rng: np.random.Generator):
    sims = np.zeros(n_sims)
    for bezierv in list_bezierv:
        sims += bezierv.random(n_sims, rng=rng)
    cdf = np.sort(sims)
    prob = np.sum(cdf <= value) / n_sims
    return prob
    

def prune_feasibility(pulse_alg, current_node, current_path_info):
    """
    Prunes a path if it's not feasible based on probabilistic constraints.

    Args:
        pulse_alg: The Pulse algorithm instance.
        current_node: The current node being evaluated.
        current_path_info: The current path information.

    Returns:
        True if the path should be pruned, False otherwise.
    """
    if len(current_path_info.path) < 1:
        return False
    path_util = current_path_info.path.copy()
    path_util.append(current_node)
    list_bezierv = []
    for node in range(len(path_util) - 1):
        list_bezierv.append(pulse_alg.parameters.graph.nodes[path_util[node]].links[path_util[node + 1]].random["time"]["bezierv"])
    
    T_max = pulse_alg.parameters.constants["T_max"]
    alpha = pulse_alg.parameters.constants["alpha"]

    prob = get_montecarlo_prob(list_bezierv, 
                               T_max - pulse_alg.preprocessing.deterministic["min_travel_time"][current_node],
                               pulse_alg.parameters.constants["n_sims"],
                               pulse_alg.parameters.constants["rng"]
                               )
    
    if prob <= alpha:
        #print("Pruning path at node", current_node, "with probability:", prob)
        return True

    return False


def prune_bounds(pulse_alg, current_node, current_path_info):
    """
    Prunes a path if its cost exceeds the current best. Updates the best path
    if a better solution is found at the target.

    Args:
        pulse_alg: The Pulse algorithm instance.
        current_node: The current node being evaluated.
        current_path_info: The current path information.

    Returns:
        True if the path should be pruned, False otherwise.
    """
    deterministic_info = current_path_info.deterministic
    
    if deterministic_info["length_miles"] + pulse_alg.preprocessing.deterministic["length_miles"][current_node] > pulse_alg.current_optimal_objective:
        #print("Pruning path at node", current_node, "with length:", deterministic_info["length_miles"])
        return True
            
    return False


def exploration_order(pulse_alg, node):
    """
    Defines the order in which to explore neighboring nodes.

    Args:
        pulse_alg: The Pulse algorithm instance.
        node: The node index.

    Returns:
        The pre-calculated cost to reach the target from the given node.
    """
    return pulse_alg.preprocessing.deterministic["length_miles"][node]


def pulse_score(pulse_alg, current_path_info):
    """
    Scores a path to determine its priority in the pulse queue.

    Args:
        pulse_alg: The Pulse algorithm instance.
        current_path_info: The current path information.

    Returns:
        The score of the path.
    """
    return pulse_alg.preprocessing.deterministic["length_miles"][current_path_info.path[-1]]

def run_sarp(parameters, init_optimal_path=[], init_optimal_objective=float('inf')):
    pulse_alg = Pulse(parameters)
    pulse_alg.preprocess()
    pulse_alg.run(init_optimal_path=init_optimal_path, init_objective=init_optimal_objective)
    return pulse_alg