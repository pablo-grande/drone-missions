"""Creates a drone mission with SLH framework heuristic components."""
import logging
import textwrap

import numpy as np

import map_utils

from random import seed as random_seed

from slh_framework.datasets import TestInstance
from slh_framework.algorithms import simulated_annealing_heuristic as heuristic
from slh_framework.simulations import MonteCarlo as Simulation


logger = logging.getLogger("Drone mission")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)


map_size = 10
number_of_nodes = 10
seed = 8634452
test = TestInstance("drone_sweep")
test.initial_temp = 1000
test.cooling_rate = 0.95
test.min_temp = 0.1
test.instance_data["fleet_size"] = 4
test.instance_data["route_max_cost"] = 42.5

random_seed(seed)
np.random.seed(seed)

Simulation.condition_factors = {
    "weather": {"factor": 0.2},
    "unexplored_area": {"factor": 0.4},
    # in LTE network, a UE measures two parameters on reference signal: 
    # * RSRP (Reference Signal Received Power)
    # * RSRQ (Reference Signal Received Quality)
    "rsrq": {"factor": 0.5},
    "rsrp": {"factor": 0.1},
}


signal_quality, node_list = map_utils.create_map_and_nodes(map_size, number_of_nodes)
test.instance_data["node_list"] = node_list
test.instance_data["number_of_nodes"] = number_of_nodes

logger.info("Starting mission using simulated annealing heuristic")

best_solution = heuristic(
    test, test.instance_data, Simulation.simulation
)

route_data = "\n".join(textwrap.indent(str(route), '\t') for route in best_solution.routes)
logger.info(
    f"""
    Best solution:
    cost: {best_solution.cost}
    reward: {best_solution.reward}
    route: 
    {route_data}
    """
)

for index, route in enumerate(best_solution.routes):
    map_utils.plot(signal_quality, route, filename=f"simulated_annealing_route_{index}")
