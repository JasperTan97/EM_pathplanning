from typing import List, Optional
import networkx as nx
import numpy as np

from map import RoadGraph, RoadMap
from global_plannar import GlobalPlannar

class SippState():
    def __init__(self, f: float = None, g: float = None, parent: int = None) -> None:
        """
        f : cost of state with heuristics
        g : lowest cost from start state
        """
        self.f = f
        self.g = g
        self.parent = parent

class Sipp():
    """
    
    """
    def __init__(self, 
                 road_graph: RoadGraph, 
                 start_states: List[int], 
                 end_states: List[int],
                 average_velocity: float) -> None:
        """
        Initialise SIPP solver
        road_graph: graph of map
        start_states: starting nodes of all agents
        end_states: ending/goal nodes of corresponding agents
        average_velocity: to compute approximate time of arrival to nodes
        """
        self.road_graph = road_graph
        self.start_states = start_states
        self.end_states = end_states
        self.average_velocity = average_velocity
        
    def get_successors(self, current_state: int) -> List[int]:
        """
        SIPP method to get successors based on safe interval principal
        """
        neighbours = self.road_graph.road_graph.neighbors(current_state)
        for neighbour in neighbours


    def get_heuristic(self, state: int) -> float:
        """
        A_star heuristic
        """

    def plan(self, start, end) -> List[int]:
        start_state = SippState(0,0)
        OPEN_set = {start : start_state}
        current_state = start
        goal_reached = False
        while not goal_reached:
            current_state = min(OPEN_set, key=OPEN_set.f.get)
            successors = self.get_successors(current_state)
            for s_prime in successors:
                if s_prime not in OPEN_set: # visited
                    OPEN_set[s_prime] = SippState()
                    OPEN_set[s_prime].g = np.inf
                c_s_sprime = self.cost_between(current_state, s_prime)
                if OPEN_set[s_prime].g > OPEN_set[current_state].g + c_s_sprime:
                    OPEN_set[s_prime].g = OPEN_set[current_state].g + c_s_sprime
                    # update_time(s_prime) ?
                    OPEN_set[s_prime].f = OPEN_set[s_prime].g + self.get_heuristic(s_prime)
                    OPEN_set[s_prime].parent = current_state
                    if s_prime == end:
                        goal_reached = True
                        return self.get_path(s_prime)