import numpy as np
from typing import Optional, Tuple
import networkx as nx

from .map import RoadGraph, RoadMap

class GlobalPlannar():
    """
    Runs some path searching algorithm to get a vehicle from its current position to its final position
    The path searching algorithm is based on having a non disjoint graph where the start and end points are mapped
    This plannar will plot the 'rough' path from start to the end over the entire map
    """
    def __init__(self, global_graph: RoadGraph) -> None:
        self.global_graph = global_graph
        self.path = None

    def dijkstra(self, source: Tuple, target: Tuple, map_for_visualisation: Optional[RoadMap] = None) -> None:
        start_node = self.global_graph.get_N_nearest_vertices(source)
        end_node = self.global_graph.get_N_nearest_vertices(target)
        self.path = nx.dijkstra_path(G=self.global_graph.road_graph, source=start_node, target=end_node)
        if map_for_visualisation is not None:
            map_for_visualisation.visualise(graph=self.global_graph.road_graph, path=self.path)