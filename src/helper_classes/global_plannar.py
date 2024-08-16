import numpy as np
from typing import Optional, Tuple, List
import networkx as nx

from .map import RoadGraph, RoadMap
from .pathplanners.coopAstar import Astar, CAstar
from .pathplanners.pp_viz import RoadMapAnimator

class GlobalPlannar():
    """
    Runs some path searching algorithm to get a vehicle from its current position to its final position
    The path searching algorithm is based on having a non disjoint graph where the start and end points are mapped
    This plannar will plot the 'rough' path from start to the end over the entire map
    """
    def __init__(self, global_graph: RoadGraph) -> None:
        self.global_graph = global_graph
        self.path = None

    def CAstar(self, sources: List[Tuple], targets: List[Tuple], vehicles ,map_for_visualisation: Optional[RoadMap] = None) -> None:
        start_nodes = []
        end_nodes = []
        for source, target in zip(sources, targets):
            start_nodes.append(self.global_graph.get_N_nearest_vertices(source))
            end_nodes.append(self.global_graph.get_N_nearest_vertices(target))
        castar = CAstar(self.global_graph, start_nodes, end_nodes, vehicles, 0.3)
        self.paths = castar.multi_plan()
        # self.path = [x for (x,y,z) in self.paths[0]]
        print(self.global_graph.road_graph.neighbors(38))
        print([x for (x,y,z) in self.paths[2]])
        if map_for_visualisation is not None:
                # for path in self.paths:
                #     self.path = [x for (x,y,z) in path]
                #     map_for_visualisation.visualise(graph=self.global_graph.road_graph, path=self.path)
            ani = RoadMapAnimator(map_for_visualisation, self.global_graph, "ani.gif")
            ani.animate_multi_paths(self.paths)

    def Astar(self, source: Tuple, target: Tuple, map_for_visualisation: Optional[RoadMap] = None) -> None:
        start_node = self.global_graph.get_N_nearest_vertices(source)
        end_node = self.global_graph.get_N_nearest_vertices(target)
        astar = Astar(self.global_graph)
        self.path = astar.plan(start_node, end_node)
        if map_for_visualisation is not None:
            map_for_visualisation.visualise(graph=self.global_graph.road_graph, path=self.path)
            # ani = RoadMapAnimator(map_for_visualisation, self.global_graph)
            # ani.animate_single_path(self.path)

    def dijkstra(self, source: Tuple, target: Tuple, map_for_visualisation: Optional[RoadMap] = None) -> None:
        start_node = self.global_graph.get_N_nearest_vertices(source)
        end_node = self.global_graph.get_N_nearest_vertices(target)
        self.path = nx.dijkstra_path(G=self.global_graph.road_graph, source=start_node, target=end_node)
        print(self.path)
        if map_for_visualisation is not None:
            map_for_visualisation.visualise(graph=self.global_graph.road_graph, path=self.path)