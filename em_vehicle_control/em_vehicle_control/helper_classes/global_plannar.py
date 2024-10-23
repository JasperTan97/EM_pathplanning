import numpy as np
from typing import Optional, Tuple, List
import networkx as nx
from shapely import Point

from .map import RoadGraph, RoadMap
from .pathplanners.coopAstar import Astar, CAstar
from .pathplanners.pp_viz import RoadMapAnimator

class GlobalPlannar():
    """
    To use currently: 1. initialise class 2. run CAstar method 3. generate paths
    To change to: 1. initialise class 2. run plan function -> decides if RRT star or CAstar is best 3. Run the right methods 4. Generate paths
    """
    def __init__(self, road_graph: RoadGraph) -> None:
        self.road_graph = road_graph
        self.path = None
        self.paths = None # contains tuples of (index of node, time leaving parent node, and time leaving current node)
        self.spacetime_paths = None

    def identify_starting_path(self, source, path):
        """
        Using the source (robot pose), identify the two nodes which forms the LineString the robot is on
        """
        # for each edge in path
        #   check the distance from source
        #   if distance is small
        #   return those 2 nodes that formed the edge
        distance_threshold = 0.1 # metres
        robot_pose = Point(source)
        graph_geom = nx.get_edge_attributes(self.road_graph.full_graph, "geometry")
        for i in range(len(path)-1):
            node_A, node_B = path[i:i+2]
            path_geom = graph_geom[(node_A, node_B)]
            if robot_pose.distance(path_geom) < distance_threshold:
                return (node_A, node_B)
        else:
            return None

    def CAstar(self, sources: List[Tuple], targets: List[Tuple], vehicles, map_for_visualisation: Optional[RoadMap] = None) -> None:
        start_nodes = []
        end_nodes = []
        print("Spinning CAstar", flush=True)
        if self.paths is None:
            previous_paths_indices = [None] * len(sources)
        else:
            previous_paths_indices = []
            for previous_path in self.paths:
                # just keep indexes
                previous_paths_indices.append([x[0] for x in previous_path])
        # self.road_graph.loop_graph(sources, 0.02)
        start_segments = []
        for source, target, previous_path in zip(sources, targets, previous_paths_indices):
            # to choose starting node:
            if previous_path is not None:
                start_segment = self.identify_starting_path((source[0], source[1]), previous_path)
                print(f"Robot and Start segment were {(source[0], source[1])},{start_segment}")
            else:
                start_segment = None
                print("Paths were none")
            if start_segment is not None:
                start_nodes.append(start_segment[1])
                start_segments.append(start_segment)
            else:
                start_segments.append(None)
                start_nodes.append(self.road_graph.get_N_nearest_vertices((source[0], source[1])))
            end_nodes.append(self.road_graph.get_N_nearest_vertices((target[0], target[1])))
        castar = CAstar(self.road_graph, start_nodes, end_nodes, vehicles, 0.3)
        self.paths = castar.multi_plan(previous_paths_indices)
        for i, (path, start_segment) in enumerate(zip(self.paths, start_segments)):
            # append the first part of the start segment at the beginning
            # this restores the smoothness of the curves
            # note that path is defined by tuples of 
            # (index, time leaving parent node, time leaving current node)
            if start_segment is not None:
                prev_path_segment = [(start_segment[0], -1, 0)] # time is inconsequential since robot is already travelling
                self.paths[i] = prev_path_segment + path
                print("Path: ", [x[0] for x in self.paths[0]])
            else:
                print("Somehow start segment was none")
        if map_for_visualisation is not None:
                # for path in self.paths:
                #     self.path = [x for (x,y,z) in path]
                #     map_for_visualisation.visualise(graph=self.global_graph.road_graph, path=self.path)
            ani = RoadMapAnimator(map_for_visualisation, self.road_graph, "ani.gif")
            ani.animate_multi_paths(self.paths)

    def generate_paths(self):
        if self.paths is None:
            print("ERROR: Paths have not been generated")
        road_graph_details = nx.get_node_attributes(self.road_graph.full_graph, "pos")
        road_graph_geom = nx.get_edge_attributes(self.road_graph.full_graph, "geometry")
        self.spacetime_paths = []
        for path in self.paths:
            spacetime_path = []
            for i in range(len(path) - 1):
                node, next_node = path[i:i+2]
                # curr_pos = road_graph_details[node[0]]
                # next_pos = road_graph_details[next_node[0]]
                path_geom = road_graph_geom[(node[0], next_node[0])]
                for point in list(path_geom.coords):
                    spacetime_path.append((point, node[-1]))
                # spacetime_path.append((curr_pos, angle, node[-1])) # pos, angle, time to leave
            self.spacetime_paths.append(spacetime_path)
        return self.spacetime_paths

    def Astar(self, source: Tuple, target: Tuple, map_for_visualisation: Optional[RoadMap] = None) -> None:
        start_node = self.road_graph.get_N_nearest_vertices((source[0], source[1]))
        end_node = self.road_graph.get_N_nearest_vertices((target[0], target[1]))
        astar = Astar(self.road_graph)
        self.path = astar.plan(start_node, end_node)
        # print(self.path)
        if map_for_visualisation is not None:
            map_for_visualisation.visualise(graph=self.road_graph.full_graph, path=self.path)
            # ani = RoadMapAnimator(map_for_visualisation, self.global_graph)
            # ani.animate_single_path(self.path)

    def dijkstra(self, source: Tuple, target: Tuple, map_for_visualisation: Optional[RoadMap] = None) -> None:
        start_node = self.road_graph.get_N_nearest_vertices((source[0], source[1]))
        end_node = self.road_graph.get_N_nearest_vertices((target[0], target[1]))
        self.path = nx.dijkstra_path(G=self.road_graph.full_graph, source=start_node, target=end_node)
        # print(self.path)
        if map_for_visualisation is not None:
            map_for_visualisation.visualise(graph=self.road_graph.full_graph, path=self.path)