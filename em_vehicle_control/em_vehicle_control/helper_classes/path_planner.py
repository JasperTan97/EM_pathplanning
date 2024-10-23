import numpy as np
from typing import Optional, Tuple, List, Union
import networkx as nx
from shapely import Point

from .map import RoadTrack, RoadMap
from .pathplanners.coopAstar import Astar, CAstar
from .pathplanners.rrt import RRT_star_Reeds_Shepp
from .pathplanners.pp_viz import RoadMapAnimator
from .vehicles import *

PosePt2D = Tuple[float, float, float]  # (x, y, yaw) values
Pos = Tuple[float, float]
AllVehicles = Union[Edison, EdyMobile]
NodePath = List[int]


class PlanningAlgoData:
    """
    Helper class that returns algorithm name to use, as well as additional information if necessary
    """

    def __init__(self, pp_algo: str) -> None:
        self.pp_algo = pp_algo
        """
        Options: RRT*, CA*, reuse_previous 
        """
        self.current_edge = None  # for 'CA*', defined by 2 edges
        self.start_node = None  # for 'CA*'
        self.end_node = None  # for 'CA*'
        self.target = None  # for RRT*, if target is to join a graph node

    def set_start_edge_and_end_node(
        self, start_edge: Tuple[int, int], end_node: int
    ) -> None:
        if self.pp_algo != "CA*":
            print("WARNING: Defining current edge when algorithm does not use graph")
        if self.start_node != None:
            print(
                "WARNING: Defining starting edge when starting node has been already defined"
            )
        node_A, node_B = start_edge
        self.current_edge = (node_A, node_B)
        self.start_node = node_B
        self.end_node = end_node

    def set_start_and_end_nodes(self, start_node: int, end_node: int) -> None:
        if self.pp_algo != "CA*":
            print("WARNING: Defining current edge when algorithm does not use graph")
        if self.current_edge != None:
            print(
                "WARNING: Defining starting node when starting edge has been already defined"
            )
        self.start_node = start_node
        self.end_node = end_node

    def set_target(self, start_x, start_y, start_yaw=0) -> None:
        if self.pp_algo != "RRT*":
            print("WARNING: Wrong algorithm to set target pose.")
        self.target = (start_x, start_y, start_yaw)

    def __repr__(self) -> str:
        if self.pp_algo == "RRT*":
            if self.target is not None:
                return f"RRT* with node at {self.target} as target. "
            else:
                return f"RRT* with final goal as target. "
        if self.pp_algo == "CA*":
            if self.current_edge is not None:
                return f"CA* with {self.current_edge} as current edge and {self.end_node} as end node"
            else:
                return f"CA* with start and end nodes as {self.start_node}, {self.end_node}. "


class PathPointDatum:
    """
    Helper class to contain positional and time information,
    as well as direction
    """

    def __init__(self, x: float, y: float, time: float, direction: int = 1) -> None:
        """
        x: x position
        y: y position
        time: time passed before leaving is allowed
        direction: 1 is forward -1 is backward
        """
        self.x = x
        self.y = y
        self.time = time
        self.direction = direction

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"


class PathPlanner:
    """
    Class that manages which planning algorithm to use
    Runs the selected algorithm
    and generates path messages
    """

    def __init__(
        self, road_map: RoadMap, road_graph: RoadTrack, num_vehicles: float
    ) -> None:
        self.road_map = road_map
        self.road_graph = road_graph
        self.num_vehicles = num_vehicles
        self.node_paths = [
            None for _ in range(num_vehicles)
        ]  # contains tuples of (index of node, time leaving parent node, and time leaving current node)
        self.pose_paths: List[List[PosePt2D]] = [
            None for _ in range(num_vehicles)
        ]  # contains 2D poses
        self.graph_geom = nx.get_edge_attributes(
            self.road_graph.full_graph, "geometry"
        )  # linestring data of graph
        self.graph_node_pos: dict[int, Tuple[float, float]] = nx.get_node_attributes(
            self.road_graph.full_graph, "pos"
        )

        self.source_distance_threshold = 0.05  # in metres, distance to edge/node to determine sampling or graph search
        self.goal_distance_threshold = 0.3  # in metres, distance to edge/node to determine sampling or graph search
        self.path_distance_threshold = (
            0.05  # in metres, distance to nearest edge for graph search
        )
        self.RRT_goal_radius = 0.1
        self.RRT_max_iter = 20
        self.average_vel = 0.3

    def identify_starting_edge(
        self, source: PosePt2D, path: NodePath
    ) -> Tuple[int, int]:
        """
        Using the source (robot pose), identify the two nodes which forms the LineString the robot is on

        Args:
            source (PosePt2D): current pose
            path (NodePath): current tracked path
        Returns:
            Tuple[int, int]: edge described by starting and ending node indices
        """
        robot_pose = Point(source[0], source[1])
        for i in range(len(path) - 1):
            node_A, node_B = path[i : i + 2]
            path_geom = self.graph_geom[(node_A[0], node_B[0])]
            if robot_pose.distance(path_geom) < self.path_distance_threshold:
                return (node_A[0], node_B[0])
        else:
            return None

    def determine_path_algo(
        self, index: int, source: PosePt2D, goal: PosePt2D
    ) -> PlanningAlgoData:
        """
        Checks in order:
        1. close to goal (RRT*)
        2. close to path edge (CA*)
        3. close to graph node (CA*)
        4. else RRT*

        index: index of vehicle
        source: starting pose
        goal: desired pose
        return: 'RRT*' or 'CA*'
        """
        source_goal_distance = np.sqrt(
            (source[0] - goal[0]) ** 2 + (source[1] - goal[1]) ** 2
        )
        if source_goal_distance < self.goal_distance_threshold:
            return PlanningAlgoData("RRT*")
        start_node = self.road_graph.get_N_nearest_vertices((source[0], source[1]))
        end_node = self.road_graph.get_N_nearest_vertices((goal[0], goal[1]))
        if self.node_paths[index] != None:
            start_edge = self.identify_starting_edge(source, self.node_paths[index])
            if start_edge is not None:
                algo_plan_data = PlanningAlgoData("CA*")
                algo_plan_data.set_start_edge_and_end_node(start_edge, end_node)
                return algo_plan_data
            # if start edge is not found, the robot is too far from the path.
        start_node_pos = self.graph_node_pos[start_node]
        source_graph_distance = np.sqrt(
            (source[0] - start_node_pos[0]) ** 2 + (source[1] - start_node_pos[1]) ** 2
        )
        if source_graph_distance > self.source_distance_threshold:
            algo_plan_data = PlanningAlgoData("RRT*")
            dijk_path = nx.dijkstra_path(
                G=self.road_graph.full_graph, source=start_node, target=end_node
            )
            pose_1 = self.graph_node_pos[dijk_path[1]]
            dir_to_goal = np.arctan2(
                pose_1[1] - start_node_pos[1], pose_1[0] - start_node_pos[0]
            )
            algo_plan_data.set_target(*start_node_pos, dir_to_goal)
            return algo_plan_data
        else:
            algo_plan_data = PlanningAlgoData("CA*")
            algo_plan_data.set_start_and_end_nodes(start_node, end_node)
            return algo_plan_data

    def is_close_to_path(self, pose: PosePt2D, path: List[PosePt2D]) -> bool:
        """
        Checks whether pose is close to path, using the path_distance_threshold

        Args:
            pose (PosePt2D): The robot's current pose
            path (List[PosePt2D]): List of desired poses in order
        """
        pdt_squared = self.path_distance_threshold**2
        for path_pose in path:
            if (
                pdt_squared
                < (pose[0] - path_pose[0]) ** 2 + (pose[1] - path_pose[1]) ** 2
            ):
                return True
        return False

    # Helper function to convert node paths to time paths for CA*
    def convert_nodes_to_time_path(self, node_path: NodePath) -> List[PathPointDatum]:
        path_time_points = []
        for node, next_node in zip(node_path[:-1], node_path[1:]):
            path_geom = self.graph_geom[(node[0], next_node[0])]
            for point in path_geom.coords:
                path_time_points.append(
                    PathPointDatum(x=point[0], y=point[1], time=node[-1])
                )
        return path_time_points

    # Helper function to convert pose paths to time paths for RRT*
    def convert_pose_to_time_path(
        self, pose_path: List[PosePt2D]
    ) -> List[PathPointDatum]:
        return [
            PathPointDatum(x=pose[0], y=pose[1], time=0, direction=pose[2])
            for pose in pose_path
        ]

    def plan(
        self,
        sources: List[PosePt2D],
        goals: List[PosePt2D],
        vehicles: List[AllVehicles],
        map_for_visualisation: Optional[RoadMap] = None,
    ) -> List[List[PathPointDatum]]:
        """
        Using all sources and goals, plots a path in space and time to guide all robots towards their destination

        sources: list of starting 2D poses
        goals: list of final 2D poses respectively
        vehicles: list of vehicle classes for dynamic obstacle avoidance
        map_for_visualisation: to visualise plan TODO

        return: list of pose time paths
        """
        if (
            len(sources) != self.num_vehicles
            or len(goals) != self.num_vehicles
            or len(vehicles) != self.num_vehicles
        ):
            print(
                "Error, length of sources, goals and vehicles must be equal to the number of vehicles defined"
            )
            return
        path_algos = []

        for index, (source, goal) in enumerate(zip(sources, goals)):
            path_algos.append(self.determine_path_algo(index, source, goal))

        print("Calculated: ", path_algos, flush=True)
        pose_time_paths = [None] * self.num_vehicles

        # Handle all RRT* planning first
        for index, path_algo in enumerate(path_algos):
            if path_algo.pp_algo == "RRT*":
                if self.pose_paths[index] is not None:
                    prev_pose_path = self.pose_paths[index]
                    if self.is_close_to_path(source, prev_pose_path):
                        continue
                if path_algo.target is not None:
                    goal = path_algo.target
                local_map = self.road_map.get_local_map(
                    [(source[0], source[1]), (goal[0], goal[1])]
                )
                pose_path = RRT_star_Reeds_Shepp.create_and_plan(
                    source,
                    goal,
                    0.1,
                    vehicles[index],
                    local_map,
                    max_iter=20,
                    search_until_max_iter=False,
                    visualise=False,
                )
                self.pose_paths[index] = pose_path
                self.node_paths[index] = None
                # Convert to pose_time_path
                pose_time_paths[index] = self.convert_pose_to_time_path(pose_path)

        # Handle CA* planning in bulk
        castar_sources, castar_targets, castar_vehicles, prev_castar_paths = (
            [],
            [],
            [],
            [],
        )
        for index, path_algo in enumerate(path_algos):
            if path_algo.pp_algo == "CA*":
                castar_sources.append(path_algo.start_node)
                castar_targets.append(path_algo.end_node)
                castar_vehicles.append(vehicles[index])
                prev_castar_paths.append(self.node_paths[index])
        castar = CAstar(
            self.road_graph,
            castar_sources,
            castar_targets,
            castar_vehicles,
            average_velocity=0.3,
        )
        prev_paths_tmp = [
            [node[0] for node in path] if path is not None else None
            for path in prev_castar_paths
        ]
        castar_paths = castar.multi_plan(prev_paths_tmp)

        for index, path_algo in enumerate(path_algos):
            if path_algo.pp_algo == "CA*":
                self.node_paths[index] = castar_paths.pop(0)
                self.pose_paths[index] = None  # CA* only updates node paths
                # Always send the new path for CA*
                pose_time_paths[index] = self.convert_nodes_to_time_path(
                    self.node_paths[index]
                )

        return pose_time_paths
