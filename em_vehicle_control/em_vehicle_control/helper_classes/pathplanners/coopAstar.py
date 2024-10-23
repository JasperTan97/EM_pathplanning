from typing import List, Optional, Union, Tuple
import networkx as nx
import numpy as np
import copy
from shapely import buffer
from shapely.geometry import Polygon, Point, MultiPolygon

from ..map import RoadGraph, RoadMap
from ..vehicles import *


class AstarNode:
    """
    Node to hold Astar algorithm data
    """

    def __init__(
        self, index: int = None, g: float = None, h: float = None, parent: int = None
    ) -> None:
        self.index = index
        self.g = g
        self.h = h
        self.parent = parent  # Do we need this?

    @property
    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f < other.f
    
    def __repr__(self) -> str:
        return f"Node {self.index} with parent {self.parent}"
    

class Astar:
    """
    A* algorithm for single sourced shortest path with heuristics
    """

    def __init__(self, road_graph: RoadGraph) -> None:
        """
        Initialise A* solver
        road_graph: graph of map
        """
        self.road_graph = road_graph
        self.road_graph_details = nx.get_node_attributes(self.road_graph.full_graph, "pos")

    def get_heuristic(self, state: int, end: int) -> float:
        """
        returns the heuristic of being in a certain state
        """
        # return 0
        a = self.road_graph_details[state]
        b = self.road_graph_details[end]
        return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])

    def plan(self, start, end):
        """
        Basic Astar
        Open set is important as a priority queue for the f value
        came from holds g value and parent (but the parent value is probably not necessary)
        """
        goal_reached = False
        start_node = AstarNode(start, 0, self.get_heuristic(start, end), None)
        open_set: dict[int:AstarNode] = {}
        came_from: dict[int:AstarNode] = {}
        closed_set = set()

        open_set[start] = start_node
        came_from[start] = start_node
        while open_set:
            current_index = min(open_set, key=open_set.get)
            if current_index in closed_set:
                continue
            current_node = open_set[current_index]
            open_set.pop(current_index)
            closed_set.add(current_index)
            if current_index == end:
                goal_reached = True
                break
            neighbour_indices = self.road_graph.full_graph.neighbors(current_index)
            for neighbour_index in neighbour_indices:
                if neighbour_index in closed_set:
                    continue
                added_tentative_cost = self.road_graph.full_graph.get_edge_data(
                    current_index, neighbour_index
                )["weight"]
                if neighbour_index not in came_from:
                    old_cost = np.inf
                else:
                    old_cost = came_from[neighbour_index].g
                if added_tentative_cost + came_from[current_index].g < old_cost:
                    new_node = AstarNode(
                        neighbour_index,
                        added_tentative_cost + came_from[current_index].g,
                        self.get_heuristic(neighbour_index, end),
                        current_index,
                    )
                    open_set[neighbour_index] = new_node
                    came_from[neighbour_index] = current_node

        if goal_reached:
            rev_path = [current_index]
            parent = came_from[current_index].index
            while parent != start:
                rev_path.append(parent)
                parent = came_from[parent].index
            rev_path.append(start)
            return list(reversed(rev_path))
        else:
            return None


class CAstarNode(AstarNode):
    """
    Node to hold cooperative astar data
    """

    def __init__(
        self,
        index: int = None,
        g: float = None,
        h: float = None,
        parent: int = None,
        duration: float = None,
        timestamp: float = None,
    ) -> None:
        """
        index: node index number
        g: cost to reach from start
        h: heuristics that characterises Astar algorithm
        parent: parent node index number
        duration: time taken from parent to child node
        timestamp: time leaving this node
        """
        super().__init__(index, g, h, parent)
        self.duration = duration
        self.timestamp = timestamp

    def __repr__(self) -> str:
        return super().__repr__()
    
    def __repr__(self) -> str:
        return f"Node {self.index} with parent {self.parent}, duration {self.duration:.2f}, timestamp {self.timestamp:.2f}, and f {self.f:.2f}"


class CAstar(Astar):
    """
    Cooperative astar
    Multi agent Astar search with reservation table
    """

    def __init__(
        self,
        road_graph: RoadGraph,
        start_states: List[int],
        end_states: List[int],
        vehicles: List[Union[EdyMobile]],
        average_velocity: float,
        size_buffer: float = 0.0,
        wait_time: float = 0.5,
        time_buffer: float = 0.3
    ) -> None:
        """
        Initialise A* solver
        road_graph: graph of map
        start_state: starting node
        end_state: ending/goal node
        vehicles: list of vehicles used 
        average_velocity: to compute approximate time of arrival to nodes TODO as a list?
        size_buffer: size in metres to inflate vehicle model (or half of safe distance between vehicles) to avoid close collisions
        wait_time: time in seconds to wait
        time_buffer: buffer time in seconds to avoid close collisions
        """
        super().__init__(road_graph)
        if not (len(start_states) == len(end_states) == len(vehicles)):
            print(
                "Error: The lists of start and end states and times should have the same number of entries."
            )
            print(start_states, end_states, vehicles)
        self.start_states = start_states
        self.end_states = end_states
        self.vehicles = vehicles
        self.average_velocity = average_velocity
        self.size_buffer = size_buffer
        self.wait_time = wait_time
        self.time_buffer = time_buffer

        # Reservation table exists as a list of obstacles with start and end time where that region is "unsafe"
        self.reservation_table: List[Tuple[Polygon, float, float]] = []

    def is_node_free(self, parent_node: int, child_node: int, start_time: float, end_time: float, vehicle: Union[EdyMobile]) -> bool:
        """
        parent_node: index of parent node
        child_node: index of child node
        start_time: time leaving parent node
        end_time: time leaving child node
        vehicle: vehicle object travelling along path

        return: true if no collisions were detected
        """
        travel_polygon = self.create_swept_polygon(parent_node, child_node, vehicle)
        # searching like this could be expensive, TODO check r tree
        for (obstacle, start_reserved, end_reserved) in self.reservation_table:
            # cheaper to check time than to check obstacle
            if start_time <= end_reserved + self.time_buffer and end_time >= start_reserved - self.time_buffer:
                if travel_polygon.intersects(obstacle):
                    return False
        return True
    
    def is_not_colliding_with_map(self, parent_node: int, child_node: int, vehicle: Union[EdyMobile, Edison]) -> bool:
        """
        parent_node: index of parent node
        child_node: index of child node
        vehicle: vehicle object travelling along path

        return: true if no collisions were detected
        """
        travel_polygon = self.create_swept_polygon(parent_node, child_node, vehicle)
        return travel_polygon.within(self.road_graph.road_map.map)
    
    def create_swept_polygon(self, parent: int, child: int, vehicle: Union[EdyMobile, Edison]):
        """
        parent: index of parent node (start node)
        child: index of child node (end node)
        vehicle: vehicle class

        returns: swept polygon along a line drawn between parent and child nodes
        """
        parent = self.road_graph_details[parent]
        child = self.road_graph_details[child]
        theta = np.arctan2(child[1] - parent[1], child[0] - parent[0])

        start_pose = (*parent, theta)
        vehicle.construct_vehicle(start_pose)
        start_polygon = vehicle._vehicle_model

        end_polygon = (*child, theta)
        vehicle.construct_vehicle(start_pose)
        end_polygon = vehicle._vehicle_model

        return buffer(MultiPolygon([start_polygon, end_polygon]).convex_hull, self.size_buffer)
        
    
    def multi_plan(self, previous_paths=None):
        """
        Start the multi-agent planning.
        If previous_paths is None, it will plan without biasing for any previous path.
        """
        paths = []
        if previous_paths is None:
            previous_paths = [None] * len(self.start_states)  # Ensure correct length
        
        for start, end, vehicle, previous_path in zip(self.start_states, self.end_states, self.vehicles, previous_paths):
            paths.append(self.plan(start, end, vehicle, previous_path))
        
        return paths
    
    def bias_to_prev_path(self, node, previous_path):
        bias_mult = 0.5 # scales from the cost of 1 to bias_mult (larger is more effective, ranges from 0 to 1)
        if previous_path is None:
            return 1  # No scaling
        try:
            index = previous_path.index(node)
            return 1 - (bias_mult / (index + 1))  # scales from 
        except ValueError:
            return 1  # No scaling for nodes not in the previous path

    def plan(self, start, end, vehicle, previous_path=None):
        """
        start: index of starting node
        end: index of ending node
        """
        goal_reached = False
        start_node = CAstarNode(start, 0, self.get_heuristic(start, end), None, 0, 0)
        open_set: list[CAstarNode] = []
        came_from: dict[int:AstarNode] = {} # {child : parent}
        closed_set = set()

        open_set.append(start_node)
        came_from[start] = start_node
        while open_set:
            current_node = min(open_set)
            open_set.remove(current_node) # always remove first
            if (current_node.index, current_node.timestamp + current_node.duration) in closed_set:
                continue
            closed_set.add((current_node.index, current_node.timestamp + current_node.duration))
            if current_node.index == end:
                goal_reached = True
                break
            neighbour_indices = self.road_graph.full_graph.neighbors(current_node.index)
            for neighbour_index in neighbour_indices:
                added_tentative_cost = self.road_graph.full_graph.get_edge_data(
                    current_node.index, neighbour_index
                )["weight"] * self.bias_to_prev_path(neighbour_index, previous_path)
                travel_duration = added_tentative_cost / self.average_velocity
                travel_start_time = current_node.timestamp
                travel_end_time = travel_duration + travel_start_time
                if (neighbour_index, travel_end_time) in closed_set:
                    continue
                if neighbour_index not in came_from:
                    old_cost = np.inf
                else:
                    old_cost = came_from[neighbour_index].g
                if added_tentative_cost + current_node.g > old_cost:
                    continue
                # if not self.is_not_colliding_with_map(current_node.index, neighbour_index, vehicle):
                #     continue
                if not self.is_node_free(
                    current_node.index, neighbour_index, travel_start_time, travel_end_time, vehicle
                ):
                    continue
                # here we have a collision free node of lowest cost
                new_node = CAstarNode(
                    neighbour_index,
                    g=added_tentative_cost + current_node.g,
                    h=self.get_heuristic(neighbour_index, end),
                    parent=current_node.index,
                    duration=travel_duration,
                    timestamp=travel_end_time,
                )
                open_set.append(new_node)
                came_from[neighbour_index] = current_node
            new_node = copy.deepcopy(current_node)
            new_node.timestamp = current_node.timestamp + self.wait_time
            new_node.g = current_node.g + self.wait_time * self.average_velocity
            open_set.append(new_node)

        if goal_reached:
            rev_path = [(current_node.index, current_node.timestamp, current_node.timestamp+current_node.duration)]
            parent = came_from[current_node.index]
            while parent.index != start:
                # index, time leaving parent node, time leaving current node
                rev_path.append((parent.index, came_from[parent.index].timestamp, parent.timestamp))
                parent = came_from[parent.index]
            rev_path.append((start,-1,0))
            path = list(reversed(rev_path))
            for i in range(len(rev_path)-1):
                swept_poly = self.create_swept_polygon(path[i][0], path[i+1][0], vehicle)
                self.reservation_table.append((swept_poly, path[i][1], path[i][2]))
            return path
        else:
            print(f"Path searching from {start} to {end} nodes failed")
            # this will never happen due to infinite time dimension
            return None