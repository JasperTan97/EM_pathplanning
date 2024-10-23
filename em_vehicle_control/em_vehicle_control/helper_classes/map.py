import shapely
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import CirclePolygon as MplCirclePolygon, FancyArrowPatch
import matplotlib.patches as patches
from typing import Tuple, Optional, List, Dict
import numpy as np
import itertools
import networkx as nx
from scipy.spatial import KDTree
from scipy.interpolate import make_interp_spline, CubicSpline
from copy import deepcopy


class RoadSegment:
    """A road segment describes the full rectangular part of the road"""

    # Note: This class only works for axis_aligned roads

    def __init__(
        self,
        point_A: Tuple[float, float],
        point_B: Tuple[float, float],
        invert_direction: bool = False,
    ):
        """
        pointA: (x, y) coordinate of bottom left point
        pointB: (x, y) coordinate of top right point
        invert_direction: False is the long side of the road is the direction the car moves in, true if otherwise
        """
        cornerBL = (point_A[0], point_A[1])
        cornerBR = (point_B[0], point_A[1])
        cornerTL = (point_A[0], point_B[1])
        cornerTR = (point_B[0], point_B[1])
        self.x_min = point_A[0]
        self.x_max = point_B[0]
        self.y_min = point_A[1]
        self.y_max = point_B[1]
        self.x_length = np.abs(point_B[0] - point_A[0])
        self.y_length = np.abs(point_B[1] - point_A[1])
        self.road_segment_polygon = Polygon((cornerBL, cornerBR, cornerTR, cornerTL))
        if not self.road_segment_polygon.is_valid:
            print(
                f"Error: The outer boundary given is not valid for road {point_A}, {point_B}"
            )

        if self.x_length > self.y_length:
            # if the length along the x-axis is longer than the length along the y-axis
            # i.e., the road is going from left to right
            self.direction = -1
        else:
            self.direction = 1  # the road is going up to down

        if invert_direction:
            self.direction *= -1


class RoadMap:
    """
    The road map is the full map consisting of parallel and perpendicular road segments

    """

    def __init__(self, roads: list[RoadSegment], debug: bool = False) -> None:
        self.roads = roads
        self.debug = debug
        road_intersections = self.get_intersections()
        self.road_intersections = road_intersections
        self.road_intersections_poly = [x[0] for x in road_intersections]
        self.stations = []  # TODO should this be a dictionary with names?
        self.map = shapely.unary_union([road.road_segment_polygon for road in roads])

    def get_intersections(self) -> list[Tuple[Polygon, RoadSegment, RoadSegment]]:
        """
        Function that iteratively looks through all roads and identify intersection points
        Returns a list of tuples that include the Polygon of the intersection
        and the indices of the 2 roads that creates the intersection
        """
        road_intersections = []
        for i in range(len(self.roads)):
            for j in range(i + 1, len(self.roads)):
                if shapely.intersects(
                    self.roads[i].road_segment_polygon,
                    self.roads[j].road_segment_polygon,
                ):
                    road_intersection = shapely.intersection(
                        self.roads[i].road_segment_polygon,
                        self.roads[j].road_segment_polygon,
                    )
                    if self.debug:
                        print(
                            f"Intersection ({road_intersection}) created by roads ({i}, {self.roads[i]}) and ({j}, {self.roads[j]})"
                        )
                    road_intersections.append((road_intersection, i, j))
        return road_intersections

    def add_station(
        self,
        location: Tuple[float, float],
        orientation: float,
        radius: Optional[float] = 0.05,
    ) -> Tuple[Polygon, float]:
        """
        Method to add stations, signfying the end goal of the robot
        location: (x, y) coordinates of the station
        orientation: final orientation of the vehicle
        radius: radius describing the tolerance of the end point (default at 5cm radius)
        """
        station = Point(location)
        station.buffer(radius, resolution=100)
        self.stations.append((station, location, orientation, radius))

    def get_local_map(self, locations: List[Tuple[float, float]], radius: float = 0.5) -> Polygon:
        """
        points: (x,y) list of points of interest
        radius: radius of important region around each point
        return: intersection between convex hull of all important regions and the map
        """
        buffers = [Point(loc).buffer(radius) for loc in locations]
        union_buffers = unary_union(buffers)
        region_of_interest = union_buffers.convex_hull
        local_map = self.map.intersection(region_of_interest)
        return local_map

    @staticmethod
    def shapely_to_mplpolygons(
        shapely_polygon: Polygon, colour: str = "lightblue"
    ) -> Tuple[MplPolygon, MplPolygon]:
        x, y = shapely_polygon.exterior.xy
        exterior_polygon = MplPolygon(
            list(zip(x, y)), closed=True, edgecolor="black", facecolor=colour, alpha=0.5
        )
        hole_polygons = []
        for hole in shapely_polygon.interiors:
            x, y = hole.xy
            hole_polygons.append(
                MplPolygon(
                    list(zip(x, y)),
                    closed=True,
                    edgecolor="black",
                    facecolor="white",
                    alpha=1,
                )
            )
        return exterior_polygon, hole_polygons

    def visualise(
        self,
        show_intersections: bool = False,
        show_stations: bool = True,
        graph: Optional[nx.Graph] = None,
        path: Optional[List] = None,
    ) -> None:
        """Visualises the built map"""
        _, ax = plt.subplots()
        exterior_polygon, interior_polygons = self.shapely_to_mplpolygons(self.map)
        ax.add_patch(exterior_polygon)
        for hole_polygon in interior_polygons:
            ax.add_patch(hole_polygon)
        if show_intersections:
            for road_intersection in self.road_intersections:
                rd_int_polygon, _ = self.shapely_to_mplpolygons(
                    road_intersection[0], "purple"
                )
                ax.add_patch(rd_int_polygon)
        if show_stations:
            for station in self.stations:
                station_polygon = MplCirclePolygon(
                    station[1],
                    radius=station[3],
                    resolution=100,
                    facecolor="green",
                    alpha=0.5,
                )
                ax.add_patch(station_polygon)
                # print(station[1][0], station[1][1], station[1][0]+station[3]*np.cos(station[2]), station[1][1]+station[3]*np.sin(station[2]))
                ax.arrow(
                    station[1][0],
                    station[1][1],
                    station[3] * np.cos(station[2]),
                    station[3] * np.sin(station[2]),
                    width=station[3] / 10,
                )
        if graph is not None:
            pos = nx.get_node_attributes(graph, "pos")
            nx.draw(
                graph,
                pos,
                # node_size=500,
                with_labels=True,
                node_color="skyblue",
                # font_weight="bold",
                edgelist=[],
                edge_color="gray",
                font_size=10,  # Increase the font size for better visibility
                ax=ax,  # Specify the axis to plot on
            )
            for u, v, data in graph.edges(data=True):
                if "geometry" in data:
                    line = data["geometry"]  # This should be a LineString
                    x, y = line.xy  # Get the x, y coordinates from the LineString
                    ax.plot(x, y, color="gray")
            if path is not None:
                path_edges = list(zip(path, path[1:]))
                nx.draw_networkx_nodes(
                    graph,
                    pos,
                    nodelist=path,
                    node_size=7,
                    node_color="green",
                )
                nx.draw_networkx_edges(
                    graph, pos, edgelist=path_edges, edge_color="green"
                )
        ax.axis("scaled")
        plt.show()


class RoadGraph:
    def __init__(
        self,
        min_length: float,
        min_width: float,
        road_map: RoadMap,
        visualise: bool = False,
    ) -> None:
        """
        Class that creates grid of dots, conencts the edges between nearest vertices
        """
        self.min_length = min_length
        self.min_width = min_width
        self.road_map = road_map
        self.visualise = visualise
        self.buffer_distance = 0.1

        self.max_length = 2 * self.min_length
        self.vertices = []
        self.global_kdtree = None
        self.global_graph: nx.Graph = None
        self.full_graph: nx.Graph = None

        self.make_vertices()

    def _create_graph(
        self, points: list, min_length: float, max_length: float
    ) -> nx.Graph:
        """
        points: list of (x,y) points
        min_length: minimum edge length (L2) between 2 points
        max_length: maximum edge length between 2 points
        returns the graph of edges and vertices
        """
        self.global_kdtree = KDTree(points)
        self.global_graph = nx.Graph()
        for idx, point in enumerate(points):
            self.global_graph.add_node(idx, pos=point)

        # for every point, search for their neighbour
        for idx, point in enumerate(points):
            indices = self.global_kdtree.query_ball_point(point, max_length)
            for neighbour_idx in indices:
                if neighbour_idx == idx:
                    # avoid calling itself
                    continue
                dist = np.linalg.norm(np.array(point) - np.array(points[neighbour_idx]))
                if not min_length <= dist <= max_length:
                    # outside allowable lengths
                    continue
                candidate_edge = LineString([points[neighbour_idx], point])
                if not candidate_edge.within(self.road_map.map):
                    # motion path is disallowed
                    continue
                # TODO: decide on weights
                self.global_graph.add_edge(idx, neighbour_idx, weight=dist)
        return self.global_graph

    def make_vertices(self) -> None:
        """
        min_length: the minimum safe distance between vertices parallel to the direction of the road
        min_width: the minimum safe distance between vertices perpendicular to the direction of the road
        visualise: plots the grid on the map
        min_length should be longer than min_width
        """
        buffered_map = self.road_map.map.buffer(-self.buffer_distance)
        for road in self.road_map.roads:
            if road.direction == -1:  # x longer than y
                # x is the length and y is the width
                min_x_dist = self.min_length
                min_y_dist = self.min_width
            else:
                min_x_dist = self.min_width
                min_y_dist = self.min_length
            x_segments = np.floor(road.x_length / min_x_dist).astype(int)
            y_segments = np.floor(road.y_length / min_y_dist).astype(int)
            x_seg_points = [
                (road.x_max - road.x_min) / (x_segments * 2) * (2 * i + 1) + road.x_min
                for i in range(x_segments)
            ]
            y_seg_points = [
                (road.y_max - road.y_min) / (y_segments * 2) * (2 * i + 1) + road.y_min
                for i in range(y_segments)
            ]
            grid_points = list(itertools.product(x_seg_points, y_seg_points))
            for point in grid_points:
                if Point(point).within(buffered_map):
                    self.vertices.append(point)

        self.global_graph = self._create_graph(
            self.vertices, self.min_length, self.max_length
        )

        if self.visualise:
            self.road_map.visualise(True, True, self.global_graph)

    def get_N_nearest_vertices(
        self, point: Tuple, N: int = 1
    ) -> list[int]:
        """
        point: Queried point (x,y)
        N: number of points to return
        returns N nearest vertices to the queried point
        """
        _, idx = self.global_kdtree.query(point, N)
        return idx

    def add_edges_between_consecutive_rings(
        self,
        local_graph: nx.Graph,
        current_side: List[Tuple[float, float]],
        next_side: List[Tuple[float, float]],
        point_indices: dict[int : Tuple[float, float]],
        local_graph_bias: float,
    ) -> nx.Graph:
        """
        Add edges between points on the current side and the next side of the local graph.

        local_graph: The networkx graph representing the local graph.
        current_side: List of points on the current ring's side (top, right, bottom, or left).
        next_side: List of points on the next ring's side.
        point_indices: Dictionary mapping points to their indices in the graph.
        local_graph_bias: Bias factor for the local graph's edge weights.
        """
        for i in range(len(current_side)):  # Avoid going out of bounds
            candidate_edge = LineString([next_side[i], current_side[i]])
            if not candidate_edge.within(self.road_map.map):
                continue
            # Connect current point to two closest points in the next ring
            local_graph.add_edge(
                point_indices[current_side[i]],
                point_indices[next_side[i]],
                weight=local_graph_bias
                * np.linalg.norm(np.array(next_side[i]) - np.array(current_side[i])),
            )
        for i in range(len(current_side)):
            candidate_edge = LineString([next_side[i + 1], current_side[i]])
            if not candidate_edge.within(self.road_map.map):
                continue
            local_graph.add_edge(
                point_indices[current_side[i]],
                point_indices[next_side[i + 1]],
                weight=local_graph_bias
                * np.linalg.norm(
                    np.array(next_side[i + 1]) - np.array(current_side[i])
                ),
            )

        return local_graph

    def add_edges_between_following_rings(
        self,
        local_graph: nx.Graph,
        current_side: List[Tuple[float, float]],
        following_side: List[Tuple[float, float]],
        point_indices: dict[int : Tuple[float, float]],
        local_graph_bias: float,
    ) -> nx.Graph:
        """
        Add edges between points on the current side and the following side of the local graph.

        local_graph: The networkx graph representing the local graph.
        current_side: List of points on the current ring's side (top, right, bottom, or left).
        following_side: List of points on the following ring's side.
        point_indices: Dictionary mapping points to their indices in the graph.
        local_graph_bias: Bias factor for the local graph's edge weights.
        """
        for i in range(len(current_side)):  # Avoid going out of bounds
            candidate_edge = LineString([following_side[i], current_side[i]])
            if not candidate_edge.within(self.road_map.map):
                continue
            # Connect current point to two closest points in the next ring
            local_graph.add_edge(
                point_indices[current_side[i]],
                point_indices[following_side[i + 1]],
                weight=local_graph_bias
                * np.linalg.norm(
                    np.array(following_side[i + 1]) - np.array(current_side[i])
                ),
            )

        return local_graph

    def generate_local_graph(
        self,
        pose: Tuple[float, float, float],
        step_size: float,
        num_rings: int = 5,
        multiplier: float = 1.3,
    ) -> nx.Graph:
        """
        pose: tuple containing the x, y, yaw values of the robot
        step_size: step size of each ring in metres
        num_rings: number of rings of points surrounding the robot
        multiplier: scales the rings distance by multiplier ^ ring number

        returns: local graph generated around
        """
        # first generate all points about the origin
        # then translate and rotate all points
        local_graph = nx.Graph()
        current_index = 0
        origin = (pose[0], pose[1])
        local_graph.add_node(current_index, pos=origin)
        c_theta = np.cos(pose[2])
        s_theta = np.sin(pose[2])
        rings_of_points = []
        point_indices = {}
        current_index += 1
        local_graph_bias = 0.5
        for i in range(1, num_rings + 1):
            sides = [[], [], [], []]  # top, right, bottom, left
            side_transforms = [
                (
                    lambda i, j: (
                        (-i + 2 * j) * step_size * multiplier**i,
                        i * step_size * multiplier**i * 0.6,
                    )
                ),  # Top
                (
                    lambda i, j: (
                        i * step_size * multiplier**i,
                        (i - 2 * j) * step_size * multiplier**i * 0.6,
                    )
                ),  # Right
                (
                    lambda i, j: (
                        (i - 2 * j) * step_size * multiplier**i,
                        -i * step_size * multiplier**i * 0.6,
                    )
                ),  # Bottom
                (
                    lambda i, j: (
                        -i * step_size * multiplier**i,
                        (-i + 2 * j) * step_size * multiplier**i * 0.6,
                    )
                ),  # Left
            ]
            for j in range(i):
                for side_idx, transform in enumerate(side_transforms):
                    x_temp, y_temp = transform(i, j)

                    transformed_point = (
                        x_temp * c_theta - y_temp * s_theta + pose[0],
                        x_temp * s_theta + y_temp * c_theta + pose[1],
                    )
                    sides[side_idx].append(transformed_point)
            rings_of_points.append(sides)

            for side_points in sides:
                for point in side_points:
                    point_indices[point] = current_index
                    local_graph.add_node(
                        current_index, pos=point
                    )  # Add node to the graph with index
                    current_index += 1
        for points in rings_of_points[0]:
            # first add connections to origin
            local_graph.add_edge(
                0,
                point_indices[points[0]],
                weight=local_graph_bias * np.linalg.norm(np.array(points[0])),
            )
        if num_rings >= 2:
            for points in rings_of_points[1]:
                local_graph.add_edge(
                    0,
                    point_indices[points[1]],
                    weight=local_graph_bias * np.linalg.norm(np.array(points[1])),
                )

        sides = [x for x in range(4)]
        for ring_idx in range(len(rings_of_points) - 1):
            # Unpack the current ring and the next ring
            current_ring = rings_of_points[ring_idx]
            next_ring = rings_of_points[ring_idx + 1]
            following_ring = None
            if ring_idx < len(rings_of_points) - 2:
                following_ring = rings_of_points[ring_idx + 2]

            # Loop over each side (top, right, bottom, left)
            for side_idx in sides:
                current_side = current_ring[side_idx]
                next_side = next_ring[side_idx]

                local_graph = self.add_edges_between_consecutive_rings(
                    local_graph,
                    current_side,
                    next_side,
                    point_indices,
                    local_graph_bias,
                )
                if following_ring is not None:
                    following_side = following_ring[side_idx]
                    local_graph = self.add_edges_between_following_rings(
                        local_graph,
                        current_side,
                        following_side,
                        point_indices,
                        local_graph_bias,
                    )

            # Handle corner connections
            for i in range(4):
                candidate_edge = LineString(
                    [current_ring[i][0], next_ring[(i - 1) % 4][-1]]
                )
                if not candidate_edge.within(self.road_map.map):
                    # motion path is disallowed
                    continue
                local_graph.add_edge(
                    point_indices[
                        current_ring[i][0]
                    ],  # First point in each side of the current ring
                    point_indices[
                        next_ring[(i - 1) % 4][-1]
                    ],  # Last point in the first anticlockwise side of the next ring
                    weight=local_graph_bias
                    * np.linalg.norm(
                        np.array(current_ring[i][0])
                        - np.array(next_ring[(i - 1) % 4][-1])
                    ),
                )
            for i in range(4):
                if following_ring is not None:
                    candidate_edge = LineString(
                        [current_ring[i][0], following_ring[(i - 1) % 4][-1]]
                    )
                    if not candidate_edge.within(self.road_map.map):
                        # motion path is disallowed
                        continue
                    local_graph.add_edge(
                        point_indices[
                            current_ring[i][0]
                        ],  # First point in each side of the current ring
                        point_indices[
                            following_ring[(i - 1) % 4][-1]
                        ],  # Last point in the first anticlockwise side of the next ring
                        weight=local_graph_bias
                        * np.linalg.norm(
                            np.array(current_ring[i][0])
                            - np.array(following_ring[(i - 1) % 4][-1])
                        ),
                    )

        # pos = nx.get_node_attributes(local_graph, 'pos')
        # labels = {node: f"({round(x, 2)}, {round(y, 2)})" for node, (x, y) in pos.items()}
        # nx.draw(local_graph, pos, labels=labels, with_labels=False, node_size=10)
        # ax = plt.gca()  # Get current axes
        # for edge in local_graph.edges():
        #     # Get positions of the nodes at the ends of the edge
        #     start_pos = pos[edge[0]]
        #     end_pos = pos[edge[1]]

        #     # Create a FancyArrowPatch for the edge
        #     arrow = FancyArrowPatch(start_pos, end_pos, arrowstyle='->', color='grey', mutation_scale=30.0)
        #     # ax.add_patch(arrow)
        # plt.show()
        return local_graph

    def combine_graph(
        self,
        main_graph: nx.graph,
        local_graph: nx.graph,
        min_length: float,
        max_length: float,
    ) -> nx.graph:

        index_shifter = max(main_graph.nodes) + 1
        for local_idx, local_point in local_graph.nodes(data="pos"):
            main_graph.add_node(local_idx + index_shifter, pos=local_point)
        for edge in local_graph.edges(data=True):
            local_idx_1, local_idx_2, edge_data = edge
            main_graph.add_edge(
                local_idx_1 + index_shifter, local_idx_2 + index_shifter, **edge_data
            )
        points = {node: data["pos"] for node, data in main_graph.nodes(data=True)}
        for local_idx, local_point in local_graph.nodes(data="pos"):
            indices = self.global_kdtree.query_ball_point(local_point, max_length)
            # print(f"For {local_point}, found neighbours: {indices}")
            for neighbour_idx in indices:
                # print(f"looking at neighbour {points[neighbour_idx]}")
                dist = np.linalg.norm(
                    np.array(local_point) - np.array(points[neighbour_idx])
                )
                if not min_length <= dist <= max_length:
                    # print("Distance outside annular")
                    # outside allowable lengths
                    continue
                candidate_edge = LineString([points[neighbour_idx], local_point])
                if not candidate_edge.within(self.road_map.map):
                    # motion path is disallowed
                    continue
                # TODO: decide on weights
                main_graph.add_edge(
                    local_idx + index_shifter, neighbour_idx, weight=dist
                )
        return main_graph, index_shifter

    def loop_graph(
        self,
        poses,
        step_size: float,
        num_rings: int = 4,
        multiplier: float = 1.3,
    ):
        self.full_graph = deepcopy(self.global_graph)
        self.pose_indices = []
        for pose in poses:
            local_graph = self.generate_local_graph(
                pose, step_size, num_rings, multiplier
            )
            self.full_graph, pose_idx = self.combine_graph(
                self.full_graph, local_graph, self.min_length, self.max_length
            )
            self.pose_indices.append(pose_idx)

        # self.road_map.visualise(True, True, self.full_graph)


class RoadTrack:
    """
    Class that generates grid points based on the assumption that there are 2 tracks
    """

    def __init__(self, road_map: RoadMap):
        self.map = road_map.map
        self.roads = road_map.roads
        self.road_intersections = road_map.road_intersections_poly
        self.road_intersections_data = road_map.road_intersections
        self.vertices: List[Point] = []
        self.vertex_to_index: dict[Point, int] = {}
        self.edges: dict[Tuple[Point, Point], LineString] = {}
        self.min_seg_length = 0.4

        self.generate_grid_points()
        self.full_graph = self.generate_nx_graph()
        self.global_kdtree = KDTree([x.coords[0] for x in self.vertices])
        # road_map.visualise(True, True, self.full_graph)

    def generate_grid_points(self):
        """
        Generate grid points by:
        1. Loop through each road, and place points along 2 tracks, except intersections
        2. Loop through each intersection, puts points in a clever way
        Use shapely and a dictionary to define these points
        """
        for road in self.roads:
            self.generate_road_points(road)
        road_pts_kd_tree = KDTree([x.coords[0] for x in self.vertices])
        for road_intersection, road1, road2 in self.road_intersections_data:
            self.generate_intersection_points(
                road_intersection,
                self.roads[road1],
                self.roads[road2],
                road_pts_kd_tree,
            )

    def generate_nx_graph(self):
        graph = nx.DiGraph()
        for idx, vertex in enumerate(self.vertices):
            graph.add_node(idx, pos=(vertex.x, vertex.y))
        for (pt_A, pt_B), line in self.edges.items():
            graph.add_edge(
                self.vertex_to_index[pt_A],
                self.vertex_to_index[pt_B],
                weight=line.length,
                geometry=line,
            )
            graph.add_edge(
                self.vertex_to_index[pt_B],
                self.vertex_to_index[pt_A],
                weight=line.length,
                geometry=LineString(list(line.coords)[::-1]),
            )
        return graph

    def get_N_nearest_vertices(
        self, point: Tuple, N: int = 1
    ) -> list[Tuple[float, float]]:
        """
        point: Queried point (x,y)
        N: number of points to return
        returns N nearest vertices to the queried point
        """
        _, idx = self.global_kdtree.query(point, N)
        return idx

    def bezier_curve(
        self, start: Point, end: Point, direction: int = None, num_points=100
    ):
        if direction == -1:
            # cp1 = Point(np.array(mid_pt.coords[0]) + np.array([0, distance * 0.5]))
            # cp2 = Point(np.array(mid_pt.coords[0]) - np.array([0, distance * 0.5]))
            cp1 = Point((end.x - start.x) / 2 + start.x, start.y)
            cp2 = Point((end.x - start.x) / 2 + start.x, end.y)
        elif direction == 1:
            cp1 = Point(start.x, (end.y - start.y) / 2 + start.y)
            cp2 = Point(end.x, (end.y - start.y) / 2 + start.y)
        elif direction == 2:
            corner = Point(start.x, end.y)
            cp1 = Point((start.x + corner.x) / 2, start.y)
            cp2 = Point(corner.x, (corner.y + end.y) / 2)
        elif direction == -2:
            corner = Point(end.x, start.y)
            cp2 = Point((end.x + corner.x) / 2, end.y)
            cp1 = Point(corner.x, (corner.y + start.y) / 2)

        t_values = np.linspace(0, 1, num_points)
        x_start, y_start = start.x, start.y
        x_cp1, y_cp1 = cp1.x, cp1.y
        x_cp2, y_cp2 = cp2.x, cp2.y
        x_end, y_end = end.x, end.y

        curve_x = (
            (1 - t_values) ** 3 * x_start
            + 3 * (1 - t_values) ** 2 * t_values * x_cp1
            + 3 * (1 - t_values) * t_values**2 * x_cp2
            + t_values**3 * x_end
        )
        curve_y = (
            (1 - t_values) ** 3 * y_start
            + 3 * (1 - t_values) ** 2 * t_values * y_cp1
            + 3 * (1 - t_values) * t_values**2 * y_cp2
            + t_values**3 * y_end
        )
        curve_points = [(x, y) for x, y in zip(curve_x, curve_y)]
        return LineString(curve_points)

    def add_points_road(self, pt_A, pt_B, pt_C, pt_D, direction):
        for pt in [pt_A, pt_B, pt_C, pt_D]:
            if pt not in self.vertex_to_index:
                self.vertex_to_index[pt] = len(self.vertices)
                self.vertices.append(pt)
        self.edges[(pt_A, pt_C)] = LineString((pt_A, pt_C))
        # self.edges[(pt_C, pt_A)] = LineString((pt_C, pt_A))
        self.edges[(pt_B, pt_D)] = LineString((pt_B, pt_D))
        # self.edges[(pt_D, pt_B)] = LineString((pt_D, pt_B))
        self.edges[(pt_A, pt_D)] = self.bezier_curve(pt_A, pt_D, direction)
        # self.edges[(pt_D, pt_A)] = self.bezier_curve(pt_D, pt_A, direction)
        self.edges[(pt_B, pt_C)] = self.bezier_curve(pt_B, pt_C, direction)
        # self.edges[(pt_C, pt_B)] = self.bezier_curve(pt_C, pt_B, direction)

    def add_points_road_intersection(
        self, pt_1: Point, pt_2: Point, road_pts_kdtree: KDTree, direction: str
    ) -> None:
        for pt in [pt_1, pt_2]:
            if pt not in self.vertex_to_index:
                self.vertex_to_index[pt] = len(self.vertices)
                self.vertices.append(pt)
        pt_a = self.get_nearest_direction_neighbour(pt_1, road_pts_kdtree, direction)
        pt_b = self.get_nearest_direction_neighbour(pt_2, road_pts_kdtree, direction)
        self.edges[(pt_1, pt_a)] = LineString((pt_1, pt_a))
        self.edges[(pt_2, pt_b)] = LineString((pt_2, pt_b))
        # self.edges[(pt_a, pt_1)] = LineString((pt_a, pt_1))
        # self.edges[(pt_b, pt_2)] = LineString((pt_b, pt_2))

    def generate_road_points(self, road: RoadSegment):
        """
        Splits road into 2 tracks, and connects points based on the min_seg_length
        """
        if road.direction == -1:
            # road longer in x
            y_points = [
                road.y_length / 4 + road.y_min,
                3 * road.y_length / 4 + road.y_min,
            ]
            x_seg_length = road.x_length / np.floor(road.x_length / self.min_seg_length)
            x_points = (
                np.arange(x_seg_length / 2, road.x_length, x_seg_length, dtype=list)
                + road.x_min
            )
            for i in range(len(x_points) - 1):
                # first check if in road_intersection
                pt_A = Point(x_points[i], y_points[0])
                pt_C = Point(x_points[i + 1], y_points[0])
                for road_intersection in self.road_intersections:
                    if pt_A.within(road_intersection) or pt_C.within(road_intersection):
                        break  # since pt_B will also be in the intersection
                else:
                    pt_B = Point(x_points[i], y_points[1])
                    pt_D = Point(x_points[i + 1], y_points[1])
                    self.add_points_road(pt_A, pt_B, pt_C, pt_D, road.direction)
        else:
            # road longer in y
            x_points = [
                road.x_length / 4 + road.x_min,
                3 * road.x_length / 4 + road.x_min,
            ]
            y_seg_length = road.y_length / np.floor(road.y_length / self.min_seg_length)
            y_points = (
                np.arange(y_seg_length / 2, road.y_length, y_seg_length, dtype=list)
                + road.y_min
            )
            for i in range(len(y_points) - 1):
                # first check if in road_intersection
                pt_A = Point(x_points[0], y_points[i])
                pt_C = Point(x_points[0], y_points[i + 1])
                for road_intersection in self.road_intersections:
                    if pt_A.within(road_intersection) or pt_C.within(road_intersection):
                        break  # since pt_B will also be in the intersection
                else:
                    pt_B = Point(x_points[1], y_points[i])
                    pt_D = Point(x_points[1], y_points[i + 1])
                    self.add_points_road(pt_A, pt_B, pt_C, pt_D, road.direction)

    def get_nearest_direction_neighbour(
        self, point: Point, kd_tree: KDTree, direction: str
    ) -> Point:
        _, indices = kd_tree.query(
            point.coords[0], k=20
        )  # maximum 8 intersection points, with 8 more points beside
        if direction == "east":
            filtered_indices = [
                idx for idx in indices if self.vertices[idx].x < point.x
            ]
        elif direction == "west":
            filtered_indices = [
                idx for idx in indices if self.vertices[idx].x > point.x
            ]
        elif direction == "north":
            filtered_indices = [
                idx for idx in indices if self.vertices[idx].y > point.y
            ]
        elif direction == "south":
            filtered_indices = [
                idx for idx in indices if self.vertices[idx].y < point.y
            ]
        else:
            raise ValueError(
                "Invalid direction! Use 'north', 'south', 'east', or 'west'."
            )
        if filtered_indices:
            nearest_idx = filtered_indices[
                0
            ]  # Nearest neighbor in the specified direction
            return Point(self.vertices[nearest_idx])
        else:
            raise Exception("Did not connect intersection point properly")

    def generate_intersection_points(
        self,
        road_intersection: Polygon,
        road1: RoadSegment,
        road2: RoadSegment,
        road_pts_kdtree: KDTree,
    ) -> None:
        """
        All intersections paths are curves connecting each entry direction
        """
        # first identify how many sides lead out of the intersection
        north, west, south, east = False, False, False, False
        int_x_min, int_y_min, int_x_max, int_y_max = road_intersection.bounds
        if road1.y_max > int_y_max or road2.y_max > int_y_max:
            north = True
        if road1.x_max > int_x_max or road2.x_max > int_x_max:
            west = True
        if road1.y_min < int_y_min or road2.y_min < int_y_min:
            south = True
        if road1.x_min < int_x_min or road2.x_min < int_x_min:
            east = True
        x_1 = (int_x_max - int_x_min) / 4 + int_x_min
        x_2 = 3 * (int_x_max - int_x_min) / 4 + int_x_min
        y_1 = (int_y_max - int_y_min) / 4 + int_y_min
        y_2 = 3 * (int_y_max - int_y_min) / 4 + int_y_min
        # six cases: north,west; north,south; north,east; west,south; west,east; south,east,
        # and all their opposite directions
        pt_N1 = Point(x_1, int_y_max)
        pt_N2 = Point(x_2, int_y_max)
        pt_W1 = Point(int_x_max, y_1)
        pt_W2 = Point(int_x_max, y_2)
        pt_S1 = Point(x_1, int_y_min)
        pt_S2 = Point(x_2, int_y_min)
        pt_E1 = Point(int_x_min, y_1)
        pt_E2 = Point(int_x_min, y_2)
        # use direction = 2 for NS -> EW and -2 for EW -> NS
        if north and west:
            self.connect_intersection_points(
                [pt_N1, pt_N2], [pt_W1, pt_W2], direction=2
            )
        if north and south:
            self.connect_intersection_points(
                [pt_N1, pt_N2], [pt_S1, pt_S2], direction=1
            )
        if north and east:
            self.connect_intersection_points(
                [pt_N1, pt_N2], [pt_E1, pt_E2], direction=2
            )
        if west and south:
            self.connect_intersection_points(
                [pt_W1, pt_W2], [pt_S1, pt_S2], direction=-2
            )
        if west and east:
            self.connect_intersection_points(
                [pt_W1, pt_W2], [pt_E1, pt_E2], direction=-1
            )
        if south and east:
            self.connect_intersection_points(
                [pt_S1, pt_S2], [pt_E1, pt_E2], direction=2
            )
        if north:
            self.add_points_road_intersection(pt_N1, pt_N2, road_pts_kdtree, "north")
        if west:
            self.add_points_road_intersection(pt_W1, pt_W2, road_pts_kdtree, "west")
        if south:
            self.add_points_road_intersection(pt_S1, pt_S2, road_pts_kdtree, "south")
        if east:
            self.add_points_road_intersection(pt_E1, pt_E2, road_pts_kdtree, "east")

    def connect_intersection_points(self, side_1, side_2, direction):
        if direction == 1 or direction == -1:
            self.edges[(side_1[0], side_2[0])] = LineString((side_1[0], side_2[0]))
            self.edges[(side_2[0], side_1[0])] = LineString((side_2[0], side_1[0]))
            self.edges[(side_1[1], side_2[1])] = LineString((side_1[1], side_2[1]))
            self.edges[(side_2[1], side_1[1])] = LineString((side_2[1], side_1[1]))
        else:
            self.edges[(side_1[0], side_2[0])] = self.bezier_curve(
                side_1[0], side_2[0], direction
            )
            # self.edges[(side_2[0], side_1[0])] = self.bezier_curve(
            #     side_2[0], side_1[0], -direction
            # )
            self.edges[(side_1[0], side_2[1])] = self.bezier_curve(
                side_1[0], side_2[1], direction
            )
            # self.edges[(side_2[1], side_1[0])] = self.bezier_curve(
            #     side_2[1], side_1[0], -direction
            # )
            self.edges[(side_1[1], side_2[1])] = self.bezier_curve(
                side_1[1], side_2[1], direction
            )
            # self.edges[(side_2[1], side_1[1])] = self.bezier_curve(
            #     side_2[1], side_1[1], -direction
            # )
            self.edges[(side_1[1], side_2[0])] = self.bezier_curve(
                side_1[1], side_2[0], direction
            )
            # self.edges[(side_2[0], side_1[1])] = self.bezier_curve(
            #     side_2[0], side_1[1], -direction
            # )

    def visualise(self):
        _, ax = plt.subplots()
        exterior_polygon, interior_polygons = RoadMap.shapely_to_mplpolygons(self.map)
        ax.add_patch(exterior_polygon)
        for hole_polygon in interior_polygons:
            ax.add_patch(hole_polygon)

        for (start, end), line in self.edges.items():
            x, y = line.xy
            plt.plot(x, y, "b-", label="Edge")
            # plt.plot(start.x, start.y, 'ro')
            # plt.plot(end.x, end.y, 'go')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Edges Visualization")
        ax.axis("scaled")
        plt.show()
