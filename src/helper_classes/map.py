import shapely
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import CirclePolygon as MplCirclePolygon
import matplotlib.patches as patches
from typing import Tuple, Optional, List
import numpy as np
import itertools
import networkx as nx
from scipy.spatial import KDTree


class RoadSegment():
    """A road segment describes the full rectangular part of the road"""
    # Note: This class only works for axis_aligned roads

    def __init__(self, point_A: Tuple[float, float], point_B: Tuple[float, float], invert_direction: bool = False):
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
        self.road_segment_polygon = Polygon(
            (cornerBL, cornerBR, cornerTR, cornerTL))
        if not self.road_segment_polygon.is_valid:
            print(
                f"Error: The outer boundary given is not valid for road {point_A}, {point_B}")

        if self.x_length > self.y_length:
            # if the length along the x-axis is longer than the length along the y-axis
            # i.e., the road is going from left to right
            self.direction = -1
        else:
            self.direction = 1  # the road is going up to down

        if invert_direction:
            self.direction *= -1


class RoadMap():
    """
    The road map is the full map consisting of parallel and perpendicular road segments

    """

    def __init__(self, roads: list[RoadSegment], debug: bool = False) -> None:
        self.roads = roads
        self.debug = debug
        self.road_intersections = self.get_intersections()
        self.stations = []  # TODO should this be a dictionary with names?
        self.map = shapely.unary_union(
            [road.road_segment_polygon for road in roads])

    def get_intersections(self) -> list[Tuple[Polygon, RoadSegment, RoadSegment]]:
        """
        Function that iteratively looks through all roads and identify intersection points
        Returns a list of tuples that include the Polygon of the intersection 
        and the indices of the 2 roads that creates the intersection
        """
        road_intersections = []
        for i in range(len(self.roads)):
            for j in range(i+1, len(self.roads)):
                if shapely.intersects(self.roads[i].road_segment_polygon, self.roads[j].road_segment_polygon):
                    road_intersection = shapely.intersection(
                        self.roads[i].road_segment_polygon, self.roads[j].road_segment_polygon)
                    if self.debug:
                        print(
                            f"Intersection ({road_intersection}) created by roads ({i}, {self.roads[i]}) and ({j}, {self.roads[j]})")
                    road_intersections.append((road_intersection, i, j))
        return road_intersections

    def add_station(self, location: Tuple[float, float], orientation: float, radius: Optional[float] = 0.05) -> Tuple[Polygon, float]:
        """
        Method to add stations, signfying the end goal of the robot
        location: (x, y) coordinates of the station
        orientation: final orientation of the vehicle
        radius: radius describing the tolerance of the end point (default at 5cm radius)
        """
        station = Point(location)
        station.buffer(radius, resolution=100)
        self.stations.append((station, location, orientation, radius))

    @staticmethod
    def shapely_to_mplpolygons(shapely_polygon: Polygon, colour: str = 'lightblue') -> Tuple[MplPolygon, MplPolygon]:
        x, y = shapely_polygon.exterior.xy
        exterior_polygon = MplPolygon(
            list(zip(x, y)), closed=True, edgecolor='black', facecolor=colour, alpha=0.5)
        hole_polygons = []
        for hole in shapely_polygon.interiors:
            x, y = hole.xy
            hole_polygons.append(MplPolygon(
                list(zip(x, y)), closed=True, edgecolor='black', facecolor='white', alpha=1))
        return exterior_polygon, hole_polygons

    def visualise(self, show_intersections: bool = False, show_stations: bool = True, graph: Optional[nx.Graph] = None, path: Optional[List] = None) -> None:
        """Visualises the built map"""
        _, ax = plt.subplots()
        exterior_polygon, interior_polygons = self.shapely_to_mplpolygons(
            self.map)
        ax.add_patch(exterior_polygon)
        for hole_polygon in interior_polygons:
            ax.add_patch(hole_polygon)
        if show_intersections:
            for road_intersection in self.road_intersections:
                rd_int_polygon, _ = self.shapely_to_mplpolygons(
                    road_intersection[0], 'purple')
                ax.add_patch(rd_int_polygon)
        if show_stations:
            for station in self.stations:
                station_polygon = MplCirclePolygon(
                    station[1], radius=station[3], resolution=100, facecolor='green', alpha=0.5)
                ax.add_patch(station_polygon)
                # print(station[1][0], station[1][1], station[1][0]+station[3]*np.cos(station[2]), station[1][1]+station[3]*np.sin(station[2]))
                ax.arrow(station[1][0], station[1][1], station[3]*np.cos(station[2]),
                         station[3]*np.sin(station[2]), width=station[3]/10)
        if graph is not None:
            pos = nx.get_node_attributes(graph, 'pos')
            nx.draw(graph, pos, node_size=5, node_color="skyblue",
                    font_weight="bold", edge_color="gray")
            if path is not None:
                path_edges = list(zip(path,path[1:]))
                nx.draw_networkx_nodes(graph, pos, nodelist=path, node_size=7, node_color="green")
                nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color="green")
        ax.axis('scaled')
        plt.show()


class RoadGraph():
    def __init__(self, road_map: RoadMap) -> None:
        """
        Class that creates grid of dots, conencts the edges between nearest vertices
        """
        self.road_map = road_map
        self.vertices = []
        self.kdtree = None
        self.road_graph: nx.Graph = None

    def _create_graph(self, points: list, min_length: float, max_length: float) -> nx.Graph:
        """
        points: list of (x,y) points
        min_length: minimum edge length (L2) between 2 points
        max_length: maximum edge length between 2 points
        returns the graph of edges and vertices
        """
        self.kdtree = KDTree(points)
        self.road_graph = nx.Graph()
        for idx, point in enumerate(points):
            self.road_graph.add_node(idx, pos=point)

        # for every point, search for their neighbour
        for idx, point in enumerate(points):
            indices = self.kdtree.query_ball_point(point, max_length)
            for neighbour_idx in indices:
                if neighbour_idx == idx:  
                    # avoid calling itself
                    continue
                dist = np.linalg.norm(
                    np.array(point) - np.array(points[neighbour_idx]))
                if not min_length <= dist <= max_length:
                    # outside allowable lengths
                    continue
                candidate_edge = LineString([points[neighbour_idx], point])
                if not candidate_edge.within(self.road_map.map):
                    # motion path is disallowed
                    continue                
                # TODO: decide on weights
                self.road_graph.add_edge(idx, neighbour_idx, weight=dist)
        return self.road_graph

    def make_vertices(self, min_length: float, min_width: float, visualise: bool = False) -> None:
        """
        min_length: the minimum safe distance between vertices parallel to the direction of the road
        min_width: the minimum safe distance between vertices perpendicular to the direction of the road
        visualise: plots the grid on the map
        min_length should be longer than min_width
        """
        for road in self.road_map.roads:
            if road.direction == -1:  # x longer than y
                # x is the length and y is the width
                min_x_dist = min_length
                min_y_dist = min_width
            else:
                min_x_dist = min_width
                min_y_dist = min_length
            x_segments = np.floor(road.x_length / min_x_dist).astype(int)
            y_segments = np.floor(road.y_length / min_y_dist).astype(int)
            x_seg_points = [(road.x_max - road.x_min)/(x_segments*2)
                            * (2*i+1)+road.x_min for i in range(x_segments)]
            y_seg_points = [(road.y_max - road.y_min)/(y_segments*2)
                            * (2*i+1)+road.y_min for i in range(y_segments)]
            grid_points = list(itertools.product(x_seg_points, y_seg_points))
            self.vertices += grid_points

        self.road_graph = self._create_graph(
            self.vertices, min_length, 2*min_length)

        if visualise:
            self.road_map.visualise(True, True, self.road_graph)

    def get_N_nearest_vertices(self, point: Tuple, N: int = 1) -> list[Tuple[float, float]]:
        """
        point: Queried point (x,y)
        N: number of points to return
        returns N nearest vertices to the queried point
        """
        _, idx = self.kdtree.query(point, N)
        return idx


class Map():
    """All base functions for a map go here"""

    def __init__(self):
        self._map = None
        self._outer_vertices = None
        self._holes = []

    def create_outer_map(self, outer_vertices: list, visualise: bool = False) -> bool:
        """Records the outer boundary of the map, and checks if it is feasible"""
        polygon = Polygon(outer_vertices)
        if not polygon.is_valid:
            print("Error: The outer boundary given is not valid")
            return False
        if visualise:
            _, ax = plt.subplots()
            outer_polygon = MplPolygon(
                outer_vertices, closed=True, fill=True, edgecolor='black')
            ax.add_patch(outer_polygon)
            ax.axis('scaled')
            plt.show()
        self._outer_vertices = outer_vertices
        return True

    def add_hole(self, hole_vertices: list) -> bool:
        """Tests if a hole is a feasible choice"""
        if self._outer_vertices == None:
            print("Error: The outer boundary has not yet been set")
            return False
        try:
            holeA = Polygon(hole_vertices)
            if not holeA.is_valid:
                print(f"Error: The hole {hole_vertices} given is not valid")
                return False
            for hole in self._holes:
                holeB = Polygon(hole)
                if shapely.overlaps(holeA, holeB):
                    print(
                        f"Error: Candidate hole {hole_vertices} overlaps with {hole}.")
                    return False
            if not holeA.within(Polygon(self._outer_vertices)):
                print(
                    f"Error: Candidate hole {hole_vertices} lies outside of defined outer vertices")
            self._holes.append(hole_vertices)
            return True
        except Exception as e:
            print(f"Error adding hole {hole_vertices}: {e}")
            return False

    def add_holes(self, list_of_hole_vertices: list) -> bool:
        """Tests a list of holes if they are feasible choices"""
        try:
            for hole_vertices in list_of_hole_vertices:
                self.add_hole(hole_vertices)
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def reset_holes(self) -> None:
        """Resets list of holes"""
        self._holes = []

    def build(self) -> bool:
        """Rebuilds the map"""
        try:
            self._map = Polygon(shell=self._outer_vertices, holes=self._holes)
            return True
        except Exception as e:
            print(f"Error: {e}")

    def shapely_to_mplpolygons(self, shapely_polygon: Polygon) -> Tuple[MplPolygon, MplPolygon]:
        x, y = shapely_polygon.exterior.xy
        exterior_polygon = MplPolygon(list(
            zip(x, y)), closed=True, edgecolor='black', facecolor='lightblue', alpha=0.5)
        hole_polygons = []
        for hole in shapely_polygon.interiors:
            x, y = hole.xy
            hole_polygons.append(MplPolygon(
                list(zip(x, y)), closed=True, edgecolor='black', facecolor='white', alpha=1))
        return exterior_polygon, hole_polygons

    def visualise(self) -> None:
        """Visualises the built map"""
        if self._map == None:
            print("Error: Map has not been built")
            return
        _, ax = plt.subplots()
        exterior_polygon, hole_polygons = self.shapely_to_mplpolygons(
            self._map)
        ax.add_patch(exterior_polygon)
        for hole_polygon in hole_polygons:
            ax.add_patch(hole_polygon)
        ax.axis('scaled')
        plt.show()
