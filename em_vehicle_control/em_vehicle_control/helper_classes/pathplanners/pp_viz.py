import shapely
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.patches import CirclePolygon as MplCirclePolygon
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import ArtistAnimation
import networkx as nx
import numpy as np
from typing import Optional

from ..map import RoadMap, RoadGraph

class RoadMapAnimator():
    """
    For animating path planning
    """
    def __init__(self, road_map: RoadMap, road_graph: RoadGraph, save_path: Optional[str]= None) -> None:
        self.road_map = road_map
        self.road_graph = road_graph
        self.save_path = save_path
        self.list_of_colours = ['green', 'red', 'blue', 'purple', 'yellow', 'black', 'white']

    def animate_multi_paths(self, paths):
        # set up static plots
        fig, ax = plt.subplots()
        exterior_polygon, interior_polygons = RoadMap.shapely_to_mplpolygons(self.road_map.map)
        ax.add_patch(exterior_polygon)
        for hole_polygon in interior_polygons:
            ax.add_patch(hole_polygon)

        pos = nx.get_node_attributes(self.road_graph.full_graph, 'pos')
        nx.draw(self.road_graph.full_graph, pos, node_size=5, node_color="skyblue", font_weight="bold", edge_color="gray", ax=ax)
        
        # loop through all paths, and update paths using start times
        frames = []
        unopened_paths = paths
        plotted_paths = [[] for _ in range(len(paths))]
        while not all(len(sublist) == 0 for sublist in unopened_paths):
            next_nodes = [x[0][1] if x else np.inf for x in unopened_paths]
            next_node_path_index = np.argmin(next_nodes)
            next_node = unopened_paths[next_node_path_index].pop(0)[0]
            plotted_paths[next_node_path_index].append(next_node)
            frame = []
            for i, plotted_path in enumerate(plotted_paths):
                path_edges = list(zip(plotted_path, plotted_path[1:]))
                if len(plotted_path) > 1:
                    frame.append(nx.draw_networkx_nodes(self.road_graph.full_graph, pos, nodelist=plotted_path, node_size=7, node_color=self.list_of_colours[i], ax=ax))
                if len(path_edges) > 1:
                    frame.append(nx.draw_networkx_edges(self.road_graph.full_graph, pos, edgelist=path_edges, edge_color=self.list_of_colours[i], ax=ax))
            frames.append(frame)
        
        ani = ArtistAnimation(fig, frames, interval=300, blit=True)
        ax.axis('scaled')
        if self.save_path:
            ani.save(self.save_path, writer='imagemagick', fps=2)

        plt.show()

    def animate_single_path(self, path):
        fig, ax = plt.subplots()
        exterior_polygon, interior_polygons = RoadMap.shapely_to_mplpolygons(self.road_map.map)
        ax.add_patch(exterior_polygon)
        for hole_polygon in interior_polygons:
            ax.add_patch(hole_polygon)

        pos = nx.get_node_attributes(self.road_graph.road_graph, 'pos')
        nx.draw(self.road_graph.road_graph, pos, node_size=5, node_color="skyblue", font_weight="bold", edge_color="gray", ax=ax)

        frames = []
        path_edges = list(zip(path, path[1:]))
        for i in range(len(path)):
            frame = []
            frame.append(nx.draw_networkx_nodes(self.road_graph.road_graph, pos, nodelist=path[:i+1], node_size=7, node_color="green", ax=ax))
            if i > 0:
                frame.append(nx.draw_networkx_edges(self.road_graph.road_graph, pos, edgelist=path_edges[:i], edge_color="green", ax=ax))
            frames.append(frame)
        
        ani = ArtistAnimation(plt.gcf(), frames, interval=300, blit=True)
        ax.axis('scaled')
        plt.show()