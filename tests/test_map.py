import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from helper_classes.map import RoadMap, RoadSegment, RoadGraph
from helper_classes.global_plannar import GlobalPlannar

test_roads = [
    RoadSegment((2.87,1.67), (3.52,13.67)),
    RoadSegment((6.9,0.3), (7.55, 15.67)),
    RoadSegment((4.72, 1.67), (5.7, 8.64)),
    RoadSegment((2.87, 8), (7.55,8.64)),
    RoadSegment((2.87, 1.67), (7.55, 2.32))
]
test_map = RoadMap(test_roads)
test_map.add_station((7.2,15), -1.57, 0.2)
# test_map.visualise(show_intersections=True)

test_graph = RoadGraph(test_map)
test_graph.make_vertices(0.4, 0.3, False)

test_global_plannar = GlobalPlannar(test_graph)
test_global_plannar.dijkstra((2,2), (7.2,15),test_map)

"""

test_map = Map()
outer_boundary = [(0, 0), (2, 0), (2, 3), (11, 3), (11, 8), (0, 8)]

##########
# Test 1 #
##########
test_map.create_outer_map(outer_boundary, False)

##########
# Test 2 #
##########
holes = [
    [(2, 5), (5, 5), (5, 6), (2, 6)],
    [(7, 5), (9, 5), (9, 6), (7, 6)]
]
test_map.add_holes(holes)
test_map.build()
test_map.visualise()
"""