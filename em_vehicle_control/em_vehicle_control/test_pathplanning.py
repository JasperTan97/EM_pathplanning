from helper_classes.map import RoadMap, RoadSegment, RoadGraph, RoadTrack
from helper_classes.path_planner import PathPlanner
from helper_classes.vehicles import EdyMobile, Edison
from helper_classes.pathplanners.rrt import *
from helper_classes.pathplanners.pp_viz import RoadMapAnimator
from shapely import within
import time
import networkx as nx

test_roads = [
    RoadSegment((2.87, 1.67), (3.52, 13.67)),
    RoadSegment((6.9, 0.3), (7.55, 15.67)),
    RoadSegment((4.72, 1.67), (5.7, 8.64)),
    RoadSegment((2.87, 8), (7.55, 8.64)),
    RoadSegment((2.87, 1.67), (7.55, 2.32)),
]
# test_roads = [
#     RoadSegment((2.87,1.67), (3.52,4.67)),
# ]
test_map = RoadMap(test_roads)
# vehicle = EdyMobile(start_position=(3.218,3,np.pi/2))
# goal = (3.218,3.3,-np.pi/2)
# goal_radius = 0.2
# rrt = RRT_star_Reeds_Shepp((3.218,3,np.pi/2), goal, goal_radius, vehicle, test_map.map,
#             max_iter=100, search_until_max_iter=False, visualise=True)
test_graph = RoadTrack(test_map)
test_pp = PathPlanner(test_map, test_graph, 3)
# test_graph.visualise()
test_rrt_start_pose = (3.145, 1.875, -3 * np.pi / 4)
test_rrt_end_pose = (5.0, 2, np.pi / 2)
test_ca_pose = (3.58, 1.832, 0)
test_goal = (5.204, 1.944, np.pi / 2)
test_pp.plan(
    [test_rrt_start_pose, test_rrt_end_pose, test_ca_pose],
    [test_goal, test_goal, test_goal],
    [EdyMobile(), EdyMobile(), EdyMobile()]
)


# test_global_plannar.Astar((3.218,3,np.pi/2),(5.14,7.49, 0), test_map)
# test_global_plannar.CAstar([(3.218,3,np.pi/2),(5.14,7.49, 0),(7.28,15, 0), (7.28,0.97, 0)], [(5.14,7.49, 0),(3.218,3,np.pi/2),(7.28,0.97, 0), (7.28,15, 0)], [Edison(), Edison(),Edison(), Edison()])
