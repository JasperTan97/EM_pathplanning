from helper_classes.map import RoadMap, RoadSegment, RoadGraph
from helper_classes.global_plannar import GlobalPlannar
from helper_classes.vehicles import EdyMobile
from helper_classes.pathplanners.rrt import *
from helper_classes.pathplanners.pp_viz import RoadMapAnimator
from shapely import within

goal = (6.2, 8.5)
goal_radius = 0.2
vehicle = EdyMobile(start_position=[3.2, 2, (np.pi* 2) * (0)])
test_roads = [
    RoadSegment((2.87,1.67), (3.52,13.67)),
    RoadSegment((6.9,0.3), (7.55, 15.67)),
    RoadSegment((4.72, 1.67), (5.7, 8.64)),
    RoadSegment((2.87, 8), (7.55,8.64)),
    RoadSegment((2.87, 1.67), (7.55, 2.32))
    # RoadSegment((0,0), (10,10))
]
test_map = RoadMap(test_roads)
local_map = test_map
other_vehicles = []
test_graph = RoadGraph(test_map)
test_graph.make_vertices(0.4, 0.3, False)

test_global_plannar = GlobalPlannar(test_graph)
# test_global_plannar.dijkstra((2,2), (7.2,15),test_map)
# test_global_plannar.Astar((2,2), (7.2,15), test_map)
test_global_plannar.CAstar([(2,2),(6,8.6),(2,4),(7.2,15)], [(7.2,15),(2,4),(6,8.6),(2,2)], [EdyMobile(),EdyMobile(),EdyMobile(),EdyMobile()], test_map)


# a = within(vehicle._vehicle_model, local_map.map)
# print(a)
# rrt = RRT((vehicle._x, vehicle._y), goal, goal_radius, vehicle, local_map.map, 
#                max_iter=2000, visualise=True)
# rrt = RRT_star((vehicle._x, vehicle._y), goal, goal_radius, vehicle, local_map.map, 
#                max_iter=2000, search_until_max_iter=True, visualise=True)
# print(rrt.check_collision_point((3.2, 2, 0)))
# print(rrt.check_collision_point((4.2, 4, 0)))
# print(rrt.check_collision_point((5, 16, 0)))
# rrt.plan()