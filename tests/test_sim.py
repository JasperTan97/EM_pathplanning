import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
from helper_classes.map import Map
from helper_classes.vehicles import EdyMobile
from helper_classes.sim_manager import SimManager

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

test_edy1 = EdyMobile(start_position=[1, 2, (np.pi* 2) * (3/5)])
test_edy2 = EdyMobile(start_position=[6, 3, (np.pi* 2) * (0/5)])
test_edy3 = EdyMobile(start_position=[8, 4, (np.pi* 2) * (1/5)])
test_edy4 = EdyMobile(start_position=[8.5, 3.9, (np.pi* 2) * (4/5)])

sim = SimManager(test_map, [test_edy1, test_edy2, test_edy3, test_edy4], 0.01)
sim.check_collisions()
sim.visualise()
