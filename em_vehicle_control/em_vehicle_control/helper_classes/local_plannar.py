import numpy as np
from typing import List, Tuple

from .map import RoadMap, RoadGraph
from .vehicles import EdyMobile
from .global_plannar import GlobalPlannar

class LocalPlannar():
    """
    The local plannar will (probably) run a mutli agent random sampling algorithm 
    on a region surrounding the current location of the vehicle.
    The start point is the current position of the vehicle and 
    the end point is a node at a certain distance L defined by the global plannar
    Other vehicles as dynamic obstacles are handled by this plannar
    """
    def __init__(self, 
                 road_map: RoadMap, 
                 vehicle_list: List, 
                 radius_of_interest: float) -> None:
        """
        road_map: contains shapely map of road
        vehicle_list: all vehicles on the map
        radius_of_interest: radius of region around vehicle used for local planning
        """
        self.road_map = road_map
        self.vehicle_list = vehicle_list
        self.radius_of_interest = radius_of_interest
    
    def isolate_surrounding_region(self, vehicle):
        """
        Constructs a map of the region surrounding the vehicle, including other vehicles
        vehicle: selected vehicle to isolate around
        """
        