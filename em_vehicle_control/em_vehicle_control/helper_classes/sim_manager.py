import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

from .map import Map
from .vehicles import *

class SimManager():
    def __init__(self, map: Map, vehicles: list, time_step: float = 0.01) -> None:
        self._map = map
        self._vehicles = vehicles
        self._dt = time_step
    
    def _shapely_to_mplpolygon(self, shapely_polygon, **kwargs):
        """Helper function to return exterior points"""
        x, y = shapely_polygon.exterior.xy
        return MplPolygon(list(zip(x, y)), **kwargs)
    
    def visualise(self) -> None:
        """Visualises the map and vehicles on the map"""
        patches = []
        patches.append(self._shapely_to_mplpolygon(self._map._map, closed=True, edgecolor='black', facecolor='lightblue', alpha=0.5))
        for interior in self._map._map.interiors:
            x, y = interior.xy
            patches.append(MplPolygon(list(zip(x, y)), closed=True, fill=True, edgecolor='black', facecolor='white'))
        for vehicle in self._vehicles:
            patches.append(self._shapely_to_mplpolygon(vehicle._vehicle_model, closed=True, edgecolor='black', facecolor='red', alpha=0.5))

        _, ax = plt.subplots()
        p = PatchCollection(patches, match_original=True)
        ax.add_collection(p)
        ax.axis('scaled')
        plt.show()

    def check_collisions(self) -> bool:
        """
        Checks if all vehicles are contained entirely within the map
        and if all vehicles are not in collision with each other
        returns True if there are NO collision
        """
        no_collisions = True
        for vehicle in self._vehicles:
            if not vehicle._vehicle_model.within(self._map._map):
                no_collisions = False
                print(f"COLLISION: Vehicle at ({vehicle._x}, {vehicle._y}) is outside the map")
        if len(self._vehicles) < 2:
            return no_collisions
        for i in range(len(self._vehicles)):
            for j in range(i+1, len(self._vehicles)):
                if self._vehicles[i]._vehicle_model.intersects(self._vehicles[j]._vehicle_model):
                    no_collisions = False
                    print(f"COLLISION: Vehicle at ({self._vehicles[i]._x}, {self._vehicles[i]._y}) has collided with vehicle at ({self._vehicles[j]._x}, {self._vehicles[j]._y})")
        return no_collisions
    
    def step(self, commands: list) -> None:
        """
        Steps through the simulation
        Input is a list of commands of left and right accelerations
        """
        if len(commands) != len(self._vehicles):
            print("Error: There are not enough acceleration commands.")
            return
        for vehicle, command in zip(self._vehicles, commands):
            vehicle.step(command)