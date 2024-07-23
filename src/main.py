from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from helper_classes.map import Map
from utilities import *

png_map, a, b = png_to_shapely(image_path = 'data/circuit_mr.png', resolution=0.005, show_plot=False)
test_map = Map()
print(png_map)
test_map._map = png_map
test_map.visualise()