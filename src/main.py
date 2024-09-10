from shapely.geometry import Polygon, MultiPolygon, LineString
from shapely.affinity import translate
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from helper_classes.map import Map, RoadMap
from utilities import *
from helper_classes.vehicles import EdyMobile

test_mobile = EdyMobile((0,0,np.pi/6))._vehicle_model
travel_path = [(0,0), (5*np.sqrt(3), 5)]
print(np.arctan2(5, 5 * np.sqrt(3)), np.pi/6)

def create_swept_polygon(line_coords, sweep_polygon):
    """
    Create a polygon by sweeping a smaller polygon along a line.

    Parameters:
    line_coords (list of tuples): Coordinates of the line [(x1, y1), (x2, y2), ...]
    sweep_polygon (shapely.geometry.Polygon): The polygon to sweep along the line

    Returns:
    shapely.geometry.Polygon: The resulting swept polygon
    """
    line = LineString(line_coords)
    swept_area = []
    
    for i in range(len(line_coords) - 1):
        start = line_coords[i]
        end = line_coords[i + 1]
        segment = LineString([start, end])

        moved_polygon = translate(sweep_polygon, xoff=end[0], yoff=end[1])
    
    # Combine all the translated polygons into a single polygon
    combined_swept_polygon = MultiPolygon([sweep_polygon, moved_polygon]).convex_hull
    return combined_swept_polygon

def plot_polygon(polygon):
    fig, ax = plt.subplots()
    mpl_polygon, _ = RoadMap.shapely_to_mplpolygons(polygon)
    ax.add_patch(mpl_polygon)
    ax.axis('scaled')
    plt.show()
    
resulting_polygon = create_swept_polygon(travel_path, test_mobile)
plot_polygon(resulting_polygon)