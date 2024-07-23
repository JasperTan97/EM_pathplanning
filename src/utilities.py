import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

def png_to_shapely(image_path: str, resolution: float, show_plot = False) -> Polygon:
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_np = np.asarray(image)
    _, binary_image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0,255,0), 3)

    # Function to convert OpenCV contour to Shapely Polygon
    def contour_to_shapely(contour, scale=1.0):
        return Polygon([(point[0][0] * scale, point[0][1] * scale) for point in contour])

    # List to store shapely polygons
    polygons = []

    # Iterate over contours
    for i, contour in enumerate(contours):
        # If contour is external, create a polygon
        if cv2.contourArea(contour) > 0:
            poly = contour_to_shapely(contour, scale=resolution)
            polygons.append(poly)

    # Function to apply symmetric difference to a list of polygons
    def symmetric_difference(polygons):
        result = polygons[0]
        for poly in polygons[1:]:
            result = result.symmetric_difference(poly)
        return result

    # Apply symmetric difference to all polygons
    multi_polygon = symmetric_difference(polygons)
    x, y = multi_polygon.exterior.xy
    external_points = list(zip(x, y))
    print(external_points)

    holes_points = []
    for interior in multi_polygon.interiors:
        x, y = interior.xy
        internal_points = list(zip(x, y))
        print(internal_points)
        holes_points.append(internal_points)

    if show_plot:
        def shapely_to_mplpolygon(shapely_polygon, **kwargs):
            patches = []
            # Exterior
            x, y = shapely_polygon.exterior.xy
            patches.append(MplPolygon(list(zip(x, y)), **kwargs))
            # Holes
            for interior in shapely_polygon.interiors:
                x, y = interior.xy
                patches.append(MplPolygon(list(zip(x, y)), closed=True, fill=True, edgecolor='black', facecolor='white'))
            return patches

        # Plotting the result
        fig, ax = plt.subplots(figsize=(10, 8))

        # Convert and plot the MultiPolygon
        if isinstance(multi_polygon, MultiPolygon):
            for poly in multi_polygon:
                patches = shapely_to_mplpolygon(poly, closed=True, edgecolor='black', facecolor='lightblue', alpha=0.5)
                for patch in patches:
                    ax.add_patch(patch)
        elif isinstance(multi_polygon, Polygon):
            patches = shapely_to_mplpolygon(multi_polygon, closed=True, edgecolor='black', facecolor='lightblue', alpha=0.5)
            for patch in patches:
                ax.add_patch(patch)

        ax.set_xlim(0, image.shape[1] * resolution)
        ax.set_ylim(0, image.shape[0] * resolution)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    return multi_polygon, external_points, holes_points