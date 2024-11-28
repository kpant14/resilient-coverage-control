import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point

def voronoicentroid(x, Q):
    # Compute bounds of the polygon
    Qb = Q.bounds
    Ql = Q.length

    # Create a larger frame around the polygon
    frame = np.array([[Qb[0] - Ql * 10, Qb[1] - Ql * 10],
                    [Qb[2] + Ql * 10, Qb[1] - Ql * 10],
                    [Qb[0] - Ql * 10, Qb[3] + Ql * 10],
                    [Qb[2] + Ql * 10, Qb[3] + Ql * 10]])

    # Append the frame to the original points
    xframe = np.append(x, frame, axis=0)

    # Compute the Voronoi diagram
    vor = Voronoi(xframe)
    # Computation of Voronoi centroids
    vcentroids = np.zeros_like(x)  # Initialize array for centroids

    for i in range(len(x)):
        # Get the vertices of the Voronoi region for the i-th point
        poly = [vor.vertices[v] for v in vor.regions[vor.point_region[i]] if v != -1]
        # Create a Polygon from the vertices
        if len(poly) > 0:
            i_cell = Q.intersection(Polygon(poly))
            if not i_cell.is_empty:
                vcentroids[i] = i_cell.centroid.coords[0]
            else:
                vcentroids[i] = np.nan  # Assign NaN if the intersection is empty
    return vcentroids, vor