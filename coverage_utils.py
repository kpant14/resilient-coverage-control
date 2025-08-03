import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, cKDTree

def calculate_locational_opt_cost(agents, x_range, y_range, domain_size):
    # Define density function phi(q)
    def phi(q):
        # Example: higher weight near center
        return np.ones(q.shape[0])

    # Create grid over domain Q = [0,1] x [0,1]
    x = np.linspace(x_range[0], x_range[1], domain_size)
    y = np.linspace(y_range[0], y_range[1], domain_size)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

    # Compute Voronoi assignment using KDTree
    tree = cKDTree(agents)
    _, indices = tree.query(grid_points)

    # Evaluate phi at each grid point
    phi_values = phi(grid_points)
    # Compute cost H(p)
    cost = 0.0
    for i in range(agents.shape[0]):
        pts = grid_points[indices == i]
        phis = phi_values[indices == i]
        distances = np.linalg.norm(pts - agents[i], axis=1)**2
        cost += np.sum(distances * phis)

    # Normalize cost by area element
    delta_q = (1.0 / domain_size) ** 2
    H = cost * delta_q
    return H


def main(args=None):
    # Parameters
    n_agents = 5
    domain_size = 100  # number of grid points per axis
    x_range = [0, 1]
    y_range = [0, 1]

    # Define agent positions randomly
    agents = np.random.rand(n_agents, 2)
    H = calculate_locational_opt_cost(agents, x_range, y_range, domain_size)
    print(f"Locational optimization cost H(p): {H:.6f}")

    plt.figure(figsize=(6,6))
    vor = Voronoi(agents)
    voronoi_plot_2d(vor, show_vertices=False, line_colors='black', show_points=True)
    plt.scatter(agents[:, 0], agents[:, 1], color='red', label='Agents')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Voronoi Cells of Agents")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()