import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

def create_adjacency_matrix(edges):
    """Creates an adjacency matrix for a given set of edges.

    Args:
        edges: A list of tuples, where each tuple represents an edge (node1, node2).

    Returns:
        A 2D list representing the adjacency matrix.
    """
    num_nodes = max(max(edge) for edge in edges) + 1
    adj_matrix = [[0] * num_nodes for _ in range(num_nodes)]

    for edge in edges:
        node1, node2 = edge
        adj_matrix[node1][node2] = 1
        adj_matrix[node2][node1] = 1  
    # For undirected graphs

    return np.array(adj_matrix)

def get_neighbors_state(agent_id, agent_states, adj_matrix):
    neighbor_states = []
    for j in range(adj_matrix.shape[1]):
        if (adj_matrix[agent_id][j]==1):
            neighbor_states.append(agent_states[j])
    return np.array(neighbor_states)


def get_neighbors_bearing(state, nb_states):
    nb_bearing = []
    for j in range(nb_states.shape[0]):
        nb_bearing.append(get_bearing(state, nb_states[j]))
    return np.array(nb_bearing)

def get_bearing(state_i, state_j):
    return (state_j[:2] - state_i[:2])/np.sqrt((state_j[0] - state_i[0])**2 + (state_j[1] - state_i[1])**2)

def find_two_closest_neighbors(agent_pos, all_positions):
    """
    Find indices of the two closest neighbors to agent_pos in all_positions.

    Args:
        agent_pos: np.array of shape (2,) or (3,) (x, y, [theta])
        all_positions: np.array of shape (n_agents, 2) or (n_agents, 3)

    Returns:
        indices: list of two indices of the closest neighbors
        distances: list of their distances
    """
    agent_xy = agent_pos[:2]
    all_xy = all_positions[:, :2]
    # Compute Euclidean distances
    dists = np.linalg.norm(all_xy - agent_xy, axis=1)
    # Exclude self (distance zero)
    self_idx = np.where((all_xy == agent_xy).all(axis=1))[0]
    dists[self_idx] = np.inf
    # Get indices of two smallest distances
    indices = np.argsort(dists)[:2]
    return indices.tolist(), dists[indices].tolist()

def generate_random_laman_graph(n, positions):
    if n < 2:
        raise ValueError("Laman graph requires at least 2 vertices")

    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1)  # Start with 2 vertices and 1 edge

    for i in range(2, n):
        G.add_node(i)
        # Only consider existing nodes for neighbors
        existing_nodes = list(range(i))
        existing_positions = positions[existing_nodes]
        # Find two closest among existing nodes
        idx, _ = find_two_closest_neighbors(positions[i], existing_positions)
        G.add_edge(i, existing_nodes[idx[0]])
        G.add_edge(i, existing_nodes[idx[1]])

    A = nx.to_numpy_array(G, dtype=int)
    return G, A

def sort_positions_by_nearest_neighbor(positions):
    """
    Sorts positions so that each next position is the closest to the previous one.
    Greedy nearest-neighbor ordering.
    """
    positions = positions.copy()
    n = len(positions)
    visited = [False] * n
    order = [0]  # start from the first position
    visited[0] = True

    for _ in range(1, n):
        last = order[-1]
        dists = np.linalg.norm(positions - positions[last], axis=1)
        dists[visited] = np.inf  # ignore already visited
        next_idx = np.argmin(dists)
        order.append(next_idx)
        visited[next_idx] = True

    return positions[order]

def main(args=None):
    # Example usage
    n_vertices = 10
    positions = np.random.rand(n_vertices, 2)
    positions = sort_positions_by_nearest_neighbor(positions)
    print(positions)

    G, A = generate_random_laman_graph(n_vertices, positions)
    # Plot
    #pos = nx.spring_layout(G)
    pos = {i: positions[i] for i in range(n_vertices)}

    plt.figure(figsize=(6,6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
    plt.title(f"Random Laman Graph with {n_vertices} vertices")
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()
