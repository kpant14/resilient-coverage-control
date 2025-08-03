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
    
def generate_random_laman_graph(n):
    if n < 2:
        raise ValueError("Laman graph requires at least 2 vertices")

    G = nx.Graph()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1)  # Start with 2 vertices and 1 edge

    for i in range(2, n):
        # Henneberg-I: add vertex i connected to 2 existing vertices
        G.add_node(i)
        existing_nodes = list(G.nodes)
        existing_nodes.remove(i)
        u, v = random.sample(existing_nodes, 2)
        G.add_edge(i, u)
        G.add_edge(i, v)

    A = nx.to_numpy_array(G, dtype=int)
    return A

def main(args=None):
    # Example usage
    n_vertices = 10
    G = generate_random_laman_graph(n_vertices)

    # Plot
    pos = nx.spring_layout(G)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
    plt.title(f"Random Laman Graph with {n_vertices} vertices")
    plt.axis('equal')
    plt.show()

if __name__ == '__main__':
    main()
