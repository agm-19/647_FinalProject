import networkx as nx, numpy as np

def create_mesh_network(rows=2, cols=3, *,   # ← new args
                        seed=42,
                        extra_prob=0.0,      # probability of random extra edges
                        cap_low=1, cap_high=5):
    """
    Build a rows×cols grid (mesh) + optional random extra links.
    Returns (graph G, capacity matrix c).
    """
    np.random.seed(seed)
    G = nx.grid_2d_graph(rows, cols)            # base lattice
    G = nx.convert_node_labels_to_integers(G)   # relabel 0…N-1
    N = rows * cols

    # sprinkle additional random edges to make it 'meshier'
    for i in range(N):
        for j in range(i+1, N):
            if np.random.rand() < extra_prob and not G.has_edge(i, j):
                G.add_edge(i, j)

    adj = nx.to_numpy_array(G)
    c = np.random.randint(cap_low, cap_high+1, size=adj.shape) * adj
    return G, c
