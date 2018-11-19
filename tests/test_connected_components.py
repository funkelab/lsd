import lsd
import numpy as np

if __name__ == "__main__":

    nodes = np.array([0, 1, 2, np.uint64(-10)], dtype=np.uint64)
    edges = np.array([[1, 2], [2, np.uint64(-10)]], dtype=np.uint64)
    edge_scores = np.array([0.1, 0.5], dtype=np.float32)

    components = lsd.connected_components(nodes, edges, edge_scores, 0.0)
    print(components)

    components = lsd.connected_components(nodes, edges, edge_scores, 0.1)
    print(components)

    components = lsd.connected_components(nodes, edges, edge_scores, 0.3)
    print(components)

    components = lsd.connected_components(nodes, edges, edge_scores, 0.5)
    print(components)

    components = lsd.connected_components(nodes, edges, edge_scores, 0.7)
    print(components)
