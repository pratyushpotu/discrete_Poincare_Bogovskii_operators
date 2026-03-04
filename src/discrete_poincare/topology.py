import numpy as np
from scipy.sparse import lil_matrix

def calculate_d0_matrix(vertices, edges):
    d0 = lil_matrix((len(edges), len(vertices)), dtype=np.float64)
    for i, (u, v) in enumerate(edges):
        d0[i, u] = -1.0
        d0[i, v] = 1.0
    return d0.tocsc()

def calculate_d1_matrix(triangles, edges, edge_map):
    d1 = lil_matrix((len(triangles), len(edges)), dtype=np.float64)
    for i, tri in enumerate(triangles):
        # CCW boundary of triangle
        local_edges = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for u, v in local_edges:
            # Look up sorted edge index
            idx = edge_map[tuple(sorted((u, v)))]
            # Determine orientation sign
            sign = 1.0 if u < v else -1.0
            d1[i, idx] = sign
    return d1.tocsc()