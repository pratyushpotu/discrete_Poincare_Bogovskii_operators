import numpy as np

def compute_P1_combinatorial(vertices, edges, w_1chain, edge_map, contraction):
    """P1: 1-chain -> 0-chain"""
    res = np.zeros(len(vertices))
    for v_idx in range(len(vertices)):
        chain = contraction.contract_chain((v_idx,))
        val = 0.0
        for simplex, sign in chain:
            u, v = simplex
            if u == v: continue

            key = tuple(sorted((u, v)))
            if key not in edge_map: continue
            e_idx = edge_map[key]

            # Integral orientation
            orient = 1.0 if u < v else -1.0
            val += sign * orient * w_1chain[e_idx]
        res[v_idx] = val
    return res

def compute_P2_combinatorial(triangles, edges, w_2chain, edge_map, contraction):
    """P2: 2-chain -> 1-chain"""
    res = np.zeros(len(edges))
    tri_map = {tuple(sorted(t)): i for i, t in enumerate(triangles)}

    for e_idx, edge in enumerate(edges):
        chain = contraction.contract_chain(edge)
        val = 0.0
        for simplex, sign in chain:
            # Filter degenerate
            if len(set(simplex)) < 3: continue

            # Identify triangle index
            key = tuple(sorted(simplex))
            if key not in tri_map: continue
            t_idx = tri_map[key]

            # Parity check
            stored = triangles[t_idx]
            v_map = {v: k for k, v in enumerate(stored)}
            p = [v_map[n] for n in simplex]
            parity = np.sign((p[1]-p[0]) * (p[2]-p[0]) * (p[2]-p[1]))

            val += sign * parity * w_2chain[t_idx]
        res[e_idx] = val
    return res