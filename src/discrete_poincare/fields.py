import numpy as np
from scipy.integrate import quad

def create_nedelec_field_evaluator_cached(verts, tris, u1, edges, edge_map, tol=1e-12):
    verts, tris = np.asarray(verts), np.asarray(tris, int)
    cache = []

    for v_idx in tris:
        pts = verts[v_idx]
        mat = np.column_stack([pts[:,0], pts[:,1], [1,1,1]])
        area = 0.5 * np.linalg.det(mat)
        if abs(area) < 1e-14:
            cache.append(None); continue

        invA = np.linalg.inv(mat.T) # Barycentric transform matrix
        grads = np.zeros((3,2)) # Gradients of lambda_i
        # grad lambda_i formula based on rotating edge vectors
        grads[0] = [pts[1,1]-pts[2,1], pts[2,0]-pts[1,0]]
        grads[1] = [pts[2,1]-pts[0,1], pts[0,0]-pts[2,0]]
        grads[2] = [pts[0,1]-pts[1,1], pts[1,0]-pts[0,0]]
        grads /= (2 * area)

        local_dofs = np.zeros(3)
        pairs = [(0,1), (1,2), (2,0)]
        for k, (i, j) in enumerate(pairs):
            gi, gj = v_idx[i], v_idx[j]
            key = tuple(sorted((gi, gj)))
            sign = 1.0 if (gi, gj) == key else -1.0
            local_dofs[k] = sign * u1[edge_map[key]]

        cache.append({"invA": invA, "grads": grads, "dofs": local_dofs})

    last_tri = [0]
    def eval_field(p):
        pvec = np.array([p[0], p[1], 1.0])
        # Try cached triangle, then search
        indices = [last_tri[0]] + list(range(len(cache)))
        for idx in indices:
            if idx >= len(cache) or cache[idx] is None: continue
            data = cache[idx]
            bary = data["invA"] @ pvec
            if np.all(bary >= -tol) and np.all(bary <= 1.0 + tol):
                last_tri[0] = idx
                g, u = data["grads"], data["dofs"]
                # Whitney basis sum: u_ij * (Li grad Lj - Lj grad Li)
                val = (u[0]*(bary[0]*g[1]-bary[1]*g[0]) +
                       u[1]*(bary[1]*g[2]-bary[2]*g[1]) +
                       u[2]*(bary[2]*g[0]-bary[0]*g[2]))
                return val
        return np.zeros(2)
    return eval_field

def compute_line_integral_with_splits(field_func, start, end, splits):
    total, vec = 0.0, end - start
    for t0, t1 in zip(splits[:-1], splits[1:]):
        if t1 - t0 < 1e-12: continue
        seg_vec = (t1 - t0) * vec
        base = start + t0 * vec
        func = lambda u: np.dot(field_func(base + u * seg_vec), seg_vec)
        total += quad(func, 0, 1, limit=50)[0]
    return total