import numpy as np

from discrete_poincare.mesh import generate_diagonal_aligned_mesh, extract_edges
from discrete_poincare.topology import calculate_d0_matrix, calculate_d1_matrix
from discrete_poincare.Bogovskii_operators import compute_B1_Whitney, compute_B2_Whitney

def run_experiment():
    print("--- Running Experiment: Homotopy Identities (Homogeneous Boundary Conditions) ---")
    nx, ny = 8, 8
    pts, tris = generate_diagonal_aligned_mesh(nx, ny)
    edges, edge_map = extract_edges(tris)
    d0 = calculate_d0_matrix(pts, edges)
    d1 = calculate_d1_matrix(tris, edges, edge_map)
    a = [0.433, 0.412]
    eps = 1e-10

    # Test 1: C^0
    u = np.random.rand(len(pts))
    boundary_mask_pts = (
        (pts[:, 0] < eps) | (pts[:, 0] > 1.0 - eps) |
        (pts[:, 1] < eps) | (pts[:, 1] > 1.0 - eps)
    )
    u[boundary_mask_pts] = 0.0

    d0u = d0 @ u
    B1d0u = compute_B1_Whitney(pts, edges, tris, d0u, edge_map, a)
    diff = u - B1d0u
    print(f"Test 1 (C^0): Error between u and Bdu = {np.max(np.abs(diff)):.6e}")

    # Test 2: C^1
    w = np.random.rand(len(edges))
    edge_indices = np.array(edges)
    p1 = pts[edge_indices[:, 0]]
    p2 = pts[edge_indices[:, 1]]
    on_left   = (p1[:, 0] < eps) & (p2[:, 0] < eps)
    on_right  = (p1[:, 0] > 1.0 - eps) & (p2[:, 0] > 1.0 - eps)
    on_bottom = (p1[:, 1] < eps) & (p2[:, 1] < eps)
    on_top    = (p1[:, 1] > 1.0 - eps) & (p2[:, 1] > 1.0 - eps)
    boundary_mask_edges = on_left | on_right | on_bottom | on_top
    w[boundary_mask_edges] = 0.0

    B1w = compute_B1_Whitney(pts, edges, tris, w, edge_map, a)
    d0B1w = d0 @ B1w
    d1w = d1 @ w
    B2d1w = compute_B2_Whitney(pts, edges, tris, d1w, a)
    print(f"Test 2 (C^1): Error between w and (dB + Bd)w = {np.max(np.abs(d0B1w + B2d1w - w)):.6e}")

    # Test 3: C^2
    q = np.random.rand(len(tris))
    tri_areas = np.zeros(len(tris))
    for i, tri in enumerate(tris):
        v = pts[tri]
        d1_vec, d2_vec = v[1]-v[0], v[2]-v[0]
        area = 0.5 * np.abs(d1_vec[0]*d2_vec[1] - d1_vec[1]*d2_vec[0])
        tri_areas[i] = area

    current_integral = np.sum(q * tri_areas)
    total_area = np.sum(tri_areas)
    mean_val = current_integral / total_area
    q -= mean_val

    B2q = compute_B2_Whitney(pts, edges, tris, q, a)
    d1B2q = d1 @ B2q
    print(f"Test 3 (C^2): Error between q and dBq = {np.max(np.abs(q - d1B2q)):.6e}")

if __name__ == "__main__":
    run_experiment()