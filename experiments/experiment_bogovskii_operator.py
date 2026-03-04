import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from discrete_poincare.mesh import generate_diagonal_aligned_mesh, extract_edges
from discrete_poincare.topology import calculate_d0_matrix, calculate_d1_matrix
from discrete_poincare.fields import create_nedelec_field_evaluator_cached
from discrete_poincare.Bogovskii_operators import compute_B1_Whitney, compute_B2_Whitney
from discrete_poincare.visualization import project_to_nedelec, plot_field_refined

def phi_func(x, y):
    return x * (1.0 - x) * y * (1.0 - y)

def f_func(p):
    x, y = p[0], p[1]
    ux = (1.0 - 2.0 * x) * y * (1.0 - y)
    uy = x * (1.0 - x) * (1.0 - 2.0 * y)
    return np.array([ux, uy])

def g_zero_mean_func(p):
    x, y = p[0], p[1]
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def run_experiment():
    print("--- Running Experiment: Bogovskii Operator ---")
    nx, ny = 8, 8
    pts, tris = generate_diagonal_aligned_mesh(nx, ny)
    edges, edge_map = extract_edges(tris)
    d1 = calculate_d1_matrix(tris, edges, edge_map)

    # --- 1-Form Test ---
    print("\n[Bogovskii on 1-forms]")
    fh_cochain = project_to_nedelec(f_func, pts, edges)
    a_star_1 = [0.6, 0.5]
    phi_h = compute_B1_Whitney(pts, edges, tris, fh_cochain, edge_map, a_star_1)
    
    phi_exact_at_nodes = phi_func(pts[:,0], pts[:,1])
    error1 = np.max(np.abs(phi_exact_at_nodes - phi_h))
    print(f"||Pi phi - phi_h||_L^inf = {error1:.6e}")

    # --- 2-Form Zero-Mean Test ---
    print("\n[Bogovskii on 2-forms with Zero Mean]")
    nx2, ny2 = 10, 10
    pts2, tris2 = generate_diagonal_aligned_mesh(nx2, ny2)
    edges2, edge_map2 = extract_edges(tris2)
    d1_2 = calculate_d1_matrix(tris2, edges2, edge_map2)

    gh_cochain = np.zeros(len(tris2))
    tri_areas = np.zeros(len(tris2))
    for i, tri in enumerate(tris2):
        v = pts2[tri]
        centroid = np.mean(v, axis=0)
        d1_vec, d2_vec = v[1]-v[0], v[2]-v[0]
        area = 0.5 * np.abs(d1_vec[0]*d2_vec[1] - d1_vec[1]*d2_vec[0])
        tri_areas[i] = area
        gh_cochain[i] = g_zero_mean_func(centroid) * area

    print(f"Total Integral of g_h: {np.sum(gh_cochain):.4e} (Should be ~0)")

    a_star_2 = [0.52, 0.511]
    vh = compute_B2_Whitney(pts2, edges2, tris2, gh_cochain, a_star_2)
    recon_cochain = d1_2 @ vh
    
    gh_density = gh_cochain / tri_areas
    recon_density = recon_cochain / tri_areas
    density_error = np.max(np.abs(gh_density - recon_density))
    print(f"||nabla x v_h - g_h||_L^inf = {density_error:.6e}")

if __name__ == "__main__":
    run_experiment()