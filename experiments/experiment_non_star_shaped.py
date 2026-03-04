import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from discrete_poincare.mesh import generate_u_mesh, extract_edges
from discrete_poincare.topology import calculate_d0_matrix, calculate_d1_matrix
from discrete_poincare.fields import create_nedelec_field_evaluator_cached
from discrete_poincare.L_contraction_operators import compute_P1_L_contraction, compute_P2_L_contraction
from discrete_poincare.Whitney_Poincare_operators import compute_P1_Whitney
from discrete_poincare.visualization import project_to_nedelec, plot_field_refined

def f_func(p):
    x, y = p[0] - 0.5, p[1] - 0.5
    return np.array([y, x])

def run_experiment():
    print("--- Running Experiment: Non-Star-Shaped Domain ---")
    n = 4
    pts, tris = generate_u_mesh(n)
    edges, edge_map = extract_edges(tris)
    d0 = calculate_d0_matrix(pts, edges)
    d1 = calculate_d1_matrix(tris, edges, edge_map)

    # 1. Project
    fh_cochain = project_to_nedelec(f_func, pts, edges)

    # 2. Potential (L-contraction for f_h visualization)
    a_L = [0.2, 0.2] # Use a base point that routes the L-path safely through the bottom corridor
    phi_h = compute_P1_L_contraction(pts, edges, tris, fh_cochain, edge_map, a_L)
    recon_cochain = d0 @ phi_h
    
    error_vec = fh_cochain - recon_cochain
    infinity_error = np.max(np.abs(error_vec))
    print(f"[Visualization] ||f_h - nabla phi_h||_L^inf = {infinity_error:.6e}")

    # --- L-Contraction Homotopy Tests ---
    print("\n[L-Contraction Homotopy Identities]")
    a_L = [0.2, 0.2]

    # Test 1: C^0
    u = np.random.rand(len(pts))
    d0u = d0 @ u
    P1d0u = compute_P1_L_contraction(pts, edges, tris, d0u, edge_map, a_L)
    diff = u - P1d0u
    error0 = np.max(np.abs(diff - diff[0]))
    print(f"Test 1 (C^0): Error between u and Pdu (- constant) = {error0:.6e}")

    # Test 2: C^1
    w = np.random.rand(len(edges))
    P1w = compute_P1_L_contraction(pts, edges, tris, w, edge_map, a_L)
    d0P1w = d0 @ P1w
    d1w = d1 @ w
    P2d1w = compute_P2_L_contraction(pts, edges, tris, d1w, a_L)
    error1 = np.max(np.abs(d0P1w + P2d1w - w))
    print(f"Test 2 (C^1): Error between w and (dP + Pd)w = {error1:.6e}")

    # Test 3: C^2
    q = np.random.rand(len(tris))
    P2q = compute_P2_L_contraction(pts, edges, tris, q, a_L)
    d1P2q = d1 @ P2q
    error2 = np.max(np.abs(q - d1P2q))
    print(f"Test 3 (C^2): Error between q and dPq = {error2:.6e}")

if __name__ == "__main__":
    run_experiment()