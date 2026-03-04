import numpy as np

from discrete_poincare.mesh import generate_diagonal_aligned_mesh, extract_edges
from discrete_poincare.topology import calculate_d0_matrix, calculate_d1_matrix
from discrete_poincare.Whitney_Poincare_operators import compute_P1_Whitney, compute_P2_Whitney
from discrete_poincare.combinatorial_operators import compute_P1_combinatorial, compute_P2_combinatorial
from discrete_poincare.contraction import SimplicialMeshWrapper, DiscreteContraction, generate_compatible_contraction

def run_experiment():
    print("--- Running Experiment: Homotopy Identities ---")
    nx, ny = 8, 8
    pts, tris = generate_diagonal_aligned_mesh(nx, ny)
    edges, edge_map = extract_edges(tris)
    d0 = calculate_d0_matrix(pts, edges)
    d1 = calculate_d1_matrix(tris, edges, edge_map)
    a = [0.4, 0.4]

    # --- Whitney Form Based Operator Tests ---
    print("\n[Whitney Form Operators]")
    
    # Test 1: C^0
    u = np.random.rand(len(pts))
    d0u = d0 @ u
    P1d0u = compute_P1_Whitney(pts, edges, tris, d0u, edge_map, a)
    diff0 = u - P1d0u
    error0 = np.max(np.abs(diff0 - diff0[0]))
    print(f"Test 1 (C^0): Error between u and Pdu (- constant) = {error0:.6e}")

    # Test 2: C^1
    w = np.random.rand(len(edges))
    P1w = compute_P1_Whitney(pts, edges, tris, w, edge_map, a)
    d0P1w = d0 @ P1w
    d1w = d1 @ w
    P2d1w = compute_P2_Whitney(pts, edges, tris, d1w, a)
    error1 = np.max(np.abs(d0P1w + P2d1w - w))
    print(f"Test 2 (C^1): Error between w and (dP + Pd)w = {error1:.6e}")

    # Test 3: C^2
    q = np.random.rand(len(tris))
    P2q = compute_P2_Whitney(pts, edges, tris, q, a)
    d1P2q = d1 @ P2q
    error2 = np.max(np.abs(q - d1P2q))
    print(f"Test 3 (C^2): Error between q and dPq = {error2:.6e}")

    # --- Purely Combinatorial Operator Tests ---
    print("\n[Combinatorial Operators]")
    mesh_wrapper = SimplicialMeshWrapper(pts, tris)
    maps = generate_compatible_contraction(nx, ny, pts)
    contraction = DiscreteContraction(mesh_wrapper, maps)

    # Test 1: C^0
    P1d0u_comb = compute_P1_combinatorial(pts, edges, d0u, edge_map, contraction)
    diff0_comb = u - P1d0u_comb
    error0_comb = np.max(np.abs(diff0_comb - diff0_comb[0]))
    print(f"Test 1 (C^0): Error between u and Pdu (- constant) = {error0_comb:.6e}")

    # Test 2: C^1
    P1w_comb = compute_P1_combinatorial(pts, edges, w, edge_map, contraction)
    d0P1w_comb = d0 @ P1w_comb
    P2d1w_comb = compute_P2_combinatorial(tris, edges, d1w, edge_map, contraction)
    error1_comb = np.max(np.abs(d0P1w_comb + P2d1w_comb - w))
    print(f"Test 2 (C^1): Error between w and (dP + Pd)w = {error1_comb:.6e}")

    # Test 3: C^2
    P2q_comb = compute_P2_combinatorial(tris, edges, q, edge_map, contraction)
    d1P2q_comb = d1 @ P2q_comb
    error2_comb = np.max(np.abs(q - d1P2q_comb))
    print(f"Test 3 (C^2): Error between q and dPq = {error2_comb:.6e}")

if __name__ == "__main__":
    run_experiment()