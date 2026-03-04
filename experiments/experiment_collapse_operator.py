import numpy as np

from discrete_poincare.mesh import generate_diagonal_aligned_mesh
from discrete_poincare.collapse_operators import (
    SimplicialComplex,
    build_coboundary_operators,
    find_collapse_sequence,
    build_collapse_poincare_operator
)

def run_experiment():
    print("--- Running Experiment: Collapse-based Operator Homotopy Identities ---")
    
    # Generate a small mesh to test the combinatorial collapse
    # We use a 4x4 grid (which is strongly collapsible)
    nx, ny = 4, 4
    pts, tris = generate_diagonal_aligned_mesh(nx, ny)
    
    print(f"Generated mesh with {len(pts)} vertices and {len(tris)} triangles.")
    
    # 1. Initialize the discrete complex
    c = SimplicialComplex(pts, tris)
    
    # 2. Build the coboundary operators based on the complex's internal geometric orientation
    d0, d1 = build_coboundary_operators(c)
    
    # 3. Compute the collapse sequence
    print("Computing collapse sequence...")
    collapse_seq, final_vertex = find_collapse_sequence(c)
    final_idx = c.vertex_map[final_vertex]
    print(f"Collapse sequence found with {len(collapse_seq)} steps. Contracted to vertex index: {final_idx}")
    
    # 4. Build the Poincaré operator matrices from the sequence
    print("Building Poincaré operator matrices P0 and P1...")
    P0, P1 = build_collapse_poincare_operator(c, collapse_seq)
    
    # --- Homotopy Identity Tests ---
    print("\n[Homotopy Identity Tests: dP + Pd = id]")
    
    # Test 1: 0-Cochains (Vertices)
    # For a 0-cochain u, we expect: P_0 d_0 u = u - constant
    u = np.random.rand(c.num_vertices)
    d0u = d0 @ u
    P0d0u = P0 @ d0u
    diff0 = u - P0d0u
    # The difference should be a constant vector
    error0 = np.max(np.abs(diff0 - diff0[0]))
    print(f"Test 1 (C^0): Error between u and P_0 d_0 u (- constant) = {error0:.6e}")
    
    # Test 2: 1-Cochains (Edges)
    # For a 1-cochain w, we expect: d_0 P_0 w + P_1 d_1 w = w
    w = np.random.rand(c.num_edges)
    d0P0w = d0 @ (P0 @ w)
    P1d1w = P1 @ (d1 @ w)
    error1 = np.max(np.abs(d0P0w + P1d1w - w))
    print(f"Test 2 (C^1): Error between w and (d_0 P_0 + P_1 d_1)w = {error1:.6e}")
    
    # Test 3: 2-Cochains (Faces)
    # For a 2-cochain q, we expect: d_1 P_1 q = q  (since d_2 = 0)
    q = np.random.rand(c.num_faces)
    d1P1q = d1 @ (P1 @ q)
    error2 = np.max(np.abs(q - d1P1q))
    print(f"Test 3 (C^2): Error between q and d_1 P_1 q = {error2:.6e}")

if __name__ == "__main__":
    run_experiment()