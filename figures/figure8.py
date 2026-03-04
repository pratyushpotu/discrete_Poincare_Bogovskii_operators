import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from discrete_poincare.mesh import generate_diagonal_aligned_mesh, extract_edges
from discrete_poincare.topology import calculate_d1_matrix
from discrete_poincare.fields import create_nedelec_field_evaluator_cached
from discrete_poincare.Whitney_Poincare_operators import compute_P2_Whitney
from discrete_poincare.collapse_operators import (
    SimplicialComplex,
    build_coboundary_operators,
    find_collapse_sequence,
    build_collapse_poincare_operator
)
from discrete_poincare.visualization import plot_field_refined

def g_func(p):
    """Continuous scalar function g(x,y)"""
    x, y = p[0], p[1]
    return x * (1.0 - x) * y * (1.0 - y)

def run_experiment():
    print("Generating Figure 8: Discrete 2D Vector Potentials (Section 9.1)")
    
    # --- 1. Mesh Setup ---
    nx, ny = 8, 8
    pts, tris = generate_diagonal_aligned_mesh(nx, ny)
    edges, edge_map = extract_edges(tris)
    d1 = calculate_d1_matrix(tris, edges, edge_map)

    # --- 2. Projection g -> g_h ---
    gh_cochain = np.zeros(len(tris))
    tri_areas = np.zeros(len(tris))

    for i, tri in enumerate(tris):
        v = pts[tri]
        centroid = np.mean(v, axis=0)
        d1_vec, d2_vec = v[1] - v[0], v[2] - v[0]
        area = 0.5 * np.abs(d1_vec[0] * d2_vec[1] - d1_vec[1] * d2_vec[0])
        tri_areas[i] = area
        gh_cochain[i] = g_func(centroid) * area

    gh_density = gh_cochain / tri_areas

    # --- 3. Compute Whitney Operator (v_h) ---
    print("Computing Whitney geometric potential...")
    a_star = [0.5, 0.5]
    vh_whitney = compute_P2_Whitney(pts, edges, tris, gh_cochain, a_star)
    recon_cochain_whitney = d1 @ vh_whitney
    recon_density_whitney = recon_cochain_whitney / tri_areas
    
    eval_vh_whitney = create_nedelec_field_evaluator_cached(pts, tris, vh_whitney, edges, edge_map)

    # --- 4. Compute Collapse Operator (w_h) ---
    print("Computing combinatorial collapse sequence and operator...")
    c = SimplicialComplex(pts, tris)
    # Re-extract perfectly aligned d1 matrix directly from the complex to guarantee orientation match
    _, d1_collapse_matrix = build_coboundary_operators(c)
    
    collapse_seq, final_vertex = find_collapse_sequence(c)
    print(f"Mesh collapsed to vertex {final_vertex} in {len(collapse_seq)} steps.")
    
    P0, P1 = build_collapse_poincare_operator(c, collapse_seq)

    # P1 maps 2-cochains (faces) to 1-cochains (edges) - equivalent to P^2
    wh_collapse = P1 @ gh_cochain
    recon_cochain_collapse = d1_collapse_matrix @ wh_collapse
    recon_density_collapse = recon_cochain_collapse / tri_areas
    
    eval_wh_collapse = create_nedelec_field_evaluator_cached(pts, tris, wh_collapse, edges, edge_map)

    # --- 5. Visualization (Publication Layout) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ((ax_g, ax_whitney_v, ax_collapse_v), (ax_gh, ax_whitney_recon, ax_collapse_recon)) = axes

    # Column 1: Densities (Continuous vs Discrete)
    x_fine, y_fine = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_fine, y_fine)
    Z = g_func([X, Y])
    
    c1 = ax_g.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax_g.set_title(r"(a) Continuous Function $g$")
    ax_g.set_aspect('equal')
    fig.colorbar(c1, ax=ax_g, label="Value")

    pc2 = ax_gh.tripcolor(pts[:,0], pts[:,1], tris, facecolors=gh_density, edgecolors='k', lw=0.1, cmap='viridis')
    ax_gh.set_title(r"(d) Projected $g_h = \Pi g$")
    ax_gh.set_aspect('equal')
    fig.colorbar(pc2, ax=ax_gh, label="Value")

    # Column 2: Whitney Operator (Geometric)
    plot_field_refined(eval_vh_whitney, [0,1], [0,1], N=16, ax=ax_whitney_v, mesh=(pts, tris), 
                       title=r"(b) Whitney Potential $v_h = \mathcal{P} g_h$")
    fig.colorbar(ax_whitney_v.collections[0], ax=ax_whitney_v, label="Magnitude")

    pc_w_recon = ax_whitney_recon.tripcolor(pts[:,0], pts[:,1], tris, facecolors=recon_density_whitney, edgecolors='k', lw=0.1, cmap='viridis')
    ax_whitney_recon.set_title(r"(e) Reconstruction $\nabla \times v_h$")
    ax_whitney_recon.set_aspect('equal')
    fig.colorbar(pc_w_recon, ax=ax_whitney_recon, label="Value")

    # Column 3: Collapse Operator (Combinatorial)
    plot_field_refined(eval_wh_collapse, [0,1], [0,1], N=16, ax=ax_collapse_v, mesh=(pts, tris), 
                       title=r"(c) Collapse Potential $w_h = W P \mathcal{R} g_h$")
    fig.colorbar(ax_collapse_v.collections[0], ax=ax_collapse_v, label="Magnitude")

    pc_c_recon = ax_collapse_recon.tripcolor(pts[:,0], pts[:,1], tris, facecolors=recon_density_collapse, edgecolors='k', lw=0.1, cmap='viridis')
    ax_collapse_recon.set_title(r"(f) Reconstruction $\nabla \times w_h$")
    ax_collapse_recon.set_aspect('equal')
    fig.colorbar(pc_c_recon, ax=ax_collapse_recon, label="Value")

    # plt.suptitle("Figure 8: Discrete Vector Potentials on a Star-Shaped Domain", fontsize=18, fontweight='bold')
    
    # Save the figure in high resolution for the paper
    plt.savefig("figure8_vector_potentials.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_experiment()