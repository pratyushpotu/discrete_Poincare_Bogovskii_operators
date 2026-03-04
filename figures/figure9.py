import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from discrete_poincare.mesh import generate_u_mesh, extract_edges
from discrete_poincare.topology import calculate_d0_matrix
from discrete_poincare.fields import create_nedelec_field_evaluator_cached
from discrete_poincare.L_contraction_operators import compute_P1_L_contraction
from discrete_poincare.combinatorial_operators import compute_P1_combinatorial
from discrete_poincare.contraction import SimplicialMeshWrapper, DiscreteContraction, generate_u_mesh_contraction
from discrete_poincare.visualization import project_to_nedelec, plot_field_refined

def f_func(p):
    """A continuous curl-free vector field f(x,y) = (y-0.5, x-0.5)"""
    x, y = p[0] - 0.5, p[1] - 0.5
    return np.array([y, x])

def run_experiment():
    print("Generating Figure 9: Discrete Scalar Potentials on a Non-Star-Shaped Domain (Section 9.2)")
    
    # --- 1. Mesh Setup ---
    level = 4
    pts, tris = generate_u_mesh(level)
    edges, edge_map = extract_edges(tris)
    d0 = calculate_d0_matrix(pts, edges)

    # --- 2. Projection f -> f_h ---
    fh_cochain = project_to_nedelec(f_func, pts, edges)
    eval_fh = create_nedelec_field_evaluator_cached(pts, tris, fh_cochain, edges, edge_map)

    # --- 3. Compute Whitney Operator (L-Contraction) ---
    print("Computing Whitney geometric potential (L-Contraction)...")
    a_L = [0.2, 0.2]  # Contraction point safely within the base of the U-shape
    phi_h_whitney = compute_P1_L_contraction(pts, edges, tris, fh_cochain, edge_map, a_L)
    
    recon_cochain_whitney = d0 @ phi_h_whitney
    eval_recon_whitney = create_nedelec_field_evaluator_cached(pts, tris, recon_cochain_whitney, edges, edge_map)

    # --- 4. Compute Combinatorial Operator (Strong Collapse) ---
    print("Computing combinatorial potential (Strong Collapse Sequence)...")
    mesh_wrapper = SimplicialMeshWrapper(pts, tris)
    maps = generate_u_mesh_contraction(level, pts)
    contraction = DiscreteContraction(mesh_wrapper, maps)

    phi_h_comb = compute_P1_combinatorial(pts, edges, fh_cochain, edge_map, contraction)
    
    recon_cochain_comb = d0 @ phi_h_comb
    eval_recon_comb = create_nedelec_field_evaluator_cached(pts, tris, recon_cochain_comb, edges, edge_map)

    # --- 5. Visualization (Publication Layout) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ((ax_f, ax_whitney_phi, ax_comb_phi), (ax_fh, ax_whitney_recon, ax_comb_recon)) = axes

    # Column 1: Vector Fields (Continuous vs Projected)
    plot_field_refined(f_func, [0, 1], [0, 1], N=16, ax=ax_f, title=r"(a) Continuous Curl-Free Field $f$")
    fig.colorbar(ax_f.collections[0], ax=ax_f, label="Magnitude")

    plot_field_refined(eval_fh, [0, 1], [0, 1], N=16, ax=ax_fh, mesh=(pts, tris), title=r"(d) Projected $f_h = \Pi f$")
    fig.colorbar(ax_fh.collections[0], ax=ax_fh, label="Magnitude")

    # Column 2: Whitney L-Contraction (Geometric)
    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    
    contour_w = ax_whitney_phi.tricontourf(triang, phi_h_whitney, levels=20, cmap='viridis')
    ax_whitney_phi.triplot(triang, 'k-', lw=0.3, alpha=0.3)
    # ax_whitney_phi.plot(a_L[0], a_L[1], 'r*', markersize=12, label='Contraction Point $a$')
    # ax_whitney_phi.legend(loc='upper left')
    ax_whitney_phi.set_aspect('equal')
    ax_whitney_phi.set_title(r"(b) Whitney Potential $\phi_h^W = \mathcal{P} f_h$")
    fig.colorbar(contour_w, ax=ax_whitney_phi, label="Value")

    plot_field_refined(eval_recon_whitney, [0, 1], [0, 1], N=16, ax=ax_whitney_recon, mesh=(pts, tris), 
                       title=r"(e) Reconstruction $\nabla \phi_h^W$")
    fig.colorbar(ax_whitney_recon.collections[0], ax=ax_whitney_recon, label="Magnitude")

    # Column 3: Combinatorial Operator (Strong Collapse)
    contour_c = ax_comb_phi.tricontourf(triang, phi_h_comb, levels=20, cmap='viridis')
    ax_comb_phi.triplot(triang, 'k-', lw=0.3, alpha=0.3)
    ax_comb_phi.set_aspect('equal')
    ax_comb_phi.set_title(r"(c) Combinatorial Potential $\phi_h^C = W P \mathcal{R} f_h$")
    fig.colorbar(contour_c, ax=ax_comb_phi, label="Value")

    plot_field_refined(eval_recon_comb, [0, 1], [0, 1], N=16, ax=ax_comb_recon, mesh=(pts, tris), 
                       title=r"(f) Reconstruction $\nabla \phi_h^C$")
    fig.colorbar(ax_comb_recon.collections[0], ax=ax_comb_recon, label="Magnitude")

    # plt.suptitle("Figure 9: Discrete Scalar Potentials on a U-Shaped Domain", fontsize=18, fontweight='bold')
    
    # Save the figure in high resolution for the paper
    plt.savefig("figure9_non_star_shaped.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_experiment()