import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from discrete_poincare.mesh import generate_diagonal_aligned_mesh, extract_edges
from discrete_poincare.topology import calculate_d0_matrix, calculate_d1_matrix
from discrete_poincare.fields import create_nedelec_field_evaluator_cached
from discrete_poincare.Whitney_Poincare_operators import compute_P1_Whitney
from discrete_poincare.combinatorial_operators import compute_P1_combinatorial
from discrete_poincare.contraction import SimplicialMeshWrapper, DiscreteContraction, generate_compatible_contraction
from discrete_poincare.visualization import project_to_nedelec, plot_field_refined

def f_func(p):
    x, y = p[0] - 0.5, p[1] - 0.5
    return np.array([y, x])

def run_experiment():
    print("--- Running Experiment: Potential of a Curl-Free Vector Field ---")
    nx, ny = 8, 8
    pts, tris = generate_diagonal_aligned_mesh(nx, ny)
    edges, edge_map = extract_edges(tris)
    d0 = calculate_d0_matrix(pts, edges)

    # 1. Project
    fh_cochain = project_to_nedelec(f_func, pts, edges)
    eval_fh = create_nedelec_field_evaluator_cached(pts, tris, fh_cochain, edges, edge_map)

    # --- Whitney Operator ---
    a_star = [0.25, 0.25]
    phi_h_whitney = compute_P1_Whitney(pts, edges, tris, fh_cochain, edge_map, a_star)
    recon_cochain_whitney = d0 @ phi_h_whitney
    
    error_whitney = np.max(np.abs(fh_cochain - recon_cochain_whitney))
    print(f"[Whitney] ||f_h - nabla phi_h||_L^inf = {error_whitney:.6e}")

    # --- Combinatorial Operator ---
    mesh_wrapper = SimplicialMeshWrapper(pts, tris)
    maps = generate_compatible_contraction(nx, ny, pts)
    contraction = DiscreteContraction(mesh_wrapper, maps)
    
    phi_h_comb = compute_P1_combinatorial(pts, edges, fh_cochain, edge_map, contraction)
    recon_cochain_comb = d0 @ phi_h_comb
    
    error_comb = np.max(np.abs(fh_cochain - recon_cochain_comb))
    print(f"[Combinatorial] ||f_h - nabla phi_h||_L^inf = {error_comb:.6e}")

    # --- Visualization (Whitney Example) ---
    eval_recon = create_nedelec_field_evaluator_cached(pts, tris, recon_cochain_whitney, edges, edge_map)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    ((ax1, ax2), (ax3, ax4)) = axes

    plot_field_refined(f_func, [0,1], [0,1], N=16, ax=ax1, title=r"1. Original Field $f$")
    fig.colorbar(ax1.collections[0], ax=ax1, label='Magnitude')

    plot_field_refined(eval_fh, [0,1], [0,1], N=16, ax=ax2, mesh=(pts, tris), title=r"2. Nedelec Projection $f_h$")
    fig.colorbar(ax2.collections[0], ax=ax2, label='Magnitude')

    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    contour = ax3.tricontourf(triang, phi_h_whitney, levels=20, cmap='viridis')
    ax3.triplot(triang, 'k-', lw=0.3, alpha=0.3)
    ax3.set_aspect('equal')
    ax3.set_title(r"3. Discrete Potential $\phi_h = \mathcal{P} f_h$ (Whitney)")
    fig.colorbar(contour, ax=ax3, label="Potential Value")

    plot_field_refined(eval_recon, [0,1], [0,1], N=16, ax=ax4, mesh=(pts, tris), title=r"4. Reconstruction $\nabla \phi_h$")
    fig.colorbar(ax4.collections[0], ax=ax4, label='Magnitude')

    plt.suptitle("Poincaré Operator on 1-forms", fontsize=16)
    plt.show()

if __name__ == "__main__":
    run_experiment()