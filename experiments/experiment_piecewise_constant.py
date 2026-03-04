import numpy as np
import matplotlib.pyplot as plt

from discrete_poincare.mesh import generate_diagonal_aligned_mesh, extract_edges
from discrete_poincare.topology import calculate_d1_matrix
from discrete_poincare.fields import create_nedelec_field_evaluator_cached
from discrete_poincare.Whitney_Poincare_operators import compute_P2_Whitney
from discrete_poincare.combinatorial_operators import compute_P2_combinatorial
from discrete_poincare.contraction import SimplicialMeshWrapper, DiscreteContraction, generate_compatible_contraction
from discrete_poincare.visualization import plot_field_refined

def g_func(p):
    x, y = p[0], p[1]
    return x * (1.0 - x) * y * (1.0 - y)

def run_experiment():
    print("--- Running Experiment: Vector Potential for Piecewise Constant ---")
    nx, ny = 8, 8
    pts, tris = generate_diagonal_aligned_mesh(nx, ny)
    edges, edge_map = extract_edges(tris)
    d1 = calculate_d1_matrix(tris, edges, edge_map)

    # Project g to Discrete 2-form
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

    # --- Whitney Operator ---
    a_star = [0.5, 0.5]
    vh_whitney = compute_P2_Whitney(pts, edges, tris, gh_cochain, a_star)
    recon_cochain_whitney = d1 @ vh_whitney
    recon_density_whitney = recon_cochain_whitney / tri_areas
    
    error_whitney = np.max(np.abs(gh_density - recon_density_whitney))
    print(f"[Whitney] ||nabla x v_h - g_h||_L^inf = {error_whitney:.6e}")

    # --- Combinatorial Operator ---
    mesh_wrapper = SimplicialMeshWrapper(pts, tris)
    maps = generate_compatible_contraction(nx, ny, pts)
    contraction = DiscreteContraction(mesh_wrapper, maps)

    vh_comb = compute_P2_combinatorial(tris, edges, gh_cochain, edge_map, contraction)
    recon_cochain_comb = d1 @ vh_comb
    recon_density_comb = recon_cochain_comb / tri_areas
    
    error_comb = np.max(np.abs(gh_density - recon_density_comb))
    print(f"[Combinatorial] ||nabla x v_h - g_h||_L^inf = {error_comb:.6e}")

    # --- Visualization (Whitney Example) ---
    eval_vh = create_nedelec_field_evaluator_cached(pts, tris, vh_whitney, edges, edge_map)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    ((ax1, ax2), (ax3, ax4)) = axes

    x_fine, y_fine = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_fine, y_fine)
    Z = g_func([X, Y])
    
    c1 = ax1.contourf(X, Y, Z, levels=50, cmap='viridis')
    ax1.set_title(r"1. Continuous Function $g$")
    ax1.set_aspect('equal')
    fig.colorbar(c1, ax=ax1, label="Value")

    pc2 = ax2.tripcolor(pts[:,0], pts[:,1], tris, facecolors=gh_density, edgecolors='k', lw=0.1, cmap='viridis')
    ax2.set_title(r"2. Piecewise Constant Projection $g_h$")
    ax2.set_aspect('equal')
    fig.colorbar(pc2, ax=ax2, label="Density")

    plot_field_refined(eval_vh, [0,1], [0,1], N=16, ax=ax3, mesh=(pts, tris), title=r"3. Discrete Vector Potential $v_h = \mathcal{P}\, g_h$")
    fig.colorbar(ax3.collections[0], ax=ax3, label="Magnitude")

    pc4 = ax4.tripcolor(pts[:,0], pts[:,1], tris, facecolors=recon_density_whitney, edgecolors='k', lw=0.1, cmap='viridis')
    ax4.set_title(r"4. Reconstruction $\nabla \times v_h$")
    ax4.set_aspect('equal')
    fig.colorbar(pc4, ax=ax4, label="Density")

    plt.suptitle("Discrete Poincare Operator on 2-forms", fontsize=16)
    plt.show()

if __name__ == "__main__":
    run_experiment()