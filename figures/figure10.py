import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from discrete_poincare.mesh import generate_diagonal_aligned_mesh, extract_edges
from discrete_poincare.topology import calculate_d1_matrix
from discrete_poincare.fields import create_nedelec_field_evaluator_cached
from discrete_poincare.Bogovskii_operators import compute_B2_Whitney
from discrete_poincare.visualization import plot_field_refined

def g_base_func(p):
    """Base polynomial function before zero-mean correction."""
    x, y = p[0], p[1]
    return x * (1.0 - x) * y * (1.0 - y)

def run_experiment():
    print("Generating Figure 10: Discrete Bogovskii Operator (Section 9.3)")
    
    # --- 1. Mesh Setup ---
    nx, ny = 10, 10
    pts, tris = generate_diagonal_aligned_mesh(nx, ny)
    edges, edge_map = extract_edges(tris)
    d1 = calculate_d1_matrix(tris, edges, edge_map)

    # --- 2. Projection g -> g_h (Zero-Mean 2-form) ---
    gh_cochain = np.zeros(len(tris))
    tri_areas = np.zeros(len(tris))
    
    for i, tri in enumerate(tris):
        v = pts[tri]
        centroid = np.mean(v, axis=0)
        d1_vec, d2_vec = v[1] - v[0], v[2] - v[0]
        area = 0.5 * np.abs(d1_vec[0] * d2_vec[1] - d1_vec[1] * d2_vec[0])
        tri_areas[i] = area
        gh_cochain[i] = g_base_func(centroid) * area

    # Enforce zero mean discretely
    current_integral = np.sum(gh_cochain)
    total_area = np.sum(tri_areas)
    mean_val = current_integral / total_area
    gh_cochain -= mean_val * tri_areas

    gh_density = gh_cochain / tri_areas

    # --- 3. Compute Bogovskii Operator ---
    # We use a star point slightly offset from the exact center 
    # to avoid intersecting mesh facets, per the paper's remark.
    a_star = [0.52, 0.511]
    print(f"Computing Bogovskii potential with star point {a_star}...")
    vh_bogovskii = compute_B2_Whitney(pts, edges, tris, gh_cochain, a_star)
    
    recon_cochain = d1 @ vh_bogovskii
    recon_density = recon_cochain / tri_areas
    
    density_error = np.max(np.abs(gh_density - recon_density))
    print(f"||nabla x v_h - g_h||_L^inf = {density_error:.6e}")

    eval_vh = create_nedelec_field_evaluator_cached(pts, tris, vh_bogovskii, edges, edge_map)

    # --- 4. Visualization (Single Row Layout) ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    (ax_g, ax_gh, ax_pot, ax_recon) = axes

    # (a) Continuous Function g (Zero-Mean)
    x_fine, y_fine = np.linspace(0, 1, 100), np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x_fine, y_fine)
    # The continuous mean of x(1-x)y(1-y) over [0,1]^2 is 1/36
    Z = g_base_func([X, Y]) - (1.0 / 36.0)
    
    c1 = ax_g.contourf(X, Y, Z, levels=50, cmap='coolwarm')
    ax_g.set_title(r"(a) Continuous Function $g$")
    ax_g.set_aspect('equal')
    fig.colorbar(c1, ax=ax_g, label="Value")

    # (b) Projected Piecewise Constant g_h
    pc2 = ax_gh.tripcolor(pts[:,0], pts[:,1], tris, facecolors=gh_density, edgecolors='k', lw=0.1, cmap='coolwarm')
    ax_gh.set_title(r"(b) Projected $g_h = \Pi g$")
    ax_gh.set_aspect('equal')
    fig.colorbar(pc2, ax=ax_gh, label="Value")

    # (c) Discrete Bogovskii Potential v_h
    plot_field_refined(eval_vh, [0, 1], [0, 1], N=14, ax=ax_pot, mesh=(pts, tris),
                       title=r"(c) Bogovskii Potential $v_h = \mathcal{B} g_h$")
    fig.colorbar(ax_pot.collections[0], ax=ax_pot, label="Magnitude")

    # (d) Reconstruction d(v_h)
    pc4 = ax_recon.tripcolor(pts[:,0], pts[:,1], tris, facecolors=recon_density, edgecolors='k', lw=0.1, cmap='coolwarm')
    ax_recon.set_title(r"(d) Reconstruction $\nabla \times v_h$")
    ax_recon.set_aspect('equal')
    fig.colorbar(pc4, ax=ax_recon, label="Value")

    # plt.suptitle("Figure 10: Discrete Bogovskii Operator on a Star-Shaped Domain", fontsize=16, fontweight='bold')
    
    # Save the figure in high resolution
    plt.savefig("figure10_bogovskii.pdf", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_experiment()