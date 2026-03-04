import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.integrate import quad

def plot_mesh(points, triangles, show_points=False, figsize=(6, 6), color='k', title="Mesh"):
    plt.figure(figsize=figsize)
    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    plt.triplot(tri, color=color, lw=0.8)
    if show_points:
        plt.plot(points[:, 0], points[:, 1], 'o', markersize=3)
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.show()

def project_to_nedelec(func, verts, edges):
    """
    Projects a continuous vector field onto the discrete Nédélec space.
    Used heavily in setting up the numerical experiments.
    """
    vals = np.zeros(len(edges))
    for i, (i0, i1) in enumerate(edges):
        p0, p1 = verts[i0], verts[i1]
        tan = p1 - p0
        vals[i] = quad(lambda t: np.dot(func(p0 + t*tan), tan), 0, 1)[0]
    return vals

def plot_solution(pts, tris, vals, title="Solution", cmap="viridis"):
    plt.figure(figsize=(5, 4))
    tri = mtri.Triangulation(pts[:,0], pts[:,1], tris)
    plt.tricontourf(tri, vals, levels=50, cmap=cmap)
    plt.colorbar()
    plt.triplot(tri, 'k-', lw=0.3, alpha=0.3)
    plt.title(title); plt.axis('scaled'); plt.show()

def plot_field_refined(func, xlim, ylim, N=20, title="", ax=None, mesh=None):
    """
    Plots a 2D vector field.
    mesh: Optional tuple (vertices, triangles) to overlay the mesh.
    """
    if ax is None: fig, ax = plt.subplots(figsize=(6, 6))

    # 1. Generate Grid and Evaluate
    x, y = np.linspace(xlim[0], xlim[1], N), np.linspace(ylim[0], ylim[1], N)
    X, Y = np.meshgrid(x, y)
    UV = np.array([func([X[i,j], Y[i,j]]) for i in range(N) for j in range(N)]).reshape(N, N, 2)
    U, V = UV[..., 0], UV[..., 1]

    # 2. Normalize vectors for uniform arrow length (color indicates magnitude)
    M = np.hypot(U, V)
    U_norm, V_norm = np.zeros_like(U), np.zeros_like(V)
    mask = M > 1e-12
    U_norm[mask], V_norm[mask] = U[mask] / M[mask], V[mask] / M[mask]

    # 3. Plot Quiver
    q = ax.quiver(X, Y, U_norm, V_norm, M, cmap='plasma', pivot='mid', width=0.005)

    # 4. Overlay Mesh (if provided)
    if mesh:
        pts, tris = mesh
        ax.triplot(pts[:, 0], pts[:, 1], tris, color='gray', lw=0.5, alpha=0.3)

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    return ax