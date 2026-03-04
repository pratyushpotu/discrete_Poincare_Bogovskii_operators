import numpy as np
from discrete_poincare.geometry import (
    polygon_area_signed,
    bbox_overlap,
    triangle_intersection_polygon,
    find_line_triangle_intersections_robust
)
from discrete_poincare.fields import (
    create_nedelec_field_evaluator_cached,
    compute_line_integral_with_splits
)

def compute_P1_Whitney(verts, edges, tris, u1, edge_map, star):
    verts, star = np.asarray(verts), np.asarray(star)
    field = create_nedelec_field_evaluator_cached(verts, tris, u1, edges, edge_map)
    vals = np.zeros(len(verts))
    for i, v in enumerate(verts):
        splits = find_line_triangle_intersections_robust(star, v, verts, tris)
        vals[i] = compute_line_integral_with_splits(field, star, v, splits)
    return vals

def compute_P2_Whitney(verts, edges, tris, u2, star):
    verts, star = np.asarray(verts), np.asarray(star)
    t_pts = verts[tris]
    # Vectorized Area & Density
    v0, v1, v2 = t_pts[:,0], t_pts[:,1], t_pts[:,2]
    areas = 0.5 * ((v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1]) - (v2[:,0]-v0[:,0])*(v1[:,1]-v0[:,1]))
    densities = np.divide(u2, areas, out=np.zeros_like(u2), where=abs(areas)>1e-14)

    vals = np.zeros(len(edges))
    for i, (i0, i1) in enumerate(edges):
        v0, v1 = verts[i0], verts[i1]
        star_tri = np.array([star, v0, v1])
        val = 0.0
        for j, tri in enumerate(t_pts):
            if bbox_overlap(star_tri, tri):
                poly = triangle_intersection_polygon(star_tri, tri)
                if len(poly) >= 3: val += densities[j] * polygon_area_signed(poly)
        vals[i] = val
    return vals