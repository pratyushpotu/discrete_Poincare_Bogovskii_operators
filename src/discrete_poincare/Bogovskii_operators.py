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

def compute_B1_Whitney(verts, edges, tris, u1, edge_map, star):
    verts, star = np.asarray(verts), np.asarray(star)
    field = create_nedelec_field_evaluator_cached(verts, tris, u1, edges, edge_map)
    far_dist = np.linalg.norm(verts.max(0)-verts.min(0)) * 10.0
    vals = np.zeros(len(verts))

    for i, v in enumerate(verts):
        d = v - star
        dist = np.linalg.norm(d)

        if dist < 1e-12:
            far_pt = v + np.array([1.0, 0.0]) * far_dist
        else:
            far_pt = v + (d/dist) * far_dist

        splits = find_line_triangle_intersections_robust(v, far_pt, verts, tris, tol=1e-10)
        vals[i] = -compute_line_integral_with_splits(field, v, far_pt, splits)

    return vals

def compute_B2_Whitney(verts, edges, tris, u2, star):
    verts, star = np.asarray(verts), np.asarray(star)
    t_pts = verts[tris]
    v0, v1, v2 = t_pts[:,0], t_pts[:,1], t_pts[:,2]
    areas = 0.5 * ((v1[:,0]-v0[:,0])*(v2[:,1]-v0[:,1]) - (v2[:,0]-v0[:,0])*(v1[:,1]-v0[:,1]))
    densities = np.divide(u2, areas, out=np.zeros_like(u2), where=abs(areas)>1e-14)

    far_scale = np.linalg.norm(verts.max(0)-verts.min(0)) * 10.0
    vals = np.zeros(len(edges))

    for i, (i0, i1) in enumerate(edges):
        v0, v1 = verts[i0], verts[i1]
        d0, d1 = v0 - star, v1 - star
        n0, n1 = np.linalg.norm(d0), np.linalg.norm(d1)
        if n0 < 1e-12 or n1 < 1e-12: continue

        shadow = np.array([v0, v1, v1 + d1/n1*far_scale, v0 + d0/n0*far_scale])
        val = 0.0
        for j, tri in enumerate(t_pts):
            if bbox_overlap(shadow, tri):
                poly = triangle_intersection_polygon(shadow, tri)
                if len(poly) >= 3: val += densities[j] * polygon_area_signed(poly)
        vals[i] = val
    return vals