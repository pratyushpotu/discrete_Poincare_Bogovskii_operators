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

def compute_P1_L_contraction(vertices, edges, triangles, u1_edge_values, edge_to_index, star_point_a):
    vertices = np.asarray(vertices)
    star_point_a = np.asarray(star_point_a)

    tol = 1e-12
    vector_field_func = create_nedelec_field_evaluator_cached(
        vertices, triangles, u1_edge_values, edges, edge_to_index, tol=tol)

    num_vertices = vertices.shape[0]
    poincare_values = np.zeros(num_vertices)

    for i in range(num_vertices):
        vertex = vertices[i]

        intermediate_point = np.array([vertex[0], star_point_a[1]])

        segments = []

        if np.linalg.norm(intermediate_point - star_point_a) > tol:
            segments.append((star_point_a, intermediate_point))

        if np.linalg.norm(vertex - intermediate_point) > tol:
            segments.append((intermediate_point, vertex))

        total_integral = 0.0

        for seg_start, seg_end in segments:
            split_params = find_line_triangle_intersections_robust(
                seg_start, seg_end, vertices, triangles)

            val = compute_line_integral_with_splits(
                vector_field_func, seg_start, seg_end, split_params)

            total_integral += val

        poincare_values[i] = total_integral

    return poincare_values

def compute_P2_L_contraction(vertices, edges, triangles, u2_values, star_point_a):
    vertices = np.asarray(vertices, dtype=float)
    star_point_a = np.asarray(star_point_a, dtype=float)
    y_base = star_point_a[1]

    tri_coords = vertices[triangles]

    v0, v1, v2 = tri_coords[:, 0], tri_coords[:, 1], tri_coords[:, 2]
    tri_areas = 0.5 * (v0[:, 0] * (v1[:, 1] - v2[:, 1]) +
                       v1[:, 0] * (v2[:, 1] - v0[:, 1]) +
                       v2[:, 0] * (v0[:, 1] - v1[:, 1]))

    u2_constants = np.zeros_like(u2_values, dtype=float)
    valid_mask = np.abs(tri_areas) > 1e-14
    u2_constants[valid_mask] = u2_values[valid_mask] / tri_areas[valid_mask]

    p2_values = np.zeros(len(edges), dtype=float)

    for i, edge in enumerate(edges):
        v_start = vertices[edge[0]]
        v_end = vertices[edge[1]]

        p1 = v_start
        p2 = v_end
        p3 = np.array([v_end[0], y_base])
        p4 = np.array([v_start[0], y_base])

        if abs(p1[0] - p2[0]) < 1e-12:
            p2_values[i] = 0.0
            continue

        trapezoid_tris = [
            np.array([p1, p2, p3]),
            np.array([p1, p3, p4])
        ]

        integral_sum = 0.0

        for trap_tri in trapezoid_tris:
            if abs(polygon_area_signed(trap_tri)) < 1e-14:
                continue

            for j, mesh_tri_pts in enumerate(tri_coords):
                if not bbox_overlap(trap_tri, mesh_tri_pts):
                    continue

                intersection_poly = triangle_intersection_polygon(trap_tri, mesh_tri_pts)

                if len(intersection_poly) >= 3:
                    area = polygon_area_signed(intersection_poly)
                    integral_sum += u2_constants[j] * area

        p2_values[i] = integral_sum

    return p2_values