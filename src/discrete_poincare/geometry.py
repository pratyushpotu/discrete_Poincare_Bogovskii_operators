import numpy as np

def polygon_area_signed(poly):
    if poly is None or len(poly) < 3: return 0.0
    p = np.asarray(poly)
    return 0.5 * (np.dot(p[:,0], np.roll(p[:,1], -1)) - np.dot(p[:,1], np.roll(p[:,0], -1)))

def bbox_overlap(poly_pts, tri_pts):
    A, B = np.asarray(poly_pts), np.asarray(tri_pts)
    return not np.any(A.max(0) < B.min(0)) and not np.any(B.max(0) < A.min(0))

def line_intersection(p1, p2, q1, q2):
    p1, p2, q1, q2 = map(np.asarray, [p1, p2, q1, q2])
    r, s = p2 - p1, q2 - q1
    det = r[0]*s[1] - r[1]*s[0]
    if abs(det) < 1e-14: return None
    t = ((q1[0]-p1[0])*s[1] - (q1[1]-p1[1])*s[0]) / det # Cross product division
    return p1 + t * r

def clip_polygon_against_edge(poly, a, b):
    if not poly: return []
    out, pts = [], list(poly)
    is_left = lambda p: ((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])) >= -1e-12
    for i in range(len(pts)):
        curr, prev = pts[i], pts[i-1]
        curr_in, prev_in = is_left(curr), is_left(prev)
        if curr_in:
            if not prev_in:
                ip = line_intersection(prev, curr, a, b)
                if ip is not None: out.append(tuple(ip))
            out.append(tuple(curr))
        elif prev_in:
            ip = line_intersection(prev, curr, a, b)
            if ip is not None: out.append(tuple(ip))
    return out

def triangle_intersection_polygon(t1, t2):
    poly = [tuple(p) for p in t1]
    for i in range(3):
        poly = clip_polygon_against_edge(poly, t2[i], t2[(i+1)%3])
        if not poly: return []
    return poly

def find_line_triangle_intersections_robust(start, end, verts, tris, tol=1e-12):
    direction = end - start
    if np.linalg.norm(direction) < tol: return [0.0, 1.0]

    params = {0.0, 1.0}
    for tri in tris:
        pts = verts[tri]
        for i in range(3):
            edge_vec = pts[(i+1)%3] - pts[i]
            A = np.column_stack([direction, -edge_vec])
            if abs(np.linalg.det(A)) > tol:
                try:
                    t, s = np.linalg.solve(A, pts[i] - start)
                    if tol <= t <= 1.0 - tol and 0.0 <= s <= 1.0: params.add(t)
                except np.linalg.LinAlgError: continue

    sorted_p = sorted(params)
    clean = [sorted_p[0]]
    for p in sorted_p[1:]:
        if p - clean[-1] > tol: clean.append(p)
    return clean

def get_barycentric_coords(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-10: return -1, -1, -1
    v = (d11 * np.dot(v2, v0) - d01 * np.dot(v2, v1)) / denom
    w = (d00 * np.dot(v2, v1) - d01 * np.dot(v2, v0)) / denom
    return 1.0 - v - w, v, w