import numpy as np

def generate_diagonal_aligned_mesh(nx, ny):
    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)
    xx, yy = np.meshgrid(x, y)
    vertices = np.vstack([xx.ravel(), yy.ravel()]).T

    triangles = []
    for i in range(ny):
        for j in range(nx):
            # i is row (y), j is col (x)
            v0 = i * (nx + 1) + j           # Bottom-Left
            v1 = i * (nx + 1) + (j + 1)     # Bottom-Right
            v2 = (i + 1) * (nx + 1) + j     # Top-Left
            v3 = (i + 1) * (nx + 1) + (j + 1) # Top-Right

            # Use '\' diagonal: [v0, v3]
            # T1: Bottom Triangle [v0, v1, v3]
            triangles.append([v0, v1, v3])
            # T2: Top Triangle [v0, v3, v2]
            triangles.append([v0, v3, v2])

    return vertices, np.array(triangles)

def generate_u_mesh(level=1):
    nx = 3 * int(level)
    ny = 3 * int(level)

    hole_x_min = 1.0 / 3.0
    hole_x_max = 2.0 / 3.0
    hole_y_min = 1.0 / 3.0

    x = np.linspace(0, 1, nx + 1)
    y = np.linspace(0, 1, ny + 1)

    grid_to_idx = {}
    vertices = []

    # 1. Generate Vertices
    idx_counter = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            xx, yy = x[i], y[j]
            eps = 1e-9
            in_hole = (xx > hole_x_min + eps) and \
                      (xx < hole_x_max - eps) and \
                      (yy > hole_y_min + eps)

            if not in_hole:
                vertices.append([xx, yy])
                grid_to_idx[(i, j)] = idx_counter
                idx_counter += 1

    # Convert to NumPy array AFTER the vertex generation loop is completely finished
    vertices = np.array(vertices)
    triangles = []

    # 2. Generate Triangles
    for j in range(ny):
        for i in range(nx):
            cx = (x[i] + x[i+1]) / 2.0
            cy = (y[j] + y[j+1]) / 2.0

            in_hole_cell = (cx > hole_x_min) and \
                           (cx < hole_x_max) and \
                           (cy > hole_y_min)

            if not in_hole_cell:
                p00 = grid_to_idx.get((i, j))
                p10 = grid_to_idx.get((i+1, j))
                p11 = grid_to_idx.get((i+1, j+1))
                p01 = grid_to_idx.get((i, j+1))

                if all(p is not None for p in [p00, p10, p11, p01]):
                    triangles.append([p00, p10, p01])
                    triangles.append([p10, p11, p01])

    return vertices, np.array(triangles)

def extract_edges(triangles):
    """Extracts unique oriented edges from triangles."""
    edges_set = set()
    for t in triangles:
        # Sort to ensure uniqueness of undirected edge
        edges_set.add(tuple(sorted((t[0], t[1]))))
        edges_set.add(tuple(sorted((t[1], t[2]))))
        edges_set.add(tuple(sorted((t[2], t[0]))))
    edges_list = sorted(list(edges_set))
    edge_to_index = {edge: i for i, edge in enumerate(edges_list)}
    return edges_list, edge_to_index