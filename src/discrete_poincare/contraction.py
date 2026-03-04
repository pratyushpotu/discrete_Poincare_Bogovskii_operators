import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SimplicialMeshWrapper:
    vertices: np.ndarray
    simplices: np.ndarray

class DiscreteContraction:
    def __init__(self, mesh, maps: List[np.ndarray]):
        self.mesh = mesh
        self.maps = maps
        self.n_steps = len(maps) - 1

    def get_extrusion_prisms(self, sigma: Tuple[int, ...], step_idx: int):
        k = len(sigma) - 1
        prisms = []
        for j in range(k + 1):
            sign = (-1)**j
            prisms.append((sigma, j, sign))
        return prisms

    def evaluate_stack_map(self, sigma: Tuple[int, ...], j: int, step_idx: int):
        f_curr = self.maps[step_idx]
        f_next = self.maps[step_idx+1]
        v = sigma
        # Stack map definition: first j+1 verts from f_curr, rest from f_next
        left = [f_curr[v[m]] for m in range(j+1)]
        right = [f_next[v[m]] for m in range(j, len(v))]
        return tuple(left + right)

    def contract_chain(self, sigma: Tuple[int, ...]):
        chain = []
        for i in range(self.n_steps):
            prisms = self.get_extrusion_prisms(sigma, i)
            for (sig, j, sign) in prisms:
                mapped_verts = self.evaluate_stack_map(sig, j, i)
                chain.append((mapped_verts, sign))
        return chain

def generate_compatible_contraction(nx, ny, vertices):
    """
    Contracts to origin (0,0) by collapsing Y then X.
    Compatible with 'Bottom-Left to Top-Right' diagonals.
    """
    indices = np.arange(len(vertices))
    # Recover grid coordinates
    xs = indices % (nx + 1)
    ys = indices // (nx + 1)

    maps = []
    current_map = np.arange(len(vertices))
    maps.append(current_map.copy())

    # Phase 1: Collapse Rows (Y -> Y-1)
    for row in range(ny, 0, -1):
        next_map = current_map.copy()

        # Find all vertices currently mapped to 'row'
        target_ys = ys[current_map]
        active_indices = np.where(target_ys == row)[0]

        for idx in active_indices:
            # Map to same X, but Y-1
            # Index logic: idx - (nx+1) is the vertex directly below
            curr_target = current_map[idx]
            next_map[idx] = curr_target - (nx + 1)

        maps.append(next_map)
        current_map = next_map

    # Phase 2: Collapse Cols (X -> X-1)
    for col in range(nx, 0, -1):
        next_map = current_map.copy()

        target_xs = xs[current_map]
        active_indices = np.where(target_xs == col)[0]

        for idx in active_indices:
            # Map to same Y, but X-1
            curr_target = current_map[idx]
            next_map[idx] = curr_target - 1

        maps.append(next_map)
        current_map = next_map

    # Reverse to get [Constant ... Identity] sequence
    return maps[::-1]

def generate_u_mesh_contraction(level, vertices):
    nx = 3 * int(level)
    ny = 3 * int(level)

    # 1. Reconstruct grid coordinates for the unstructured vertices array
    grid_to_idx = {}
    idx_to_grid = {}
    for idx, (xx, yy) in enumerate(vertices):
        i = int(round(xx * nx))
        j = int(round(yy * ny))
        grid_to_idx[(i, j)] = idx
        idx_to_grid[idx] = (i, j)

    maps = []
    current_map = np.arange(len(vertices))
    maps.append(current_map.copy())

    # Phase 1: Collapse the Left and Right arms downwards
    # (From top row y=ny down to the top of the base y=ny//3)
    for row in range(ny, ny // 3, -1):
        next_map = current_map.copy()
        for idx in range(len(vertices)):
            curr_i, curr_j = idx_to_grid[current_map[idx]]
            if curr_j == row:
                # Map to the vertex directly below it
                target_idx = grid_to_idx[(curr_i, curr_j - 1)]
                next_map[idx] = target_idx
        maps.append(next_map)
        current_map = next_map

    # Phase 2: Collapse the solid base downwards to y=0
    for row in range(ny // 3, 0, -1):
        next_map = current_map.copy()
        for idx in range(len(vertices)):
            curr_i, curr_j = idx_to_grid[current_map[idx]]
            if curr_j == row:
                target_idx = grid_to_idx[(curr_i, curr_j - 1)]
                next_map[idx] = target_idx
        maps.append(next_map)
        current_map = next_map

    # Phase 3: Collapse the remaining line on the x-axis left to the origin (0,0)
    for col in range(nx, 0, -1):
        next_map = current_map.copy()
        for idx in range(len(vertices)):
            curr_i, curr_j = idx_to_grid[current_map[idx]]
            if curr_i == col:
                # Map to the vertex directly to its left
                target_idx = grid_to_idx[(curr_i - 1, curr_j)]
                next_map[idx] = target_idx
        maps.append(next_map)
        current_map = next_map

    # Reverse to get the [Constant ... Identity] sequence required by DiscreteContraction
    return maps[::-1]