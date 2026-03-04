import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix, csc_matrix

class SimplicialComplex:
    """
    A unified data structure for managing vertices, edges, and oriented faces 
    specifically optimized for combinatorial collapse sequences.
    """
    def __init__(self, vertices, triangles):
        self.vertices = np.asarray(vertices)
        self.oriented_triangles = np.asarray(triangles, dtype=int)
        self.faces = {tuple(sorted(f)) for f in self.oriented_triangles}
        
        self.edges = set()
        for v1, v2, v3 in self.faces:
            self.edges.add(tuple(sorted((v1, v2))))
            self.edges.add(tuple(sorted((v1, v3))))
            self.edges.add(tuple(sorted((v2, v3))))
            
        self.vertex_map = {i: i for i in range(len(vertices))}
        self.edge_map = {edge: i for i, edge in enumerate(sorted(list(self.edges)))}
        self.face_map = {tuple(sorted(t)): i for i, t in enumerate(self.oriented_triangles)}

    @property
    def num_vertices(self): 
        return len(self.vertices)
        
    @property
    def num_edges(self): 
        return len(self.edges)
        
    @property
    def num_faces(self): 
        return len(self.oriented_triangles)


def build_coboundary_operators(c: SimplicialComplex):
    """Builds geometrically consistent coboundary operators d0 and d1."""
    d0 = lil_matrix((c.num_edges, c.num_vertices), dtype=float)
    d1 = lil_matrix((c.num_faces, c.num_edges), dtype=float)

    for edge, i in c.edge_map.items():
        v1, v2 = edge  # v1 < v2 by construction
        d0[i, v1] = -1.0
        d0[i, v2] = 1.0

    # Use the geometrically-oriented triangles from the complex object
    for face_index, triangle in enumerate(c.oriented_triangles):
        v = triangle  # Original [i, j, k] from Delaunay, assumed CCW
        face_edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]  

        for edge_local in face_edges:
            v_start, v_end = edge_local
            edge_global = tuple(sorted(edge_local))
            global_edge_index = c.edge_map[edge_global]
            
            # Sign is +1 if local orientation matches global canonical (min_v -> max_v)
            orientation_sign = 1.0 if v_start < v_end else -1.0
            d1[face_index, global_edge_index] = orientation_sign

    return csc_matrix(d0), csc_matrix(d1)


def find_collapse_sequence(complex_obj: SimplicialComplex):
    """Computes a valid collapse sequence reducing the complex to a single vertex."""
    current_faces = set(complex_obj.faces)
    current_edges = set(complex_obj.edges)
    collapse_sequence = []
    
    # Phase 1: Collapse free edges incident to exactly one face
    while current_faces:
        edge_face_counts = defaultdict(int)
        for edge in current_edges:
            for face in current_faces:
                if set(edge).issubset(set(face)):
                    edge_face_counts[edge] += 1
                    
        free_edge = next((edge for edge, count in edge_face_counts.items() if count == 1), None)
        if free_edge is None: 
            raise RuntimeError("Cannot find a free edge. The complex may not be strongly collapsible.")
            
        principal_face = next(face for face in current_faces if set(free_edge).issubset(set(face)))
        collapse_sequence.append((principal_face, free_edge))
        
        current_faces.remove(principal_face)
        current_edges.remove(free_edge)
        
    remaining_vertices = {v for edge in current_edges for v in edge}
    
    # Phase 2: Collapse free vertices incident to exactly one edge
    while len(current_edges) > 0:
        if len(remaining_vertices) <= 1: 
            break
            
        vertex_degrees = defaultdict(int)
        for v1, v2 in current_edges: 
            vertex_degrees[v1] += 1
            vertex_degrees[v2] += 1
            
        free_vertex = next((v for v in remaining_vertices if vertex_degrees[v] == 1), None)
        if free_vertex is None:
            if len(remaining_vertices) == 2 and len(current_edges) == 1:
                free_vertex = list(remaining_vertices)[0]
            else: 
                raise RuntimeError("Cannot find leaf vertex during edge collapse phase.")
                
        principal_edge = next(edge for edge in current_edges if free_vertex in edge)
        collapse_sequence.append((principal_edge, free_vertex))
        
        current_edges.remove(principal_edge)
        remaining_vertices.remove(free_vertex)

    final_vertex = list(remaining_vertices)[0] if remaining_vertices else None
    return collapse_sequence, final_vertex


def get_oriented_boundary(c: SimplicialComplex, simplex):
    """Helper to get oriented boundary chains based on the complex's geometric orientation."""
    if len(simplex) == 3:  # Face boundary
        face_idx = c.face_map[simplex]
        v = c.oriented_triangles[face_idx]
        edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]
        boundary = defaultdict(float)
        for v_start, v_end in edges:
            edge_global = tuple(sorted((v_start, v_end)))
            sign = 1.0 if v_start < v_end else -1.0
            boundary[edge_global] += sign
        return boundary
        
    elif len(simplex) == 2:  # Edge boundary
        v1, v2 = simplex
        return {v1: -1.0, v2: 1.0}
        
    return {}


def build_collapse_poincare_operator(c: SimplicialComplex, collapse_sequence):
    """Constructs the discrete Poincaré cochain operators (P0, P1) from a collapse sequence."""
    
    # 1. First, build the chain homotopy operators h0 and h1
    # h0 maps vertices (0-chains) to edges (1-chains)
    h0 = lil_matrix((c.num_edges, c.num_vertices), dtype=float)
    # h1 maps edges (1-chains) to faces (2-chains)
    h1 = lil_matrix((c.num_faces, c.num_edges), dtype=float)
    
    for sigma, tau in reversed(collapse_sequence):
        if len(sigma) == 2:
            # Edge-Vertex collapse
            edge, vertex = sigma, tau
            edge_idx, vertex_idx = c.edge_map[edge], c.vertex_map[vertex]
            
            boundary = get_oriented_boundary(c, edge)
            c_tau = boundary.pop(vertex)
            
            col = lil_matrix((c.num_edges, 1), dtype=float)
            col[edge_idx, 0] = c_tau
            
            for other_v, sign in boundary.items():
                col -= sign * c_tau * h0[:, other_v]
            h0[:, vertex_idx] = col
            
        else:
            # Face-Edge collapse
            face, edge = sigma, tau
            face_idx, edge_idx = c.face_map[face], c.edge_map[edge]
            
            boundary = get_oriented_boundary(c, face)
            c_tau = boundary.pop(edge)
            
            col = lil_matrix((c.num_faces, 1), dtype=float)
            col[face_idx, 0] = c_tau
            
            for other_e, sign in boundary.items():
                col -= sign * c_tau * h1[:, c.edge_map[other_e]]
            h1[:, edge_idx] = col
            
    # 2. The discrete Poincaré cochain operators P^k are the transposes of h_k
    P0 = csc_matrix(h0).T  # Now maps edges (num_edges) -> vertices (num_vertices)
    P1 = csc_matrix(h1).T  # Now maps faces (num_faces) -> edges (num_edges)
    
    return P0, P1