"""
Discrete Poincaré and Bogovskii Operators
=========================================

A package for computing discrete Poincaré and Bogovskii operators on cochains 
and Whitney forms over simplicial complexes, corresponding to the methods 
outlined in the accompanying paper.
"""

from .mesh import (
    generate_diagonal_aligned_mesh,
    generate_u_mesh,
    extract_edges
)

from .topology import (
    calculate_d0_matrix,
    calculate_d1_matrix
)

from .contraction import (
    SimplicialMeshWrapper,
    DiscreteContraction,
    generate_compatible_contraction,
    generate_u_mesh_contraction 
)

from .combinatorial_operators import (
    compute_P1_combinatorial,
    compute_P2_combinatorial
)

from .Whitney_Poincare_operators import (
    compute_P1_Whitney,
    compute_P2_Whitney
)

from .Bogovskii_operators import (
    compute_B1_Whitney,
    compute_B2_Whitney
)

from .L_contraction_operators import (
    compute_P1_L_contraction,
    compute_P2_L_contraction
)

from .collapse_operators import (
    SimplicialComplex,
    build_coboundary_operators,
    find_collapse_sequence,
    build_collapse_poincare_operator
)

# Explicitly define what gets imported when someone uses `from discrete_poincare import *`
__all__ = [
    "generate_diagonal_aligned_mesh",
    "generate_u_mesh",
    "extract_edges",
    "calculate_d0_matrix",
    "calculate_d1_matrix",
    "SimplicialMeshWrapper",
    "DiscreteContraction",
    "generate_compatible_contraction",
    "generate_u_mesh_contraction",
    "compute_P1_combinatorial",
    "compute_P2_combinatorial",
    "compute_P1_Whitney",
    "compute_P2_Whitney",
    "compute_B1_Whitney",
    "compute_B2_Whitney",
    "compute_P1_L_contraction",
    "compute_P2_L_contraction",
    "SimplicialComplex",
    "build_coboundary_operators",
    "find_collapse_sequence",
    "build_collapse_poincare_operator"
]