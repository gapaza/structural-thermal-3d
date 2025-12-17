import numpy as np

""" NOTES:

iK are rows, and rows repeat
jK are cols, and cols tile

The number of dof per node is a parameter.

Nominal CCW node ordering for hex4 elements:

n4 -------- n3
 |           |
n1 -------- n2


Element Ordering
- ely: Fastest
- elx: Slowest

"""

#  _   _                  _    ____            _            _
# | | | |  __ _  _ __  __| |  / ___| ___    __| |  ___   __| |
# | |_| | / _` || '__|/ _` | | |    / _ \  / _` | / _ \ / _` |
# |  _  || (_| || |  | (_| | | |___| (_) || (_| ||  __/| (_| |
# |_| |_| \__,_||_|   \__,_|  \____|\___/  \__,_| \___| \__,_|
#


def _build_edof_matrices(nelx, nely):
    """Compute DOF indices."""

    num_elems = nelx * nely

    edofMat_mech = np.zeros((num_elems, 8), dtype=int)
    edofMat_therm = np.zeros((num_elems, 4), dtype=int)

    for elx in range(nelx):
        for ely in range(nely):
            el = ely + elx * nely

            # CCW starting at bottom left node
            n1 = (ely + 1) + elx * (nely + 1)        # Bottom Left
            n2 = (ely + 1) + (elx + 1) * (nely + 1)  # Bottom Right
            n3 = ely + (elx + 1) * (nely + 1)        # Top Right
            n4 = ely + elx * (nely + 1)              # Top Left


            edofMat_therm[el, :] = [n1, n2, n3, n4]

            # Mechanical Nodes (Vector x,y)
            edofMat_mech[el, :] = [
                2 * n1, 2 * n1 + 1,
                2 * n2, 2 * n2 + 1,
                2 * n3, 2 * n3 + 1,
                2 * n4, 2 * n4 + 1
            ]

    iK_mech = np.kron(edofMat_mech, np.ones((1, 8))).flatten()
    jK_mech = np.kron(edofMat_mech, np.ones((8, 1))).flatten()
    iK_therm = np.kron(edofMat_therm, np.ones((1, 4))).flatten()
    jK_therm = np.kron(edofMat_therm, np.ones((4, 1))).flatten()

    return iK_mech, jK_mech, iK_therm, jK_therm

#   ____                           _
#  / ___|  ___  _ __    ___  _ __ (_)  ___
# | |  _  / _ \| '_ \  / _ \| '__|| | / __|
# | |_| ||  __/| | | ||  __/| |   | || (__
#  \____| \___||_| |_| \___||_|   |_| \___|
#

def build_edof_matrices_generic(nelx, nely, dof_per_node):
    """
    Generates DOF indices for 2D topology optimization with variable DOFs per node.
    Vectorized replacement for nested loops.

    Parameters:
      nelx, nely:   Grid dimensions.
      dof_per_node: Degrees of freedom per node (e.g., 2 for Mech, 1 for Therm).

    Returns:
      iK, jK:  Indexing vectors for sparse matrix assembly.
      edofMat: Element DOF matrix (n_elem, 4 * dof_per_node).
    """
    # 1. Coordinate Grids
    # Create element indices. order='F' ensures we match the column-major
    # element numbering (counting down Y first, then across X) used in your original loop.
    elx, ely = np.meshgrid(range(nelx), range(nely), indexing='xy')
    elx = elx.flatten(order='F')
    ely = ely.flatten(order='F')

    num_elems = len(elx)

    # 2. Node Calculation
    # Node mapping matches the original logic:
    # n1 (Bottom Left), n2 (Bottom Right), n3 (Top Right), n4 (Top Left)
    # Note: Indices here are 0-based relative to the node grid.
    ny_n = nely + 1  # Nodes in Y direction

    n1 = (ely + 1) + elx * ny_n
    n2 = (ely + 1) + (elx + 1) * ny_n
    n3 = ely + (elx + 1) * ny_n
    n4 = ely + elx * ny_n

    # Stack into base node matrix (num_elems, 4)
    # Order: [n1, n2, n3, n4]
    base_nodes = np.stack([n1, n2, n3, n4], axis=1).astype(int)

    # 3. DOF Expansion (Broadcasting)
    # Expand base_nodes to (num_elems, 4, 1)
    nodes_expanded = base_nodes[:, :, None]

    # Create offsets: [0, 1, ... dof-1]
    offsets = np.arange(dof_per_node)[None, None, :]

    # Calculate actual DOFs: (NodeID * dof) + offset
    # Result shape: (num_elems, 4, dof_per_node)
    edof_2d = (nodes_expanded * dof_per_node) + offsets

    # Flatten to (num_elems, 4 * dof_per_node)
    # This creates the pattern: [n1_u, n1_v, n2_u, n2_v ...] for dof=2
    edofMat = edof_2d.reshape(num_elems, -1)

    # 4. Assembly Index Vectors
    # Total DOFs per element
    n_dof_element = 4 * dof_per_node

    # Create assembly vectors (kron equivalent using repeat/tile)
    iK = np.repeat(edofMat, n_dof_element, axis=1).flatten()
    jK = np.tile(edofMat, (1, n_dof_element)).flatten()

    return iK, jK, edofMat

def build_coupling_indices(edofMat_target, edofMat_source):
    """
    Generates assembly indices for a coupling matrix that maps
    Source DOFs -> Target DOFs.

    Parameters:
      edofMat_target: (n_elem, N) matrix of Row indices (The "Output" physics)
      edofMat_source: (n_elem, M) matrix of Col indices (The "Input" physics)

    Returns:
      iK_coup: Row indices
      jK_coup: Col indices
    """
    # 1. Validation
    # Ensure both matrices correspond to the same number of elements
    assert edofMat_target.shape[0] == edofMat_source.shape[0], \
        "Mismatch in number of elements between target and source matrices."

    n_dof_target = edofMat_target.shape[1]  # e.g., 24 for Mech
    n_dof_source = edofMat_source.shape[1]  # e.g., 8 for Therm

    # 2. Vectorized Index Generation
    # For every element, we need a rectangular block of size (n_dof_target x n_dof_source)

    # ROWS (i): correspond to the Target.
    # We must repeat every target DOF 'n_source' times.
    # Ex: Target=[A, B], Source=[1, 2] -> Pairs: (A,1), (A,2), (B,1), (B,2)
    # Result i: [A, A, B, B]
    iK_coup = np.repeat(edofMat_target, n_dof_source, axis=1).flatten()

    # COLS (j): correspond to the Source.
    # We must tile the entire source list 'n_target' times.
    # Result j: [1, 2, 1, 2]
    jK_coup = np.tile(edofMat_source, (1, n_dof_target)).flatten()

    return iK_coup, jK_coup

#  _____           _    _
# |_   _|___  ___ | |_ (_) _ __    __ _
#   | | / _ \/ __|| __|| || '_ \  / _` |
#   | ||  __/\__ \| |_ | || | | || (_| |
#   |_| \___||___/ \__||_||_| |_| \__, |
#                                 |___/

if __name__ == '__main__':
    nelx, nely = 2, 2

    # Hard coded loop version
    iK_mech, jK_mech, iK_therm, jK_therm = _build_edof_matrices(nelx, nely)

    # Generic vectorized version
    iK_mech2, jK_mech2, edof_Mech = build_edof_matrices_generic(nelx, nely, dof_per_node=2)
    iK_therm2, jK_therm2, edof_Therm = build_edof_matrices_generic(nelx, nely, dof_per_node=1)
    iK_coup2, jK_coup2 = build_coupling_indices(edof_Mech, edof_Therm)

    print(iK_mech.tolist())
    print(iK_mech2.tolist())
    print('-------------------')
    print(iK_therm.tolist())
    print(iK_therm2.tolist())
    print('-------------------')
    # print(iK_coup.tolist())
    print(iK_coup2.tolist())













