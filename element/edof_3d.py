import numpy as np


""" NOTES:

iK are rows, and rows repeat
jK are cols, and cols tile

The number of dof per node is a parameter.

Nominal CCW node ordering for hex8 elements (CCW top face -> CCW bottom face):

Top Face:
n4 -------- n3
 |           |
n1 -------- n2
        
Bottom Face:
n8 -------- n7
 |           |
n5 -------- n6

Element Ordering
- ely: Fastest
- elx: Medium
- elz: Slowest

"""

#  _   _                  _    ____            _            _
# | | | |  __ _  _ __  __| |  / ___| ___    __| |  ___   __| |
# | |_| | / _` || '__|/ _` | | |    / _ \  / _` | / _ \ / _` |
# |  _  || (_| || |  | (_| | | |___| (_) || (_| ||  __/| (_| |
# |_| |_| \__,_||_|   \__,_|  \____|\___/  \__,_| \___| \__,_|
#

def _build_edof_matrices(nelx, nely, nelz):
    num_elems = nelx * nely * nelz

    edofMat_mech2 = np.zeros((num_elems, 24), dtype=int)
    edofMat_therm2 = np.zeros((num_elems, 8), dtype=int)

    for elz in range(nelz):
        for elx in range(nelx):
            for ely in range(nely):
                el = ely + (elx * nely) + (elz * (nelx * nely))

                # CCW Top Face
                n1 = ((ely + 1) + elx * (nely + 1))       + (elz * ((nelx + 1) * (nely + 1)))
                n2 = ((ely + 1) + (elx + 1) * (nely + 1)) + (elz * ((nelx + 1) * (nely + 1)))
                n3 = (ely + (elx + 1) * (nely + 1))       + (elz * ((nelx + 1) * (nely + 1)))
                n4 = (ely + elx * (nely + 1))             + (elz * ((nelx + 1) * (nely + 1)))

                # CCW Bottom Face
                n5 = ((ely + 1) + elx * (nely + 1))       + ((elz + 1) * ((nelx + 1) * (nely + 1)))
                n6 = ((ely + 1) + (elx + 1) * (nely + 1)) + ((elz + 1) * ((nelx + 1) * (nely + 1)))
                n7 = (ely + (elx + 1) * (nely + 1))       + ((elz + 1) * ((nelx + 1) * (nely + 1)))
                n8 = (ely + elx * (nely + 1))             + ((elz + 1) * ((nelx + 1) * (nely + 1)))

                # print('\t', n1, n2, n3, n4, n5, n6, n7, n8)

                edofMat_therm2[el, :] = [n1, n2, n3, n4, n5, n6, n7, n8]

                edofMat_mech2[el, :] = [
                    3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                    3 * n2, 3 * n2 + 1, 3 * n2 + 2,
                    3 * n3, 3 * n3 + 1, 3 * n3 + 2,
                    3 * n4, 3 * n4 + 1, 3 * n4 + 2,
                    3 * n5, 3 * n5 + 1, 3 * n5 + 2,
                    3 * n6, 3 * n6 + 1, 3 * n6 + 2,
                    3 * n7, 3 * n7 + 1, 3 * n7 + 2,
                    3 * n8, 3 * n8 + 1, 3 * n8 + 2,
                ]

    iK_mech2 = np.kron(edofMat_mech2, np.ones((1, 24))).flatten()
    jK_mech2 = np.kron(edofMat_mech2, np.ones((24, 1))).flatten()
    iK_therm2 = np.kron(edofMat_therm2, np.ones((1, 8))).flatten()
    jK_therm2 = np.kron(edofMat_therm2, np.ones((8, 1))).flatten()

    return iK_mech2, jK_mech2, iK_therm2, jK_therm2

def _build_edof_matrices_vectorized(nelx, nely, nelz):
    """
    Precompute DOF indices for efficiency (Vectorized for 3D).
    Generates:
       - edofMat_mech: (n_elem, 24) matrix of mechanical DOFs
       - edofMat_therm: (n_elem, 8) matrix of thermal DOFs
       - Indexing vectors (iK, jK) for assembly (rows repeat, columns tile)
    """
    # 1. Coordinate Grids
    # "Fastest" dimension is Y (rows), then X (cols), then Z (depth)
    # We use meshgrid with 'F' (Fortran) to mimic the column-major logic of Top3d
    nelx, nely, nelz = nelx, nely, nelz

    # Generates grid of element indices
    elx, ely, elz = np.meshgrid(range(nelx), range(nely), range(nelz), indexing='xy')
    elx = elx.flatten(order='F')
    ely = ely.flatten(order='F')
    elz = elz.flatten(order='F')

    num_elems = len(elx)

    # 2. Node Mapping
    # Node strides
    ny_n = nely + 1
    nx_n = nelx + 1

    # Calculate the "n1" (Bottom-Left-Back) node for every element at once
    # n = y + x*ny_n + z*ny_n*nx_n
    n1 = (ely + 1) + elx * ny_n + elz * ny_n * nx_n
    n2 = (ely + 1) + (elx + 1) * ny_n + elz * ny_n * nx_n
    n3 = ely + (elx + 1) * ny_n + elz * ny_n * nx_n
    n4 = ely + elx * ny_n + elz * ny_n * nx_n

    # Shift to front face (z+1) is just adding the slice stride
    slice_stride = ny_n * nx_n
    n5 = n1 + slice_stride
    n6 = n2 + slice_stride
    n7 = n3 + slice_stride
    n8 = n4 + slice_stride

    # 3. Thermal DOF Matrix (8 Nodes per element, 1 DOF per node)
    # Shape: (num_elems, 8)
    edofMat_therm = np.stack([n1, n2, n3, n4, n5, n6, n7, n8], axis=1).astype(int)

    # 4. Mechanical DOF Matrix (8 Nodes, 3 DOFs per node)
    # Shape: (num_elems, 24)
    # DOFs are: 3*n, 3*n+1, 3*n+2
    edofMat_mech = np.zeros((num_elems, 24), dtype=int)

    # Vectorized expansion of 3 DOFs per node
    # We repeat the node IDs 3 times and add [0, 1, 2] offsets
    base_nodes = edofMat_therm  # (N, 8)

    # Map: [n1_u, n1_v, n1_w, n2_u, ...]
    # This loop is small (runs 8 times), so it's fine.
    for i in range(8):
        node_col = base_nodes[:, i]
        edofMat_mech[:, 3 * i + 0] = 3 * node_col  # u
        edofMat_mech[:, 3 * i + 1] = 3 * node_col + 1  # v
        edofMat_mech[:, 3 * i + 2] = 3 * node_col + 2  # w

    # 5. Assembly Index Vectors
    # K_mech (24x24)
    iK_mech = np.repeat(edofMat_mech, 24, axis=1).flatten()
    jK_mech = np.tile(edofMat_mech, (1, 24)).flatten()

    # K_therm (8x8)
    iK_therm = np.repeat(edofMat_therm, 8, axis=1).flatten()
    jK_therm = np.tile(edofMat_therm, (1, 8)).flatten()

    # Coupling Matrix C_ethm (24x8) - RECTANGULAR
    # Rows (i) are Mechanical DOFs (24), Cols (j) are Thermal DOFs (8)
    # Logic: For every element, we have a 24x8 block.
    # i must repeat each mech DOF 8 times: [m1...m1 (8 times), m2...m2 (8 times)]
    # j must tile the thermal DOFs 24 times: [t1..t8, t1..t8...]
    iK_coup = np.repeat(edofMat_mech, 8, axis=1).flatten()
    jK_coup = np.tile(edofMat_therm, (1, 24)).flatten()

    return iK_mech, jK_mech, iK_therm, jK_therm, iK_coup, jK_coup

#   ____                           _
#  / ___|  ___  _ __    ___  _ __ (_)  ___
# | |  _  / _ \| '_ \  / _ \| '__|| | / __|
# | |_| ||  __/| | | ||  __/| |   | || (__
#  \____| \___||_| |_| \___||_|   |_| \___|
#

def build_edof_matrices_generic(nelx, nely, nelz, dof_per_node):
    """
    Generates DOF indices for 3D topology optimization with variable DOFs per node.

    Parameters:
      nelx, nely, nelz: Grid dimensions.
      dof_per_node:     Degrees of freedom per node (e.g., 3 for Elasticity, 1 for Thermal).

    Returns:
      iK, jK:  Indexing vectors for sparse matrix assembly.
      edofMat: Element DOF matrix (n_elem, 8 * dof_per_node).
    """
    # 1. Coordinate Grids
    # "Fastest" dimension is Y (rows), then X (cols), then Z (depth)
    elx, ely, elz = np.meshgrid(range(nelx), range(nely), range(nelz), indexing='xy')
    elx = elx.flatten(order='F')
    ely = ely.flatten(order='F')
    elz = elz.flatten(order='F')

    num_elems = len(elx)

    # 2. Node Mapping
    # Node strides (nodes per row/column)
    ny_n = nely + 1
    nx_n = nelx + 1

    # Calculate the 8 corner nodes for every element
    # Base nodes logic (same as original)
    n1 = (ely + 1) + elx * ny_n + elz * ny_n * nx_n
    n2 = (ely + 1) + (elx + 1) * ny_n + elz * ny_n * nx_n
    n3 = ely + (elx + 1) * ny_n + elz * ny_n * nx_n
    n4 = ely + elx * ny_n + elz * ny_n * nx_n

    slice_stride = ny_n * nx_n
    n5 = n1 + slice_stride
    n6 = n2 + slice_stride
    n7 = n3 + slice_stride
    n8 = n4 + slice_stride

    # Stack into (num_elems, 8) base node matrix
    # Order: [n1, n2, n3, n4, n5, n6, n7, n8]
    base_nodes = np.stack([n1, n2, n3, n4, n5, n6, n7, n8], axis=1).astype(int)

    # 3. DOF Expansion (Vectorized)
    # We want to transform node indices N into [N*dof, N*dof+1, ..., N*dof+(dof-1)]

    # Expand base_nodes to (num_elems, 8, 1) for broadcasting
    nodes_expanded = base_nodes[:, :, None]

    # Create offset vector [0, 1, ..., dof-1]
    offsets = np.arange(dof_per_node)[None, None, :]

    # Broadcast: (NodeID * dof_per_node) + offset
    # Result shape: (num_elems, 8, dof_per_node)
    edof_3d = (nodes_expanded * dof_per_node) + offsets

    # Flatten the last two dimensions to get the standard edofMat
    # New shape: (num_elems, 8 * dof_per_node)
    # The order corresponds to: Node1_dof1, Node1_dof2... Node2_dof1...
    edofMat = edof_3d.reshape(num_elems, -1)

    # 4. Assembly Index Vectors
    # Total DOFs per element
    n_dof_element = 8 * dof_per_node

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
    nelx, nely, nelz = 2, 2, 2

    # Hard coded loop version
    # iK_mech2, jK_mech2, iK_therm2, jK_therm2 = _build_edof_matrices(nelx, nely, nelz)
    # print(iK_mech2.tolist())

    # Hard coded vectorized version
    iK_mech, jK_mech, iK_therm, jK_therm, iK_coup, jK_coup = _build_edof_matrices_vectorized(nelx, nely, nelz)


    # Generic vectorized version
    iK_mech2, jK_mech2, edof_Mech = build_edof_matrices_generic(nelx, nely, nelz, dof_per_node=3)
    iK_therm2, jK_therm2, edof_Therm = build_edof_matrices_generic(nelx, nely, nelz, dof_per_node=1)
    iK_coup2, jK_coup2 = build_coupling_indices(edof_Mech, edof_Therm)

    # --- Comparison ---
    print(iK_mech.tolist())
    print(iK_mech2.tolist())
    print('-------------------')
    print(iK_therm.tolist())
    print(iK_therm2.tolist())
    print('-------------------')
    print(iK_coup.tolist())
    print(iK_coup2.tolist())














