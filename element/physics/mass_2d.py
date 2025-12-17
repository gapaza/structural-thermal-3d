"""Code for getting the local mass matrix for a 2D Quad4 element."""
import numpy as np

def get_mechanical_mass(rho: float, lx: float = 1.0, ly: float = 1.0, thick: float = 1.0):
    """
    Builds 8x8 consistent mass matrix (Quad4) - Plane Stress.
    Integral( rho * N.T * N ) * t * dA

    Nodes: 1(-1,-1), 2(1,-1), 3(1,1), 4(-1,1)
    """
    # --- Gauss Points (2x2) ---
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])

    # --- Node Definitions ---
    xi_nodes = np.array([-1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1])

    # --- Jacobian & Geometry (Rectangular Element) ---
    # For a rectangular element, detJ is constant
    detJ = (lx * ly) / 4.0  # Area / 4

    me = np.zeros((8, 8))

    # --- Integration ---
    for xi in gp:
        for eta in gp:
            # Shape Functions N (Vector of size 4)
            # N_i = 0.25 * (1 + xi*xi_node) * (1 + eta*eta_node)
            N_vec = 0.25 * (1 + xi_nodes * xi) * (1 + et_nodes * eta)

            # N Matrix (2x8) - organizes N_vec into x and y DOFs
            # Rows: 0->u (x-displacement), 1->v (y-displacement)
            N = np.zeros((2, 8))

            # Map shape functions to x-dofs (columns 0, 2, 4, 6)
            N[0, 0::2] = N_vec
            # Map shape functions to y-dofs (columns 1, 3, 5, 7)
            N[1, 1::2] = N_vec

            # Accumulate me (rho * N.T * N * detJ * thickness)
            me += (N.T @ N) * rho * detJ * thick

    return me


if __name__ == "__main__":
    # Simple test
    rho = 7800.0  # kg/m^3 (Steel)

    me_mech = get_mechanical_mass(rho)
    print("Mechanical Mass Matrix (me):")
    print(me_mech.tolist())







