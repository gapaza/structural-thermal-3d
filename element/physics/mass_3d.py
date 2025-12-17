"""Code for getting the local mass matrix for a 3D Hex8 element."""
import numpy as np


def get_mechanical_mass(rho: float, lx: float = 1.0, ly: float = 1.0, lz: float = 1.0):
    """
    Builds 24x24 consistent mass matrix (Hex8).
    Integral( rho * N.T * N ) dV

    Nodes: 1(-1,-1,-1) ... 8(-1,1,1)
    """
    # --- Gauss Points (2x2x2) ---
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])

    # --- Node Definitions (Standard Hex8) ---
    xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    ze_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    # --- Jacobian & Geometry ---
    # For a rectangular hex element, detJ is constant volume / 8
    detJ = (lx * ly * lz) / 8.0

    me = np.zeros((24, 24))

    # --- Integration ---
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                # Shape Functions N_vec (Vector of size 8)
                # N_i = 1/8 * (1 + xi*xi_node) * (1 + eta*eta_node) * (1 + zeta*zeta_node)
                N_vec = 0.125 * (1 + xi_nodes * xi) * \
                        (1 + et_nodes * eta) * \
                        (1 + ze_nodes * zeta)

                # N Matrix (3x24) - organizes N_vec into x, y, z DOFs
                # Rows: 0->u (x), 1->v (y), 2->w (z)
                N = np.zeros((3, 24))

                # Map shape functions to x-dofs (columns 0, 3, 6... 21)
                N[0, 0::3] = N_vec
                # Map shape functions to y-dofs (columns 1, 4, 7... 22)
                N[1, 1::3] = N_vec
                # Map shape functions to z-dofs (columns 2, 5, 8... 23)
                N[2, 2::3] = N_vec

                # Accumulate me (rho * N.T * N * detJ)
                me += (N.T @ N) * rho * detJ

    return me


if __name__ == "__main__":
    # Example usage
    rho = 7800.0  # kg/m^3 (Steel)

    me_hex = get_mechanical_mass(rho)
    print("3D Mechanical Mass Matrix (me):")
    print(me_hex.tolist())
    print(f"Shape: {me_hex.shape}")
