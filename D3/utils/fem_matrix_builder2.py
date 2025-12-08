import numpy as np


def fe_melthm_3d_flexible(nu: float, E: float, lx: float = 1.0, ly: float = 1.0, lz: float = 1.0):
    """
    Args:
        lx, ly, lz: Physical dimensions of the element in x, y, z.
    """
    # --- Gauss points ---
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])

    # --- Node Definition ---
    xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    ze_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    # --- Elasticity Matrix ---
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam + 2 * mu * np.eye(3)
    D[:3, :3] -= 2 * mu * np.eye(3)  # Wait, clearer way below:

    # Reset D for clarity to match previous correct logic
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[range(3), range(3)] += 2 * mu
    D[3:, 3:] = np.diag([mu, mu, mu])

    # --- GEOMETRY MAPPING (The Update) ---
    # For a rectangular element of size lx, ly, lz mapped from [-1,1]:
    # x = xi * (lx/2)  --> dx/dxi = lx/2
    # J = diag(lx/2, ly/2, lz/2)
    # invJ = diag(2/lx, 2/ly, 2/lz)

    invJ = np.diag([2.0 / lx, 2.0 / ly, 2.0 / lz])
    detJ = (lx * ly * lz) / 8.0  # (lx/2)*(ly/2)*(lz/2)

    ke = np.zeros((24, 24))

    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN_dxi = np.zeros((8, 3))
                dN_dxi[:, 0] = 0.125 * xi_nodes * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)
                dN_dxi[:, 1] = 0.125 * et_nodes * (1 + xi_nodes * xi) * (1 + ze_nodes * zeta)
                dN_dxi[:, 2] = 0.125 * ze_nodes * (1 + xi_nodes * xi) * (1 + et_nodes * eta)

                # Map gradients to physical space
                dN_dx = dN_dxi @ invJ

                B = np.zeros((6, 24))
                B[0, 0::3] = dN_dx[:, 0]
                B[1, 1::3] = dN_dx[:, 1]
                B[2, 2::3] = dN_dx[:, 2]
                B[3, 0::3] = dN_dx[:, 1];
                B[3, 1::3] = dN_dx[:, 0]
                B[4, 1::3] = dN_dx[:, 2];
                B[4, 2::3] = dN_dx[:, 1]
                B[5, 0::3] = dN_dx[:, 2];
                B[5, 2::3] = dN_dx[:, 0]

                ke += (B.T @ D @ B) * detJ

    return ke