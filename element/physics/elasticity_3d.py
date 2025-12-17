import numpy as np

def get_mechanical_stiffness(nu: float, E: float, lx: float = 1.0, ly: float = 1.0, lz: float = 1.0):
    """
    Builds 24x24 mechanical stiffness matrix (Hex8).
    Integral( B.T * D * B ) dV
    """
    # --- Gauss Points & Geometry Setup ---
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])
    xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    ze_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    # Jacobian terms
    invJ = np.diag([2.0 / lx, 2.0 / ly, 2.0 / lz])
    detJ = (lx * ly * lz) / 8.0

    # --- Material Matrix (D) ---
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[range(3), range(3)] += 2 * mu
    D[3:, 3:] = np.diag([mu, mu, mu])

    ke = np.zeros((24, 24))

    # --- Integration ---
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                # Derivatives dN/dxi (8x3)
                dN_dxi = np.zeros((8, 3))
                dN_dxi[:, 0] = 0.125 * xi_nodes * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)
                dN_dxi[:, 1] = 0.125 * et_nodes * (1 + xi_nodes * xi) * (1 + ze_nodes * zeta)
                dN_dxi[:, 2] = 0.125 * ze_nodes * (1 + xi_nodes * xi) * (1 + et_nodes * eta)

                # Physical derivatives dN/dx (8x3)
                dN_dx = dN_dxi @ invJ

                # B Matrix (6x24)
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

                # Accumulate ke
                ke += (B.T @ D @ B) * detJ

    return ke

if __name__ == "__main__":
    # Example usage
    nu = 0.3
    E = 210e9  # Pa

    ke = get_mechanical_stiffness(nu, E)
    print("Mechanical Stiffness Matrix (ke):")
    print(ke)