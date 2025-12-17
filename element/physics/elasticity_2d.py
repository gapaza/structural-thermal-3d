import numpy as np

def get_mechanical_stiffness(nu: float, E: float, lx: float = 1.0, ly: float = 1.0, thick: float = 1.0):
    """
    Builds 8x8 mechanical stiffness matrix (Quad4) - Plane Stress.
    Integral( B.T * D * B ) * t * dA

    Nodes: 1(-1,-1), 2(1,-1), 3(1,1), 4(-1,1)
    """
    # --- Gauss Points (2x2) ---
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])

    # --- Node Definitions ---
    xi_nodes = np.array([-1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1])

    # --- Jacobian & Geometry ---
    # dx/dxi = lx/2 => invJ = 2/lx
    invJ = np.diag([2.0 / lx, 2.0 / ly])
    detJ = (lx * ly) / 4.0  # Area / 4

    # --- Material Matrix (Plane Stress) ---
    # D for Plane Stress (sigma_zz = 0)
    factor = E / (1 - nu ** 2)
    D = np.zeros((3, 3))
    D[0, 0] = 1.0
    D[1, 1] = 1.0
    D[0, 1] = nu
    D[1, 0] = nu
    D[2, 2] = (1 - nu) / 2.0
    D *= factor

    ke = np.zeros((8, 8))

    # --- Integration ---
    for xi in gp:
        for eta in gp:
            # Derivatives dN/dxi (4x2)
            # col 0: dN/dxi, col 1: dN/deta
            dN_dxi = np.zeros((4, 2))
            dN_dxi[:, 0] = 0.25 * xi_nodes * (1 + et_nodes * eta)
            dN_dxi[:, 1] = 0.25 * et_nodes * (1 + xi_nodes * xi)

            # Physical derivatives dN/dx (4x2)
            dN_dx = dN_dxi @ invJ

            # B Matrix (3x8) - Plane Stress (xx, yy, xy)
            B = np.zeros((3, 8))
            # xx strain (d/dx)
            B[0, 0::2] = dN_dx[:, 0]
            # yy strain (d/dy)
            B[1, 1::2] = dN_dx[:, 1]
            # xy strain (d/dy, d/dx)
            B[2, 0::2] = dN_dx[:, 1]
            B[2, 1::2] = dN_dx[:, 0]

            # Accumulate ke (include thickness)
            ke += (B.T @ D @ B) * detJ * thick

    return ke

if __name__ == "__main__":
    # Simple test
    nu = 0.3
    E = 210e9  # Pa

    ke_mech = get_mechanical_stiffness(nu, E)
    print("Mechanical Stiffness Matrix (ke):")
    print(ke_mech)