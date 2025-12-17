"""This module contains the function to build the element stiffness matrix, conductivity matrix, and coupling matrix for thermal expansion."""

import numpy as np

#  _   _                  _    ____            _            _
# | | | |  __ _  _ __  __| |  / ___| ___    __| |  ___   __| |
# | |_| | / _` || '__|/ _` | | |    / _ \  / _` | / _ \ / _` |
# |  _  || (_| || |  | (_| | | |___| (_) || (_| ||  __/| (_| |
# |_| |_| \__,_||_|   \__,_|  \____|\___/  \__,_| \___| \__,_|
#


def get_stiffness_matrices(nu: float, e: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function builds the element stiffness matrix, conductivity matrix, and coupling matrix for thermal expansion.

    Args:
        nu (float): Poisson's ratio.
        e (float): Young's modulus (modulus of elasticity).
        k (float): Thermal conductivity.
        alpha (float): Coefficient of thermal expansion.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - KE (np.ndarray): Element stiffness matrix
            - KEth (np.ndarray): Element conductivity matrix
            - CEthm (np.ndarray): Element coupling matrix (thermal expansion)
    """
    # Construct element stiffness matrix
    kel = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )

    indices = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ]

    ke = (e / (1 - nu**2)) * kel[indices]

    # Construct element conductivity matrix
    k_eth = (k / 6) * np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]])

    # Element coupling matrix (thermal expansion)
    c_ethm = (e * alpha / (6 * (1 - nu))) * np.array(
        [
            [-2, -2, -1, -1],
            [-2, -1, -1, -2],
            [2, 2, 1, 1],
            [-1, -2, -2, -1],
            [1, 1, 2, 2],
            [1, 2, 2, 1],
            [-1, -1, -2, -2],
            [2, 1, 1, 2],
        ]
    )
    c_ethm = c_ethm / 2.0

    return ke, k_eth, c_ethm

#   ____                           _
#  / ___|  ___  _ __    ___  _ __ (_)  ___
# | |  _  / _ \| '_ \  / _ \| '__|| | / __|
# | |_| ||  __/| | | ||  __/| |   | || (__
#  \____| \___||_| |_| \___||_|   |_| \___|
#

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


def get_thermal_stiffness(k: float, lx: float = 1.0, ly: float = 1.0, thick: float = 1.0):
    """
    Builds 4x4 thermal conductivity matrix (Quad4).
    Integral( dN_dx.T * k * dN_dx ) * t * dA
    """
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])

    xi_nodes = np.array([-1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1])

    invJ = np.diag([2.0 / lx, 2.0 / ly])
    detJ = (lx * ly) / 4.0

    k_eth = np.zeros((4, 4))

    for xi in gp:
        for eta in gp:
            dN_dxi = np.zeros((4, 2))
            dN_dxi[:, 0] = 0.25 * xi_nodes * (1 + et_nodes * eta)
            dN_dxi[:, 1] = 0.25 * et_nodes * (1 + xi_nodes * xi)

            # dN/dx is (4x2)
            dN_dx = dN_dxi @ invJ

            # Accumulate k_eth (include thickness)
            k_eth += (dN_dx @ dN_dx.T) * k * detJ * thick

    return k_eth


def get_coupling_matrix(nu: float, E: float, alpha: float, lx: float = 1.0, ly: float = 1.0, thick: float = 1.0):
    """
    Builds 8x4 coupling matrix (Quad4).
    Maps Nodal Temps (4) -> Mechanical Forces (8).
    """
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])

    xi_nodes = np.array([-1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1])

    invJ = np.diag([2.0 / lx, 2.0 / ly])
    detJ = (lx * ly) / 4.0

    # --- Material Stress Vector (beta) ---
    # Plane Stress D-Matrix
    factor = E / (1 - nu ** 2)
    D = np.zeros((3, 3))
    D[0, 0] = 1.0;
    D[1, 1] = 1.0
    D[0, 1] = nu;
    D[1, 0] = nu
    D[2, 2] = (1 - nu) / 2.0
    D *= factor

    # Thermal Strain Unit Vector (Plane Stress)
    # [1, 1, 0] * alpha (no shear strain from isotropic expansion)
    thermal_strain_unit = np.array([1, 1, 0]) * alpha

    # Beta = D * alpha_vec
    beta_vec = D @ thermal_strain_unit  # Shape (3,)

    c_ethm = np.zeros((8, 4))

    for xi in gp:
        for eta in gp:
            # Shape Functions (N)
            N = 0.25 * (1 + xi_nodes * xi) * (1 + et_nodes * eta)

            # Derivatives
            dN_dxi = np.zeros((4, 2))
            dN_dxi[:, 0] = 0.25 * xi_nodes * (1 + et_nodes * eta)
            dN_dxi[:, 1] = 0.25 * et_nodes * (1 + xi_nodes * xi)
            dN_dx = dN_dxi @ invJ

            # B Matrix (3x8)
            B = np.zeros((3, 8))
            B[0, 0::2] = dN_dx[:, 0]
            B[1, 1::2] = dN_dx[:, 1]
            B[2, 0::2] = dN_dx[:, 1]
            B[2, 1::2] = dN_dx[:, 0]

            # Accumulate coupling (include thickness)
            # Contribution: B.T @ beta @ N
            f_thermal_unit = B.T @ beta_vec  # (8,)
            c_ethm += np.outer(f_thermal_unit, N) * detJ * thick

    return c_ethm


#  _____           _    _
# |_   _|___  ___ | |_ (_) _ __    __ _
#   | | / _ \/ __|| __|| || '_ \  / _` |
#   | ||  __/\__ \| |_ | || | | || (_| |
#   |_| \___||___/ \__||_||_| |_| \__, |
#                                 |___/






if __name__ == '__main__':
    # Standard steel-like parameters
    E_val = 210e9
    nu_val = 0.3
    k_val = 50.0
    alpha_val = 12e-6

    # Hard coded version
    ke, k_eth, c_ethm = get_stiffness_matrices(nu_val, E_val, k_val, alpha_val)

    # Generic version
    ke2 = get_mechanical_stiffness(nu_val, E_val)
    k_eth2 = get_thermal_stiffness(k_val)
    c_ethm2 = get_coupling_matrix(nu_val, E_val, alpha_val)

    # Printing
    print(ke.tolist())
    print(ke2.tolist())
    print(np.allclose(ke, ke2))
    print('-----------')
    print(k_eth.tolist())
    print(k_eth2.tolist())
    print(np.allclose(k_eth, k_eth2))
    print('-----------')
    print(c_ethm.tolist())
    print(c_ethm2.tolist())
    print(np.allclose(c_ethm, c_ethm2))


