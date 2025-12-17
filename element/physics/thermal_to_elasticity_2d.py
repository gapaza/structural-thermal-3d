import numpy as np

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

if __name__ == "__main__":
    # Example usage
    nu = 0.3      # Poisson's ratio
    E = 210e9     # Young's modulus in Pascals
    alpha = 1.2e-5  # Thermal expansion coefficient in 1/K

    c_ethm = get_coupling_matrix(nu, E, alpha)
    print("Coupling Matrix (c_ethm):")
    print(c_ethm)