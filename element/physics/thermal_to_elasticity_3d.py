import numpy as np

def get_coupling_matrix(nu: float, E: float, alpha: float, lx: float = 1.0, ly: float = 1.0, lz: float = 1.0):
    """
    Builds 24x8 coupling matrix.
    Maps Nodal Temps (8) -> Mechanical Forces (24).
    """
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])
    xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    ze_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    invJ = np.diag([2.0 / lx, 2.0 / ly, 2.0 / lz])
    detJ = (lx * ly * lz) / 8.0

    # --- Material Stress Vector (beta) ---
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[range(3), range(3)] += 2 * mu
    D[3:, 3:] = np.diag([mu, mu, mu])

    thermal_strain_unit = np.array([1, 1, 1, 0, 0, 0]) * alpha
    beta_vec = D @ thermal_strain_unit

    c_ethm = np.zeros((24, 8))

    for xi in gp:
        for eta in gp:
            for zeta in gp:
                # Shape Functions (N) - Needed here for the coupling
                N = 0.125 * (1 + xi_nodes * xi) * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)

                # Derivatives
                dN_dxi = np.zeros((8, 3))
                dN_dxi[:, 0] = 0.125 * xi_nodes * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)
                dN_dxi[:, 1] = 0.125 * et_nodes * (1 + xi_nodes * xi) * (1 + ze_nodes * zeta)
                dN_dxi[:, 2] = 0.125 * ze_nodes * (1 + xi_nodes * xi) * (1 + et_nodes * eta)
                dN_dx = dN_dxi @ invJ

                # B Matrix
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

                # Accumulate coupling
                # Contribution: B.T @ beta @ N
                f_thermal_unit = B.T @ beta_vec  # (24,)
                c_ethm += np.outer(f_thermal_unit, N) * detJ

    return c_ethm

if __name__ == "__main__":
    # Example usage
    nu = 0.3
    E = 210e9  # Young's Modulus in Pascals
    alpha = 1.2e-5  # Coefficient of Thermal Expansion

    c_ethm = get_coupling_matrix(nu, E, alpha)
    print("Thermo-Mechanical Coupling Matrix (24x8):")
    print(c_ethm)