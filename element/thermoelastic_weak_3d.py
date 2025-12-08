import numpy as np


def get_stiffness_matrices(nu: float, E: float, k: float, alpha: float,
                 lx: float = 1.0, ly: float = 1.0, lz: float = 1.0):
    """
    Build 3D Hex8 element matrices for thermo-elasticity:
      - ke    : 24x24 mechanical stiffness (Hex8, 3 dof/node)
      - k_eth :  8x8 thermal conductivity (Hex8, 1 dof/node)
      - c_ethm: 24x8 coupling mapping nodal temperatures to mechanical forces

    Assumptions:
      - Isotropic linear elasticity and isotropic thermal conductivity.
      - Reference Hex8 element over [-1, 1]^3.
      - 2x2x2 Gauss integration.
      - Thermal strain = alpha * dT * [1, 1, 1, 0, 0, 0]^T (Voigt).

    Node Ordering (Standard TopOpt/Abaqus):
      - Bottom: 1(-,-,-), 2(+,-,-), 3(+,+,-), 4(-,+,-)
      - Top:    5(-,-,+), 6(+,-,+), 7(+,+,+), 8(-,+,+)

    Args:
        nu (float): Poisson's ratio
        E (float): Young's modulus
        k (float): Thermal conductivity (isotropic)
        alpha (float): Coefficient of thermal expansion
        lx, ly, lz: Physical dimensions of the element in x, y, z.

    Returns:
        (ke, k_eth, c_ethm)
    """

    # --- 1. Gauss points (2x2x2) ---
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])

    # --- 2. Node Definition (Natural Coords) ---
    xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    ze_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    # --- 3. Material Properties (Constitutive Matrix D) ---
    # Lame parameters
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    # Mechanical Constitutive Matrix (Voigt: xx, yy, zz, xy, yz, zx)
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[range(3), range(3)] += 2 * mu
    D[3:, 3:] = np.diag([mu, mu, mu])

    # Thermal Stress Coefficient Vector (beta)
    # Stress induced by unit temperature change: sigma_th = D @ (alpha * [1,1,1,0,0,0].T)
    # For isotropic material: sigma = (3*lambda + 2*mu) * alpha * I
    thermal_strain_unit = np.array([1, 1, 1, 0, 0, 0]) * alpha
    beta_vec = D @ thermal_strain_unit  # Shape (6,)

    # --- 4. Geometry & Jacobian ---
    # Mapped from [-1, 1] to [0, lx], etc.
    # invJ contains derivatives of natural coords w.r.t physical coords: dxi/dx
    invJ = np.diag([2.0 / lx, 2.0 / ly, 2.0 / lz])
    detJ = (lx * ly * lz) / 8.0

    # --- 5. Allocate Output Matrices ---
    ke = np.zeros((24, 24))
    k_eth = np.zeros((8, 8))
    c_ethm = np.zeros((24, 8))

    # --- 6. Integration Loop ---
    for xi in gp:
        for eta in gp:
            for zeta in gp:
                # --- Shape Functions (N) ---
                # Shape: (8,)
                N = 0.125 * (1 + xi_nodes * xi) * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)

                # --- Shape Function Derivatives (dN_dxi) ---
                # Shape: (8, 3) -> columns are dN/dxi, dN/deta, dN/dzeta
                dN_dxi = np.zeros((8, 3))
                dN_dxi[:, 0] = 0.125 * xi_nodes * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)
                dN_dxi[:, 1] = 0.125 * et_nodes * (1 + xi_nodes * xi) * (1 + ze_nodes * zeta)
                dN_dxi[:, 2] = 0.125 * ze_nodes * (1 + xi_nodes * xi) * (1 + et_nodes * eta)

                # --- Convert to Physical Derivatives (dN_dx) ---
                # dN/dx = dN/dxi * dxi/dx
                # Shape: (8, 3) -> columns are dN/dx, dN/dy, dN/dz
                dN_dx = dN_dxi @ invJ

                # --- Build B Matrix (Strain-Displacement) ---
                # Shape: (6, 24)
                B = np.zeros((6, 24))
                # Normal Strains
                B[0, 0::3] = dN_dx[:, 0]  # xx
                B[1, 1::3] = dN_dx[:, 1]  # yy
                B[2, 2::3] = dN_dx[:, 2]  # zz
                # Shear Strains
                B[3, 0::3] = dN_dx[:, 1];
                B[3, 1::3] = dN_dx[:, 0]  # xy
                B[4, 1::3] = dN_dx[:, 2];
                B[4, 2::3] = dN_dx[:, 1]  # yz
                B[5, 0::3] = dN_dx[:, 2];
                B[5, 2::3] = dN_dx[:, 0]  # zx

                # --- Accumulate Matrices ---

                # 1. Mechanical Stiffness: B.T * D * B
                ke += (B.T @ D @ B) * detJ

                # 2. Thermal Conductivity: (dN_dx).T * k * (dN_dx)
                # Since k is scalar isotropic, we can pull it out.
                # dN_dx is (8x3), we want (8x8) result.
                k_eth += (dN_dx @ dN_dx.T) * k * detJ

                # 3. Thermo-Mechanical Coupling
                # We want force f = Integral( B.T * sigma_thermal ) dV
                # sigma_thermal = beta_vec * T_scalar
                # T_scalar = N_vector . T_nodal
                # So: C_ethm = B.T @ beta_vec(6x1) @ N(1x8)

                # Reshape N to (1, 8) for matrix broadcast
                N_row = N.reshape(1, 8)

                # Compute local thermal stress vector contribution B^T * beta
                # (24x6) @ (6,) -> (24,)
                f_thermal_unit = B.T @ beta_vec

                # Outer product to map to nodes: (24,1) * (1,8) -> (24,8)
                c_ethm += np.outer(f_thermal_unit, N_row) * detJ

    return ke, k_eth, c_ethm


# --- Example Usage / Verification ---
if __name__ == "__main__":
    # Standard steel-like parameters
    E_val = 210e9
    nu_val = 0.3
    k_val = 50.0
    alpha_val = 12e-6

    ke, k_eth, c_ethm = get_stiffness_matrices(nu_val, E_val, k_val, alpha_val)

    print(f"Ke shape: {ke.shape}")
    print(f"K_eth shape: {k_eth.shape}")
    print(f"C_ethm shape: {c_ethm.shape}")
    print(f"Trace(Ke): {np.trace(ke):.4e}")