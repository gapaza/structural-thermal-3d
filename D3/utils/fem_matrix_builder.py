import numpy as np

def fe_melthm_3d(nu: float, E: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 3D Hex8 element matrices for thermo-elasticity:
      - ke    : 24x24 mechanical stiffness (Hex8, 3 dof/node)
      - k_eth :  8x8 thermal conductivity (Hex8, 1 dof/node)
      - c_ethm: 24x8 coupling mapping nodal temperatures to equivalent mechanical forces

    Assumptions:
      - Isotropic linear elasticity and isotropic thermal conductivity.
      - Reference Hex8 element over [-1, 1]^3 with a unit-cube physical mapping.
      - 2x2x2 Gauss integration.
      - Thermal strain = alpha * ΔT * [1, 1, 1, 0, 0, 0]^T (Voigt).

    Args:
        nu (float): Poisson's ratio
        E (float): Young's modulus
        k (float): Thermal conductivity (isotropic)
        alpha (float): Coefficient of thermal expansion
        lx, ly, lz: Physical dimensions of the element in x, y, z.

    Returns:
        (ke, k_eth, c_ethm)
    """
    # --- Gauss points (2x2x2) and weights ---
    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w  = np.array([1.0, 1.0])

    # --- Hex8 shape functions and derivatives in natural coordinates ---
    # Node order: (-,-,-), (+,-,-), (+,+,-), (-,+,-), (-,-,+), (+,-,+), (+,+,+), (-,+,+)
    xi_nodes = np.array([-1,  1,  1, -1, -1,  1,  1, -1], dtype=float)
    et_nodes = np.array([-1, -1,  1,  1, -1, -1,  1,  1], dtype=float)
    ze_nodes = np.array([-1, -1, -1, -1,  1,  1,  1,  1], dtype=float)

    def shape_fun_and_derivs(xi, eta, zeta):
        """
        Returns:
          N: (8,) shape functions
          dN_dxi: (8,3) derivatives w.r.t. [xi, eta, zeta]
        """
        N = 0.125 * (1 + xi_nodes*xi) * (1 + et_nodes*eta) * (1 + ze_nodes*zeta)

        # Derivatives with respect to natural coords
        dN_dxi  = np.zeros((8, 3))
        dN_dxi[:, 0] = 0.125 * xi_nodes  * (1 + et_nodes*eta) * (1 + ze_nodes*zeta)    # ∂N/∂xi
        dN_dxi[:, 1] = 0.125 * et_nodes  * (1 + xi_nodes*xi) * (1 + ze_nodes*zeta)    # ∂N/∂eta
        dN_dxi[:, 2] = 0.125 * ze_nodes  * (1 + xi_nodes*xi) * (1 + et_nodes*eta)     # ∂N/∂zeta
        return N, dN_dxi

    # --- Geometry & Jacobian ---
    # For a unit cube in physical space mapped from [-1,1]^3:
    # x = (xi+1)/2, y = (eta+1)/2, z = (zeta+1)/2 -> J = diag(0.5, 0.5, 0.5)
    J = np.diag([0.5, 0.5, 0.5])
    detJ = np.linalg.det(J)            # = 0.125
    invJ = np.linalg.inv(J)            # = diag(2,2,2)

    # --- Elasticity matrix (Voigt 6x6) ---
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    D = np.zeros((6, 6))
    # Fill upper-left 3x3 with lambda
    D[:3, :3] = lam
    # Add 2*mu to the diagonal of the upper-left 3x3
    D[:3, :3] += 2 * mu * np.eye(3)
    # Fill lower-right 3x3 diagonal with shear modulus mu
    D[3:, 3:] = np.diag([mu, mu, mu])

    # Thermal "volumetric" strain direction in Voigt
    e_th = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    # --- Allocate element matrices ---
    ke    = np.zeros((24, 24))
    k_eth = np.zeros(( 8,  8))
    c_ethm= np.zeros((24,  8))

    # --- Integration loop ---
    for i, xi in enumerate(gp):
        for j, eta in enumerate(gp):
            for kq, zeta in enumerate(gp):
                N, dN_dxi = shape_fun_and_derivs(xi, eta, zeta)

                # Gradients in physical coords: dN_dx = dN_dxi * invJ
                dN_dx = dN_dxi @ invJ      # (8,3)

                # Build B-matrix (6 x 24) for Hex8, 3 dof/node (u,v,w)
                B = np.zeros((6, 24))
                for a in range(8):
                    ix = 3*a
                    dy, dx_, dz = dN_dx[a,1], dN_dx[a,0], dN_dx[a,2]  # clarity
                    # normal strains
                    B[0, ix+0] = dx_                  # εxx from u_x
                    B[1, ix+1] = dy                   # εyy from v_y
                    B[2, ix+2] = dz                   # εzz from w_z
                    # shear strains (engineering)
                    B[3, ix+0] = dy                   # γxy from u_y
                    B[3, ix+1] = dx_                  # γxy from v_x
                    B[4, ix+1] = dz                   # γyz from v_z
                    B[4, ix+2] = dy                   # γyz from w_y
                    B[5, ix+0] = dz                   # γzx from u_z
                    B[5, ix+2] = dx_                  # γzx from w_x

                # Thermal gradient matrix for conduction: G = grad(N) (3 x 8)
                G = dN_dx.T  # rows: [dN/dx; dN/dy; dN/dz]

                # Weight factor
                wt = w[i]*w[j]*w[kq]*detJ

                # --- Accumulate ---
                # Mechanical stiffness
                ke += (B.T @ D @ B) * wt

                # Thermal conductivity (isotropic k * grad N^T grad N)
                k_eth += (G.T @ G) * (k * wt)

                # Thermo-mech coupling: columns correspond to nodal temperatures (via N_j)
                # fe_th = ∫ B^T D (alpha * e_th * ΔT) dV, and ΔT at GP = Σ N_j θ_j
                # => C_ethm[:, j] contribution = ∫ B^T D (alpha * e_th) * N_j dV
                De_th = D @ (alpha * e_th)         # (6,)
                c_ethm += (B.T @ De_th[:,None] @ N[None,:]) * wt

    return ke, k_eth, c_ethm


def fe_melthm_3d_abacus(nu: float, E: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build 3D Hex8 element matrices for thermo-elasticity:
      - ke    : 24x24 mechanical stiffness (Hex8, 3 dof/node)
      - k_eth :  8x8 thermal conductivity (Hex8, 1 dof/node)
      - c_ethm: 24x8 coupling mapping nodal temperatures to equivalent mechanical forces

    Assumptions:
      - Isotropic linear elasticity and isotropic thermal conductivity.
      - Reference Hex8 element over [-1, 1]^3 with a unit-cube physical mapping.
      - 2x2x2 Gauss integration.
      - Thermal strain = alpha * ΔT * [1, 1, 1, 0, 0, 0]^T (Voigt).

    Args:
        nu (float): Poisson's ratio
        E (float): Young's modulus
        k (float): Thermal conductivity (isotropic)
        alpha (float): Coefficient of thermal expansion

    Returns:
        (ke, k_eth, c_ethm)
    """
    return None, None, None







if __name__ == '__main__':

    nu = 0.3      # Poisson's ratio
    e = 1.0       # Modulus of elasticity
    k = 1.0       # Thermal conductivity
    alpha = 5e-4  # Coefficient of thermal expansion

    ke, k_eth, c_ethm = fe_melthm_3d(nu, e, k, alpha)

    # print("Mechanical stiffness ke:\n", ke)                  # shape: (24, 24)
    # print("\nThermal conductivity k_eth:\n", k_eth)          # shape: (8, 8)
    # print("\nThermo-mechanical coupling c_ethm:\n", c_ethm)  # shape: (24, 8)

    # from D3.utils.fem_matrix_builder2 import fe_melthm_3d_flexible
    # ke2 = fe_melthm_3d_flexible(nu, e, lx=1, ly=1, lz=1)
    # ke_diff = ke2 - ke
    # print('Diff', ke_diff)
    # print(np.allclose(ke2, ke))

    from D3.utils.fem_matrix_builder3 import fe_melthm_3d as fe_melthm_3d_v2
    ke2, k_eth2, c_ethm2 = fe_melthm_3d_v2(nu, e, k, alpha, lx=1, ly=1, lz=1)

    print(np.allclose(ke, ke2))
    print(np.allclose(k_eth, k_eth2))
    print(np.allclose(c_ethm, c_ethm2))






























