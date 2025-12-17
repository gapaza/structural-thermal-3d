import numpy as np

def get_thermal_stiffness(k: float, lx: float = 1.0, ly: float = 1.0, lz: float = 1.0):
    """
    Builds 8x8 thermal conductivity matrix (Hex8).
    Integral( dN_dx.T * k * dN_dx ) dV
    """
    val = 1.0 / np.sqrt(3)
    gp = np.array([-val, val])
    xi_nodes = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    et_nodes = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    ze_nodes = np.array([-1, -1, -1, -1, 1, 1, 1, 1])

    invJ = np.diag([2.0 / lx, 2.0 / ly, 2.0 / lz])
    detJ = (lx * ly * lz) / 8.0

    k_eth = np.zeros((8, 8))

    for xi in gp:
        for eta in gp:
            for zeta in gp:
                dN_dxi = np.zeros((8, 3))
                dN_dxi[:, 0] = 0.125 * xi_nodes * (1 + et_nodes * eta) * (1 + ze_nodes * zeta)
                dN_dxi[:, 1] = 0.125 * et_nodes * (1 + xi_nodes * xi) * (1 + ze_nodes * zeta)
                dN_dxi[:, 2] = 0.125 * ze_nodes * (1 + xi_nodes * xi) * (1 + et_nodes * eta)

                dN_dx = dN_dxi @ invJ

                # Accumulate k_eth
                # dN_dx is (8x3), resulting term is (8x8)
                k_eth += (dN_dx @ dN_dx.T) * k * detJ

    return k_eth

if __name__ == "__main__":
    # Example usage
    k = 10.0  # Thermal conductivity

    k_eth = get_thermal_stiffness(k)
    print("Thermal Conductivity Matrix (Hex8):")
    print(k_eth)