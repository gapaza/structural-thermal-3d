import numpy as np

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

if __name__ == "__main__":
    # Example usage
    k = 10.0  # Thermal conductivity

    k_eth = get_thermal_stiffness(k)
    print("Thermal Conductivity Matrix (k_eth):")
    print(k_eth)