import numpy as np

def compute_elastic_metrics(obj, x, U1):
    """
    Computes Strain Energy Density and Von Mises Stress for all elements.
    Note: Ignores thermal expansion strains (calculate stress from total deformation).

    Args:
        x: Density array (flat or shaped)
        U1: Global displacement vector

    Returns:
        strain_energy: (num_elems,) array of element strain energies
        von_mises: (num_elems,) array of element von Mises stresses
    """
    # 1. Setup Material Interpolation (E_element)
    # Using the same penalization as the stiffness matrix assembly
    # x = x.flatten(order='F')
    x_penal = x ** obj.penal
    E_elem = (obj.Emin + x_penal * (obj.E0 - obj.Emin))

    # 2. Extract Element Displacements
    # Shape: (num_elems, 24)
    U_e = U1[obj.edofMat_mech]

    # ---------------------------------------------------------
    # A. Strain Energy Calculation (0.5 * u^T * K * u)
    # ---------------------------------------------------------
    # We perform the matrix multiplication: vector_i * Matrix * vector_i^T
    # Einstein Summation: i=element, j=row, k=col
    # Ke0 is constant, so we compute U_e @ Ke0 first

    # Shape: (num_elems, 24)
    KU_e = U_e @ obj.Ke0

    # Dot product: sum(U_e * KU_e) along axis 1
    # Scale by 0.5 and the element-wise Young's Modulus factor
    # Note: Ke0 was computed with E=1.0, so we just multiply by E_elem
    strain_energy = 0.5 * np.sum(U_e * KU_e, axis=1) * E_elem

    # ---------------------------------------------------------
    # B. Von Mises Stress Calculation
    # ---------------------------------------------------------
    # We need the Strain-Displacement matrix B and Constitutive matrix D
    # Calculated at the element centroid (xi=0, eta=0, zeta=0)

    # 1. Build B0 (6x24) at centroid
    # Diffs for 8-node hex at origin are +/- 1/4 scaled by length
    dx = 1.0 / (4.0 * obj.lx)
    dy = 1.0 / (4.0 * obj.ly)
    dz = 1.0 / (4.0 * obj.lz)

    # Node signs relative to centroid (Standard Hex8 Ordering)
    # 1:(-,-,-), 2:(+,-,-), 3:(+,+,-), 4:(-,+,-)
    # 5:(-,-,+), 6:(+,-,+), 7:(+,+,+), 8:(-,+,+)
    # (Check your specific node ordering in _build_edof_matrices.
    # Based on your grid logic, it aligns with standard anti-clockwise layers)

    # Gradients of shape functions dN/dx, dN/dy, dN/dz
    # Shape (3, 8)
    # Signs for [x, y, z] directions for nodes 1..8
    signs = np.array([
        [-1, -1, -1], [ 1, -1, -1], [ 1,  1, -1], [-1,  1, -1],
        [-1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [-1,  1,  1]
    ]).T

    dNx = signs[0, :] * dx
    dNy = signs[1, :] * dy
    dNz = signs[2, :] * dz

    # Construct B0 (Voigt: xx, yy, zz, xy, yz, zx)
    B0 = np.zeros((6, 24))
    for i in range(8):
        col = 3 * i
        # epsilon_xx = du/dx
        B0[0, col]   = dNx[i]
        # epsilon_yy = dv/dy
        B0[1, col+1] = dNy[i]
        # epsilon_zz = dw/dz
        B0[2, col+2] = dNz[i]
        # gamma_xy = du/dy + dv/dx
        B0[3, col]   = dNy[i]; B0[3, col+1] = dNx[i]
        # gamma_yz = dv/dz + dw/dy
        B0[4, col+1] = dNz[i]; B0[4, col+2] = dNy[i]
        # gamma_zx = du/dz + dw/dx
        B0[5, col]   = dNz[i]; B0[5, col+2] = dNx[i]

    # 2. Build Constitutive Matrix D0 (6x6) for E=1.0
    # Isotropic Linear Elasticity
    nu = obj.nu
    c1 = ((1 - nu) / ((1 + nu) * (1 - 2 * nu)))
    c2 = (nu / ((1 + nu) * (1 - 2 * nu)))
    c3 = (1 / (2 * (1 + nu))) # Shear modulus G

    D0 = np.zeros((6, 6))
    D0[0:3, 0:3] = c2
    np.fill_diagonal(D0, [c1, c1, c1, c3, c3, c3])

    # 3. Compute Strains and Stresses (Vectorized)
    # Strain = U_e * B^T -> Shape (num_elems, 6)
    epsilon = U_e @ B0.T

    # Stress = (D0 * epsilon) * E_scale
    # We multiply D0 @ epsilon.T, then transpose back, then scale
    # Result sigma is (num_elems, 6) [sig_x, sig_y, sig_z, tau_xy, tau_yz, tau_zx]
    sigma0 = epsilon @ D0  # D0 is symmetric, so order doesn't strictly matter for D0 @ vec
    sigma = sigma0 * E_elem[:, None]

    # 4. Compute Von Mises
    # VM = sqrt(0.5 * [(s_x-s_y)^2 + (s_y-s_z)^2 + (s_z-s_x)^2 + 6*(t_xy^2 + ...)])
    # Or standard form: sqrt(s_x^2 + s_y^2 + s_z^2 - s_x*s_y - ... + 3*shear^2)

    sx, sy, sz = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    txy, tyz, tzx = sigma[:, 3], sigma[:, 4], sigma[:, 5]

    vm_sq = (sx**2 + sy**2 + sz**2) - (sx*sy + sy*sz + sz*sx) + \
            3.0 * (txy**2 + tyz**2 + tzx**2)

    von_mises = np.sqrt(np.maximum(0, vm_sq))

    return strain_energy, von_mises













