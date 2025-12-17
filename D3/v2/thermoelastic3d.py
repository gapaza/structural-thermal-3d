import numpy as np
from math import ceil
import time
import os
import sys
from dataclasses import dataclass
import argparse
from scipy.sparse import coo_matrix, linalg
from scipy.sparse.linalg import spsolve

from D3.utils.linear_solver import solve_spd_with_amg


# ==========================================
# 1. MMA Interface
# ==========================================

# from D3.utils.mma_subroutine import mmasub, MMAInputs
from D2.flexure_v2.mma_subroutine import mmasub, MMAInputs



# ==========================================
# 2. Finite Element Helper / Materials
# ==========================================

from element.thermoelastic_weak_3d import get_stiffness_matrices



# ==========================================
# 3. Optimization Class
# ==========================================

NUMPY_SAVE_DIR = '/Users/gapaza/repos/ideal/structural-thermal-3d/D3/v2/designs'
# NUMPY_SAVE_DIR = '/home/gapaza/scratch/repos/structural-thermal-3d/D3/v2/designs'


class ThermoelasticTopologyOptimization3D:
    def __init__(self, nelx, nely, nelz, volfrac, penal, rmin, lx=1.0, ly=1.0, lz=1.0, iter_solve=False, fname='design.npz', el_weight=0.5, plot=False):
        self.nelx = nelx
        self.nely = nely
        self.nelz = nelz
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.fname = fname
        self.el_weight = el_weight
        self.th_weight = 1.0 - el_weight
        self.plot = plot

        if iter_solve is False:
            self.sparse_solver = spsolve
        else:
            self.sparse_solver = solve_spd_with_amg

        # Material Properties (Arbitrary units for example, polypropylene units in comments)
        self.nu = 0.3       # Poisson ratio (0.45)
        self.E0 = 2e5       # Young's Modulus (2e5 MPa -> 2e11 Pa)
        self.Emin = 1e-9    # Void stiffness

        # self.rho0 = 7850.0  # Density (930 kg/m^3) WANT kg/mm^3
        self.rho0 = 7.85e-6  # Density (7.85e-6 kg/mm^3 = 7850 kg/m^3)

        self.k0 = 3.0       # Thermal Conductivity (0.22 J/(smK))
        self.kmin = 1e-9
        self.alpha = 1.2e-5   # Expansion coeff (8.0e-5 /K)
        self.Cp = 4.0       # Heat Capacity (1.9e3 J/(kgK))
        self.Cmin = 1e-9    # Add this

        # Geometry
        self.ndof_mech = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
        self.ndof_therm = (nelx + 1) * (nely + 1) * (nelz + 1)
        self.num_elems = nelx * nely * nelz

        # Build Element Matrices (Unit Material)
        self.lx = lx  # mm
        self.ly = ly  # mm
        self.lz = lz  # mm
        # self.Ke0, self.Kt0, self.Cethm0 = get_stiffness_matrices(self.nu, self.E0, self.k0, self.alpha, lx=self.lx, ly=self.ly, lz=self.lz)
        self.Ke0, self.Kt0, self.Cethm0 = get_stiffness_matrices(self.nu, 1.0, 1.0, 1.0, lx=self.lx, ly=self.ly, lz=self.lz)

        # DOF Mappings
        self._build_edof_matrices()
        # self._build_edof_matrices_manual()

        # Filters
        self._build_filters()

        # MMA State Initialization
        self.iter = 0
        self.x = self.volfrac * np.ones(self.num_elems)
        self.xold1 = self.x.copy()
        self.xold2 = self.x.copy()
        self.low = np.zeros_like(self.x)
        self.upp = np.ones_like(self.x)
        self.a0 = 1.0
        self.a_mma = np.zeros(1)  # m=1 constraint
        self.c_mma = 1000.0 * np.ones(1)
        self.d_mma = np.zeros(1)

    def _build_filters_old(self):
        """3D cone (hat) filter over elements. Returns sparse H and row-sum hs."""

        # Element indexing: we use (ix, iy, iz) with shape (nelx, nely, nelz)
        # Flattening order matches the node numbering used elsewhere (z fastest).
        def e_index(ix, iy, iz) -> int:
            return (nely * nelz) * ix + (nelz) * iy + iz

        i_h: list[int] = []
        j_h: list[int] = []
        s_h: list[float] = []

        rceil = ceil(rmin) - 1  # integer neighborhood radius

        for ix in range(nelx):
            ix_min = max(ix - rceil, 0)
            ix_max = min(ix + rceil, nelx - 1)
            for iy in range(nely):
                iy_min = max(iy - rceil, 0)
                iy_max = min(iy + rceil, nely - 1)
                for iz in range(nelz):
                    iz_min = max(iz - rceil, 0)
                    iz_max = min(iz + rceil, nelz - 1)

                    e1 = e_index(ix, iy, iz)
                    for jx in range(ix_min, ix_max + 1):
                        for jy in range(iy_min, iy_max + 1):
                            for jz in range(iz_min, iz_max + 1):
                                e2 = e_index(jx, jy, jz)
                                # Euclidean distance in element index space
                                dx = ix - jx
                                dy = iy - jy
                                dz = iz - jz
                                dist = (dx * dx + dy * dy + dz * dz) ** 0.5
                                w = max(0.0, rmin - dist)
                                if w > 0.0:
                                    i_h.append(e1)
                                    j_h.append(e2)
                                    s_h.append(w)

        n = nelx * nely * nelz
        H = coo_matrix((s_h, (i_h, j_h)), shape=(n, n)).tocsr()
        hs = np.array(H.sum(axis=1)).ravel()
        self.H = H
        self.hs = hs

    def _build_filters(self):
        """3D cone (hat) filter over elements. Returns sparse H and row-sum hs."""

        # Local references for cleaner code
        nelx, nely, nelz = self.nelx, self.nely, self.nelz
        rmin = self.rmin

        # ---------------------------------------------------------
        # Indexing Logic:
        # y is fastest (stride 1)
        # x is middle  (stride nely)
        # z is slowest (stride nelx * nely)
        # ---------------------------------------------------------
        def e_index(ix, iy, iz) -> int:
            return (nelx * nely) * iz + (nely) * ix + iy

        i_h: list[int] = []
        j_h: list[int] = []
        s_h: list[float] = []

        rceil = ceil(rmin) - 1

        # Loop Order: Slowest -> Fastest (z -> x -> y)
        for iz in range(nelz):
            iz_min = max(iz - rceil, 0)
            iz_max = min(iz + rceil, nelz - 1)

            for ix in range(nelx):
                ix_min = max(ix - rceil, 0)
                ix_max = min(ix + rceil, nelx - 1)

                for iy in range(nely):
                    iy_min = max(iy - rceil, 0)
                    iy_max = min(iy + rceil, nely - 1)

                    e1 = e_index(ix, iy, iz)

                    # Neighbor loops (Order generally doesn't matter for logic,
                    # but matching the outer loops helps cache locality)
                    for jz in range(iz_min, iz_max + 1):
                        for jx in range(ix_min, ix_max + 1):
                            for jy in range(iy_min, iy_max + 1):

                                e2 = e_index(jx, jy, jz)

                                dx = ix - jx
                                dy = iy - jy
                                dz = iz - jz
                                dist = (dx * dx + dy * dy + dz * dz) ** 0.5

                                w = max(0.0, rmin - dist)
                                if w > 0.0:
                                    i_h.append(e1)
                                    j_h.append(e2)
                                    s_h.append(w)

        n = nelx * nely * nelz
        H = coo_matrix((s_h, (i_h, j_h)), shape=(n, n)).tocsr()
        hs = np.array(H.sum(axis=1)).ravel()

        self.H = H
        self.hs = hs

    def _build_edof_matrices_manual(self):
        self.edofMat_mech2 = np.zeros((self.num_elems, 24), dtype=int)
        self.edofMat_therm2 = np.zeros((self.num_elems, 8), dtype=int)

        for elz in range(self.nelz):
            for elx in range(self.nelx):
                for ely in range(self.nely):
                    el = ely + (elx * self.nely) + (elz * (self.nelx * self.nely))
                    print('elz', elz, '- elx', elx, '- ely', ely, '- elz', elz, ':', el)

                    n1 = ((ely + 1) + elx * (self.nely + 1))       + (elz * ((self.nelx + 1) * (self.nely + 1)))
                    n2 = ((ely + 1) + (elx + 1) * (self.nely + 1)) + (elz * ((self.nelx + 1) * (self.nely + 1)))
                    n3 = (ely + (elx + 1) * (self.nely + 1))       + (elz * ((self.nelx + 1) * (self.nely + 1)))
                    n4 = (ely + elx * (self.nely + 1))             + (elz * ((self.nelx + 1) * (self.nely + 1)))

                    n5 = ((ely + 1) + elx * (self.nely + 1))       + ((elz + 1) * ((self.nelx + 1) * (self.nely + 1)))
                    n6 = ((ely + 1) + (elx + 1) * (self.nely + 1)) + ((elz + 1) * ((self.nelx + 1) * (self.nely + 1)))
                    n7 = (ely + (elx + 1) * (self.nely + 1))       + ((elz + 1) * ((self.nelx + 1) * (self.nely + 1)))
                    n8 = (ely + elx * (self.nely + 1))             + ((elz + 1) * ((self.nelx + 1) * (self.nely + 1)))

                    print('\t', n1, n2, n3, n4, n5, n6, n7, n8)

                    self.edofMat_therm2[el, :] = [n1, n2, n3, n4, n5, n6, n7, n8]

                    self.edofMat_mech2[el, :] = [
                        3 * n1, 3 * n1 + 1, 3 * n1 + 2,
                        3 * n2, 3 * n2 + 1, 3 * n2 + 2,
                        3 * n3, 3 * n3 + 1, 3 * n3 + 2,
                        3 * n4, 3 * n4 + 1, 3 * n4 + 2,
                        3 * n5, 3 * n5 + 1, 3 * n5 + 2,
                        3 * n6, 3 * n6 + 1, 3 * n6 + 2,
                        3 * n7, 3 * n7 + 1, 3 * n7 + 2,
                        3 * n8, 3 * n8 + 1, 3 * n8 + 2,
                    ]

        self.iK_mech2 = np.kron(self.edofMat_mech2, np.ones((1, 24))).flatten()
        self.jK_mech2 = np.kron(self.edofMat_mech2, np.ones((24, 1))).flatten()
        self.iK_therm2 = np.kron(self.edofMat_therm2, np.ones((1, 8))).flatten()
        self.jK_therm2 = np.kron(self.edofMat_therm2, np.ones((8, 1))).flatten()

    def _build_edof_matrices(self):
        """
        Precompute DOF indices for efficiency (Vectorized for 3D).
        Generates:
           - self.edofMat_mech: (n_elem, 24) matrix of mechanical DOFs
           - self.edofMat_therm: (n_elem, 8) matrix of thermal DOFs
           - Indexing vectors (iK, jK) for assembly (rows repeat, columns tile)
        """
        # 1. Coordinate Grids
        # "Fastest" dimension is Y (rows), then X (cols), then Z (depth)
        # We use meshgrid with 'F' (Fortran) to mimic the column-major logic of Top3d
        nelx, nely, nelz = self.nelx, self.nely, self.nelz

        # Generates grid of element indices
        elx, ely, elz = np.meshgrid(range(nelx), range(nely), range(nelz), indexing='xy')
        elx = elx.flatten(order='F')
        ely = ely.flatten(order='F')
        elz = elz.flatten(order='F')

        self.num_elems = len(elx)

        # 2. Node Mapping
        # Node strides
        ny_n = nely + 1
        nx_n = nelx + 1

        # Calculate the "n1" (Bottom-Left-Back) node for every element at once
        # n = y + x*ny_n + z*ny_n*nx_n
        n1 = (ely + 1) + elx * ny_n + elz * ny_n * nx_n
        n2 = (ely + 1) + (elx + 1) * ny_n + elz * ny_n * nx_n
        n3 = ely + (elx + 1) * ny_n + elz * ny_n * nx_n
        n4 = ely + elx * ny_n + elz * ny_n * nx_n

        # Shift to front face (z+1) is just adding the slice stride
        slice_stride = ny_n * nx_n
        n5 = n1 + slice_stride
        n6 = n2 + slice_stride
        n7 = n3 + slice_stride
        n8 = n4 + slice_stride

        # 3. Thermal DOF Matrix (8 Nodes per element, 1 DOF per node)
        # Shape: (num_elems, 8)
        self.edofMat_therm = np.stack([n1, n2, n3, n4, n5, n6, n7, n8], axis=1).astype(int)

        # 4. Mechanical DOF Matrix (8 Nodes, 3 DOFs per node)
        # Shape: (num_elems, 24)
        # DOFs are: 3*n, 3*n+1, 3*n+2
        self.edofMat_mech = np.zeros((self.num_elems, 24), dtype=int)

        # Vectorized expansion of 3 DOFs per node
        # We repeat the node IDs 3 times and add [0, 1, 2] offsets
        base_nodes = self.edofMat_therm  # (N, 8)

        # Map: [n1_u, n1_v, n1_w, n2_u, ...]
        # This loop is small (runs 8 times), so it's fine.
        for i in range(8):
            node_col = base_nodes[:, i]
            self.edofMat_mech[:, 3 * i + 0] = 3 * node_col  # u
            self.edofMat_mech[:, 3 * i + 1] = 3 * node_col + 1  # v
            self.edofMat_mech[:, 3 * i + 2] = 3 * node_col + 2  # w

        # 5. Assembly Index Vectors
        # K_mech (24x24)
        self.iK_mech = np.repeat(self.edofMat_mech, 24, axis=1).flatten()
        self.jK_mech = np.tile(self.edofMat_mech, (1, 24)).flatten()

        # K_therm (8x8)
        self.iK_therm = np.repeat(self.edofMat_therm, 8, axis=1).flatten()
        self.jK_therm = np.tile(self.edofMat_therm, (1, 8)).flatten()

        # Coupling Matrix C_ethm (24x8) - RECTANGULAR
        # Rows (i) are Mechanical DOFs (24), Cols (j) are Thermal DOFs (8)
        # Logic: For every element, we have a 24x8 block.
        # i must repeat each mech DOF 8 times: [m1...m1 (8 times), m2...m2 (8 times)]
        # j must tile the thermal DOFs 24 times: [t1..t8, t1..t8...]
        self.iK_coup = np.repeat(self.edofMat_mech, 8, axis=1).flatten()
        self.jK_coup = np.tile(self.edofMat_therm, (1, 24)).flatten()

    def _assemble_system(self, x):
        """
        Assembles Global matrices for 3D Thermo-elasticity.

        Args:
            x: Density array (nelx, nely, nelz) or flattened

        Returns:
            K_mech: (ndof_mech, ndof_mech) CSC
            K_therm: (ndof_therm, ndof_therm) CSC
            C_coup:  (ndof_mech, ndof_therm) CSC - Coupling Matrix
        """
        # Ensure x is flat for calculations
        x = x.flatten(order='F')

        # Material Interpolation
        x_penal = x ** self.penal

        # 1. Mechanical Stiffness K (24x24 local)
        sK = (self.Emin + x_penal * (self.E0 - self.Emin))
        vals_K = np.kron(sK, self.Ke0.flatten())  # Ke0 is 24x24
        K_mech = coo_matrix((vals_K, (self.iK_mech, self.jK_mech)),
                            shape=(self.ndof_mech, self.ndof_mech)).tocsc()

        # 2. Thermal Conductivity Kt (8x8 local)
        # Assuming simple conductivity penalization (often p=3 or p=1 depending on physics)
        # Using same x_penal here for consistency with standard SIMP
        sKt = (self.kmin + x_penal * (self.k0 - self.kmin))
        vals_Kt = np.kron(sKt, self.Kt0.flatten())  # Kt0 is 8x8
        K_therm = coo_matrix((vals_Kt, (self.iK_therm, self.jK_therm)),
                             shape=(self.ndof_therm, self.ndof_therm)).tocsc()

        # 3. Coupling Matrix C_ethm (24x8 local)
        # This maps Temperatures (cols) to Mechanical Forces (rows).
        # Interpolation: Usually follows Young's modulus (stiffness) scaling
        # because thermal stress = E * alpha * dT.
        sCoup = (self.Emin + x_penal * (self.E0 - self.Emin))
        vals_Coup = np.kron(sCoup * self.alpha, self.Cethm0.flatten())
        C_coup = coo_matrix((vals_Coup, (self.iK_coup, self.jK_coup)),
                            shape=(self.ndof_mech, self.ndof_therm)).tocsc()

        return K_mech, K_therm, C_coup

    def _build_boundary_conditions(self):

        # Some preliminary boundary node definitions
        node_TF_TL = 0
        node_TF_ML = self.nely // 2
        node_TF_BL = self.nely
        node_TF_CENTER = ((self.nely + 1) * (self.nelx//2)) + (self.nely//2)
        node_TF_TR =     ((self.nely + 1) * self.nelx)
        node_TF_BR =     ((self.nely + 1) * self.nelx) + self.nely
        nodes_TF = np.arange(node_TF_TL, node_TF_BR + 1)


        node_MF_TL =     ((self.nely + 1) * (self.nelx + 1) * (self.nelz//2)) + node_TF_TL
        node_MF_ML =     ((self.nely + 1) * (self.nelx + 1) * (self.nelz//2)) + node_TF_ML
        node_MF_BL =     ((self.nely + 1) * (self.nelx + 1) * (self.nelz//2)) + node_TF_BL
        node_MF_CENTER = ((self.nely + 1) * (self.nelx + 1) * (self.nelz//2)) + node_TF_CENTER
        node_MF_TR =     ((self.nely + 1) * (self.nelx + 1) * (self.nelz//2)) + node_TF_TR
        node_MF_BR =     ((self.nely + 1) * (self.nelx + 1) * (self.nelz//2)) + node_TF_BR
        nodes_MF = np.arange(node_MF_TL, node_MF_BR + 1)


        node_BF_TL =     ((self.nely + 1) * (self.nelx + 1) * self.nelz) + node_TF_TL
        node_BF_ML =     ((self.nely + 1) * (self.nelx + 1) * self.nelz) + node_TF_ML
        node_BF_BL =     ((self.nely + 1) * (self.nelx + 1) * self.nelz) + node_TF_BL
        node_BF_CENTER = ((self.nely + 1) * (self.nelx + 1) * self.nelz) + node_TF_CENTER
        node_BF_TR =     ((self.nely + 1) * (self.nelx + 1) * self.nelz) + node_TF_TR
        node_BF_BR =     ((self.nely + 1) * (self.nelx + 1) * self.nelz) + node_TF_BR
        nodes_BF = np.arange(node_BF_TL, node_BF_BR + 1)

        # --- A. Steady State Heat Conduction ---

        # 1. Heatsinks (Dirichlet)
        # fixed_therm_nodes = np.array([node_TF_TL])
        fixed_therm_nodes = nodes_TF
        fixed_therm_temp = 0

        # 2. Heat Generation (Neumann)
        # heat_gen_nodes = nodes_BF
        heat_gen_nodes = np.array([node_MF_CENTER])
        heat_gen_tref = 10.0

        # --- B. Mechanical Equilibrium ---

        # # 1. Loaded DOFs
        # mechanical_load = -0.001
        # # loaded_el_nodes = np.array([node_TF_BR])
        # loaded_el_nodes = nodes_TF
        # loaded_el_nodes_dof_x = (3 * loaded_el_nodes) + 0
        # loaded_el_nodes_dof_y = (3 * loaded_el_nodes) + 1
        # loaded_el_nodes_dof_z = (3 * loaded_el_nodes) + 2
        # loaded_el_dof = np.hstack([
        #     # loaded_el_nodes_dof_x,
        #     # loaded_el_nodes_dof_y,
        #     loaded_el_nodes_dof_z,
        # ])
        #
        # # 2. Fixed DOFs
        # fixed_el_nodes = np.array([node_TF_TL, node_TF_BL, node_BF_TL, node_BF_BL])
        # # fixed_el_nodes = np.array([node_BF_TL, node_BF_BL, node_BF_TR, node_BF_BR])
        # # fixed_el_nodes = np.array([node_BF_TL, node_BF_BL])
        # fixed_el_nodes_dof_x = (3 * fixed_el_nodes) + 0
        # fixed_el_nodes_dof_y = (3 * fixed_el_nodes) + 1
        # fixed_el_nodes_dof_z = (3 * fixed_el_nodes) + 2
        # fixed_el_dof = np.hstack([
        #     fixed_el_nodes_dof_x,
        #     fixed_el_nodes_dof_y,
        #     fixed_el_nodes_dof_z,
        # ])

        # 1. Loaded DOFs
        mechanical_load = 10000.0 / len(nodes_BF.tolist()) # Newtons
        # loaded_el_nodes = np.array([node_TF_BR])
        loaded_el_nodes = nodes_BF
        loaded_el_nodes_dof_x = (3 * loaded_el_nodes) + 0
        loaded_el_nodes_dof_y = (3 * loaded_el_nodes) + 1
        loaded_el_nodes_dof_z = (3 * loaded_el_nodes) + 2
        loaded_el_dof = np.hstack([
            # loaded_el_nodes_dof_x,
            loaded_el_nodes_dof_y,
            # loaded_el_nodes_dof_z,
        ])

        # 2. Fixed DOFs
        fixed_el_nodes = nodes_TF
        fixed_el_nodes_dof_x = (3 * fixed_el_nodes) + 0
        fixed_el_nodes_dof_y = (3 * fixed_el_nodes) + 1
        fixed_el_nodes_dof_z = (3 * fixed_el_nodes) + 2
        fixed_el_dof = np.hstack([
            fixed_el_nodes_dof_x,
            fixed_el_nodes_dof_y,
            fixed_el_nodes_dof_z,
        ])

        return {
            'fixed_therm_nodes': fixed_therm_nodes,
            'fixed_therm_temp': fixed_therm_temp,
            'heat_gen_nodes': heat_gen_nodes,
            'heat_gen_tref': heat_gen_tref,

            'mechanical_load': mechanical_load,
            'loaded_el_dof': loaded_el_dof,
            'fixed_el_dof': fixed_el_dof,

            'loaded_el_nodes': loaded_el_nodes,
            'fixed_el_nodes': fixed_el_nodes,
        }

    def solve_physics(self, x):
        """
        Solves:
        A. Steady-State Heat Transfer -> T1
        B. Mechanical Equilibrium (Thermal Load & Resistive) -> U1
        """

        K_mech, K_therm, C_coup = self._assemble_system(x)
        boundary_conditions = self._build_boundary_conditions()
        # print('finished assembling')

        # --- A. Steady-State Heat Transfer (Neumann BCs) ---

        # 1. Define Heatsinks (Dirichlet BCs)
        fixed_therm_nodes = boundary_conditions['fixed_therm_nodes']
        fixed_therm_temp = boundary_conditions['fixed_therm_temp']
        fixed_therm_values = np.zeros(self.ndof_therm)
        fixed_therm_values[fixed_therm_nodes] = fixed_therm_temp

        # Identify Free DOFs
        all_therm_dofs = np.arange(self.ndof_therm)
        free_therm_dofs = np.setdiff1d(all_therm_dofs, fixed_therm_nodes)

        # 2. Define Heat Generation (Neumann RHS)
        heat_gen_nodes = boundary_conditions['heat_gen_nodes']
        heat_gen_tref = boundary_conditions['heat_gen_tref']
        fth = np.zeros(self.ndof_therm)
        fth[heat_gen_nodes] = heat_gen_tref

        # 3. Solve the system
        K_therm_free = K_therm[free_therm_dofs, :][:, free_therm_dofs]
        fth_free = fth[free_therm_dofs]
        T1 = np.zeros(self.ndof_therm)
        T1[free_therm_dofs] = self.sparse_solver(K_therm_free.tocsr(), fth_free)
        # print('solved thermal system')

        # --- B. Mechanical Equilibrium (Thermal Load Ft & Resistive load Fe) ---

        # 1. Calculate Force due to Thermal Expansion
        Ft = C_coup @ T1
        # print('Thermal expansion forces', Ft)

        # 2. Define Mechanical External Forces
        mechanical_load = boundary_conditions['mechanical_load']
        loaded_el_dof = boundary_conditions['loaded_el_dof']

        F_mech_ext = np.zeros(self.ndof_mech)
        F_mech_ext[loaded_el_dof] = mechanical_load

        # 3. Combine Mechanical Forces and Thermal Forces
        # F_total = F_mech_ext + Ft
        F_total = F_mech_ext

        # 4. Define Fixed Elastic Nodes
        fixed_el_dof = boundary_conditions['fixed_el_dof']

        # Identify Free DOFs
        all_mech_dofs = np.arange(self.ndof_mech)
        free_mech_dofs = np.setdiff1d(all_mech_dofs, fixed_el_dof)

        # 4. Solve the system
        K_mech_free = K_mech[free_mech_dofs, :][:, free_mech_dofs]
        F_total_free = F_total[free_mech_dofs]
        U1 = np.zeros(self.ndof_mech)
        U1[free_mech_dofs] = self.sparse_solver(K_mech_free.tocsr(), F_total_free)
        # print('solved mechanical system')

        return U1, T1, F_mech_ext, Ft, K_mech, K_therm, C_coup

    def sensitivity_analysis(self, x, U1, T1, F_mech_ext, Ft, K_mech, K_therm, C_coup):
        """
        Calculates gradients using Coupled Adjoint Method.
        Objective J = J_mech + J_therm = (U K U) + (T Kt T)

        Properly accounts for:
        1. Stiffness changes (mech + therm)
        2. Changes in thermal load due to density (dC/dx)
        3. Changes in mech response due to temp field variation (Adjoint Lambda)
        """
        boundary_conditions = self._build_boundary_conditions()

        # --- 1. Objective Function Values ---
        # J_mech = U^T (F_ext + F_therm)
        # Note: U1.K.U1 is equivalent to U1.F_total
        F_total = F_mech_ext + Ft
        obj_mech = np.dot(U1, F_total)

        # J_therm = T^T F_heat
        # We need the thermal load vector used in the forward solve.
        # Assuming we can reconstruct it or pass it.
        # Ideally, pass f_heat or calculate T.Kt.T directly:
        obj_therm = np.dot(T1, K_therm @ T1)

        # obj_total = obj_mech + obj_therm
        obj_total = (self.el_weight * obj_mech) + (self.th_weight * obj_therm)
        # print('Mech objective', obj_mech, '| Thermal objective', obj_therm)

        # --- 2. Solve Thermal Adjoint Equation ---
        # We need lambda_adj such that: K_therm * lambda = 2 * C_coup.T * U
        # The RHS represents the sensitivity of Mech Compliance w.r.t Temperature

        # RHS_adj = 2 * C^T * U
        # C_coup is (3N, N), U is (3N,), so result is (N,)
        rhs_adj = 2.0 * (C_coup.T @ U1)

        # Identify free thermal DOFs (same as forward solve)
        # Re-using boundaries from forward solve logic
        fixed_therm_nodes = boundary_conditions['fixed_therm_nodes']
        all_therm_dofs = np.arange(self.ndof_therm)
        free_therm_dofs = np.setdiff1d(all_therm_dofs, fixed_therm_nodes)

        # Solve for Lambda (Adjoint Temperature)
        lambda_adj = np.zeros(self.ndof_therm)
        Kt_free = K_therm[free_therm_dofs, :][:, free_therm_dofs]
        rhs_free = rhs_adj[free_therm_dofs]

        lambda_adj[free_therm_dofs] = self.sparse_solver(Kt_free.tocsr(), rhs_free)

        # --- 3. Vectorized Sensitivity Calculation ---

        # Material Derivatives (SIMP p=3)
        # E(x) ~ x^p  -> dE/dx = p * x^(p-1) * (E0-Emin)
        # k(x) ~ x^p  -> dk/dx = p * x^(p-1) * (k0-kmin)
        # C(x) ~ x^p  -> scales same as E (Young's Modulus)

        x_flat = x.flatten(order='F')
        xp_deriv = self.penal * (x_flat ** (self.penal - 1))

        dE = xp_deriv * (self.E0 - self.Emin)
        dk = xp_deriv * (self.k0 - self.kmin)
        # C_coup depends on E, so it uses dE scaling

        # A. Get Element Vectors (No loops!)
        # Shape: (Num_Elems, N_DOF_Per_Elem)
        U_e = U1[self.edofMat_mech]  # (Nelem, 24)
        T_e = T1[self.edofMat_therm]  # (Nelem, 8)
        L_e = lambda_adj[self.edofMat_therm]  # Adjoint Lambda (Nelem, 8)

        # B. Mechanical Stiffness Term: - U^T * dK_m * U
        # Compute (U_e @ Ke0) * U_e efficiently
        # axis=1 sum simulates the dot product for each element
        term1_mech = -1.0 * np.sum((U_e @ self.Ke0) * U_e, axis=1) * dE

        # C. Thermal Stiffness Term: - (Lambda + T)^T * dK_t * T
        # Note: Standard thermal compliance sens is -T K' T.
        # The adjoint adds the -L K' T part.
        # Combined vector V = (Lambda + T)
        V_e = L_e + T_e
        term2_therm = -1.0 * np.sum((V_e @ self.Kt0) * T_e, axis=1) * dk

        # D. Coupling Load Term: + 2 * U^T * dC * T
        # dC = dE * c_ethm0
        # Calculation: 2 * U_e^T * c_ethm0 * T_e
        # U_e is (N, 24), c_ethm0 is (24, 8), T_e is (N, 8)
        # Step 1: temp = U_e @ c_ethm0  -> (N, 8)
        # Step 2: dot(temp, T_e)        -> (N,)
        coupling_prod = np.sum((U_e @ self.Cethm0) * T_e, axis=1)
        term3_coup = 2.0 * coupling_prod * dE * self.alpha

        # --- 4. Total Gradient ---
        # dc = term1_mech + term2_therm + term3_coup
        dc = (self.el_weight * (term1_mech + term3_coup)) + (self.th_weight * term2_therm)

        return dc, obj_total, obj_mech, obj_therm

    def filter_sensitivity(self, x, dfdx):
        xval = x.reshape(-1, 1)
        df0dx_vec = dfdx.reshape(-1, 1)
        df0dx_filt = (self.H @ (xval * df0dx_vec)) / self.hs[:, None] / np.maximum(1e-3, xval)
        return df0dx_filt

    def optimize(self, max_iter=100):

        print(f"Starting Optimization: Grid {self.nelx}x{self.nely}x{self.nelz}")

        for k in range(max_iter):
            t0 = time.time()

            # 1. Physics
            U1, T1, F_mech_ext, Ft, K_mech, K_therm, C_coup = self.solve_physics(self.x)
            self.plot_physics(U1)

            # 2. Sensitivity
            dfdx, f_val, f_val_mech, f_val_therm = self.sensitivity_analysis(self.x, U1, T1, F_mech_ext, Ft, K_mech, K_therm, C_coup)

            # 3. Filtering
            dfdx_filt = self.filter_sensitivity(self.x, dfdx)

            # 4. Constraints
            current_vol = np.sum(self.x)
            g_val = (current_vol / (self.volfrac * self.num_elems)) - 1.0
            dgdx = np.ones(self.num_elems) / (self.volfrac * self.num_elems)

            # 5. MMA Step
            inputs = MMAInputs(
                m=1,
                n=self.num_elems,
                iterr=k,
                xval=self.x,
                xmin=0.0,
                xmax=1.0,
                xold1=self.xold1,
                xold2=self.xold2,
                df0dx=dfdx_filt[:, 0],
                fval=np.array([g_val]),
                dfdx=dgdx.reshape(1, -1),
                low=self.low,
                upp=self.upp,
                a0=1.0,
                a=self.a_mma,
                c=self.c_mma,
                d=self.d_mma,
                f0val=f_val
            )

            # Call MMA
            x_new, low_new, upp_new = mmasub(inputs)

            # Update history
            self.xold2 = self.xold1.copy()
            self.xold1 = self.x.copy()
            self.low = low_new
            self.upp = upp_new


            # change = np.max(np.abs(x_new - self.x))
            change = 1.0
            self.x = x_new


            # End of Iteration Metrics
            print(f"Iter: {k} | Obj: {f_val:.4e} | Mech: {f_val_mech:.4e} | Therm: {f_val_therm:.4e} | Vol: {current_vol / (self.num_elems):.3f} | Change: {change:.4f}")
            if change < 0.001 and k > 5:
                print("Convergence reached.")
                break

            if k % 50 == 0:
                self.save()

        self.save()


    def plot_physics(self, U1):
        design = np.reshape(self.x, (self.nelz, self.nelx, self.nely))


        # Convert U1 from mm to meters
        U1 = U1 * 0.001

        # Get even elements of U1 (x displacements)
        Ux = U1[0::3]
        Uy = U1[1::3]
        Uz = U1[2::3]

        Ux = np.reshape(Ux, (self.nelz+1, self.nelx+1, self.nely+1))
        Uy = np.reshape(Uy, (self.nelz+1, self.nelx+1, self.nely+1))
        Uz = np.reshape(Uz, (self.nelz+1, self.nelx+1, self.nely+1))

        print('UX min/max:', np.min(Ux), np.max(Ux))
        print('UY min/max:', np.min(Uy), np.max(Uy))
        print('UZ min/max:', np.min(Uz), np.max(Uz))



        disp = np.sqrt(Ux**2 + Uy**2 + Uz**2)

        boundary_conditions = self._build_boundary_conditions()

        fixed_el_nodes = np.zeros(self.ndof_therm)
        fixed_el_nodes[boundary_conditions['fixed_el_nodes']] = 1
        fixed_el_nodes = np.reshape(fixed_el_nodes, (self.nelz + 1, self.nelx + 1, self.nely + 1))

        loaded_el_nodes = np.zeros(self.ndof_therm)
        loaded_el_nodes[boundary_conditions['loaded_el_nodes']] = 1
        loaded_el_nodes = np.reshape(loaded_el_nodes, (self.nelz + 1, self.nelx + 1, self.nely + 1))

        import napari
        viewer = napari.Viewer()
        viewer.add_image(design, name='rho', rendering='attenuated_mip')
        viewer.add_image(fixed_el_nodes, name='fixed_elements', rendering='attenuated_mip', visible=False, colormap='green')
        viewer.add_image(loaded_el_nodes, name='force_elements', rendering='attenuated_mip', visible=False, colormap='fire')

        viewer.add_image(Ux, name='Ux', rendering='mip', visible=False, colormap='blue')
        viewer.add_image(Uy, name='Uy', rendering='mip', visible=False, colormap='green')
        viewer.add_image(Uz, name='Uz', rendering='mip', visible=False, colormap='red')
        viewer.add_image(disp, name='Displacement Magnitude', rendering='attenuated_mip', visible=False, colormap='yellow')


        viewer.dims.ndisplay = 3  # switch to 3D view
        viewer.axes.visible = True
        napari.run()
        exit(0)





    def save(self):
        # design = np.reshape(self.x, (self.nelx, self.nely, self.nelz))
        design = np.reshape(self.x, (self.nelz, self.nelx, self.nely))

        boundary_conditions = self._build_boundary_conditions()

        fixed_therm_nodes = np.zeros(self.ndof_therm)
        fixed_therm_nodes[boundary_conditions['fixed_therm_nodes']] = 1
        # fixed_therm_nodes = np.reshape(fixed_therm_nodes, (self.nelx + 1, self.nely + 1, self.nelz + 1))
        fixed_therm_nodes = np.reshape(fixed_therm_nodes, (self.nelz + 1, self.nelx + 1, self.nely + 1))

        heat_gen_nodes = np.zeros(self.ndof_therm)
        heat_gen_nodes[boundary_conditions['heat_gen_nodes']] = 1
        # heat_gen_nodes = np.reshape(heat_gen_nodes, (self.nelx + 1, self.nely + 1, self.nelz + 1))
        heat_gen_nodes = np.reshape(heat_gen_nodes, (self.nelz + 1, self.nelx + 1, self.nely + 1))

        fixed_el_nodes = np.zeros(self.ndof_therm)
        fixed_el_nodes[boundary_conditions['fixed_el_nodes']] = 1
        # fixed_el_nodes = np.reshape(fixed_el_nodes, (self.nelx + 1, self.nely + 1, self.nelz + 1))
        fixed_el_nodes = np.reshape(fixed_el_nodes, (self.nelz + 1, self.nelx + 1, self.nely + 1))

        loaded_el_nodes = np.zeros(self.ndof_therm)
        loaded_el_nodes[boundary_conditions['loaded_el_nodes']] = 1
        # loaded_el_nodes = np.reshape(loaded_el_nodes, (self.nelx + 1, self.nely + 1, self.nelz + 1))
        loaded_el_nodes = np.reshape(loaded_el_nodes, (self.nelz + 1, self.nelx + 1, self.nely + 1))

        # Now save all these to a numpy .npz file
        numpy_file = os.path.join(NUMPY_SAVE_DIR, self.fname)
        np.savez(
            numpy_file,
            design=design,
            fixed_therm_nodes=fixed_therm_nodes,
            heat_gen_nodes=heat_gen_nodes,
            fixed_el_nodes=fixed_el_nodes,
            loaded_el_nodes=loaded_el_nodes
        )
        print(f"Successfully saved all arrays to: {numpy_file}")


        if self.plot is True:
            import napari
            viewer = napari.Viewer()
            viewer.add_image(design, name='rho', rendering='attenuated_mip')
            viewer.add_image(fixed_el_nodes, name='fixed_elements', rendering='attenuated_mip', visible=False, colormap='green')
            viewer.add_image(loaded_el_nodes, name='force_elements', rendering='attenuated_mip', visible=False, colormap='fire')
            viewer.add_image(fixed_therm_nodes, name='heatsink_elements', rendering='attenuated_mip', visible=False, colormap='purple')
            viewer.add_image(heat_gen_nodes, name='heat_gen_elements', rendering='attenuated_mip', visible=False, colormap='blue')
            viewer.dims.ndisplay = 3  # switch to 3D view
            viewer.axes.visible = True
            napari.run()
            exit(0)






def parse_arguments():
    """
    Parses command-line arguments using the argparse module.
    """
    parser = argparse.ArgumentParser(
        description="A program to process material properties and a filename."
    )
    parser.add_argument(
        '--volume_fraction',
        type=float,
        help='The volume fraction (a floating-point number).',
        default=0.3
    )
    parser.add_argument(
        '--weight',
        type=float,
        help='The weight value (a floating-point number).',
        default=0.5
    )
    parser.add_argument(
        '--fname',
        type=str,
        help='The output filename (a string).',
        default='test_design'
    )
    args = parser.parse_args()
    return args




if __name__ == '__main__':


    # Ansys Testing Steup
    nelx, nely, nelz = 2, 4, 40
    volfrac = 1.0
    penal = 3.0
    rmin = 1.5
    el_weight = 0.5
    fname = 'test_design.npz'
    plot = True

    l_ele = 50.0
    opt = ThermoelasticTopologyOptimization3D(
        nelx, nely, nelz,
        volfrac, penal, rmin,
        lx=l_ele, ly=l_ele, lz=l_ele,
        iter_solve=True,
        fname=fname,
        el_weight=el_weight,
        plot=plot
    )

    opt.optimize(max_iter=200)

    # # Testing Setup
    # nelx, nely, nelz = 20, 20, 20
    # # nelx, nely, nelz = 100, 40, 16
    # volfrac = 0.3
    # penal = 3.0
    # rmin = 1.5
    # el_weight = 0.5
    # fname = 'test_design.npz'
    # plot = True

    # # Datagen Setup
    # nelx, nely, nelz = 100, 40, 16
    # # nelx, nely, nelz = 20, 20, 20
    # args = parse_arguments()
    # volfrac = float(args.volume_fraction)
    # penal = 3.0
    # rmin = 1.5
    # el_weight = float(args.weight)
    # fname = str(args.fname) + '.npz'
    # plot = False






