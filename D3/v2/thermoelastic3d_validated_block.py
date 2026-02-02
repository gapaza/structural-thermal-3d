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
    def __init__(self, nelx, nely, nelz, volfrac, penal, rmin, lx=1.0, ly=1.0, lz=1.0, iter_solve=False, fname='design.npz', el_weight=0.5, plot=False, f=1.0):
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

        self.f = f

        if iter_solve is False:
            self.sparse_solver = spsolve
        else:
            self.sparse_solver = solve_spd_with_amg

        # Material Properties (Arbitrary units for example, polypropylene units in comments)
        self.nu = 0.3       # Poisson ratio (0.45)
        self.E0 = 2800       # Young's Modulus (2e5 MPa -> 2e11 Pa)  ->? was 2800
        self.Emin = 1e-9    # Void stiffness

        # self.rho0 = 7850.0  # Density (930 kg/m^3) WANT kg/mm^3
        # self.rho0 = 7.85e-6  # Density (7.85e-6 kg/mm^3 = 7850 kg/m^3)
        self.rho0 = 1.21e-6

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
        self.Ke0, self.Kt0, self.Cethm0 = get_stiffness_matrices(self.nu, 1.0, 1.0, 1.0, lx=self.lx, ly=self.ly, lz=self.lz)

        # DOF Mappings
        self._build_edof_matrices()

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

        # n1 = (ely + 1) + elx * ny_n + elz * ny_n * nx_n        # Bottom left
        # n2 = (ely + 1) + (elx + 1) * ny_n + elz * ny_n * nx_n  # Bottom right
        # n3 = ely + (elx + 1) * ny_n + elz * ny_n * nx_n        # Top right
        # n4 = ely + elx * ny_n + elz * ny_n * nx_n              # Top left

        n1 = (ely) + elx * ny_n + elz * ny_n * nx_n            # Top left
        n2 = (ely) + (elx + 1) * ny_n + elz * ny_n * nx_n      # Top right
        n3 = (ely + 1) + (elx + 1) * ny_n + elz * ny_n * nx_n  # Bottom right
        n4 = (ely + 1) + elx * ny_n + elz * ny_n * nx_n        # Bottom left

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

        node_TF_T_FQ = (self.nely + 1) * (self.nelx//4)
        node_TF_B_TQ = (self.nely + 1) * (3 * self.nelx//4) + self.nely
        nodes_TF_Q = np.arange(node_TF_T_FQ, node_TF_B_TQ + 1)


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

        # 1. Loaded DOFs
        # mechanical_load = -0.001
        mechanical_load = self.f / len(nodes_TF.tolist())  # Newtons
        # mechanical_load = self.f / len(nodes_TF_Q.tolist()) # Newtons
        # loaded_el_nodes = np.array([node_TF_BR])
        loaded_el_nodes = nodes_TF
        # loaded_el_nodes = nodes_TF_Q
        loaded_el_nodes_dof_x = (3 * loaded_el_nodes) + 0
        loaded_el_nodes_dof_y = (3 * loaded_el_nodes) + 1
        loaded_el_nodes_dof_z = (3 * loaded_el_nodes) + 2
        loaded_el_dof = np.hstack([
            # loaded_el_nodes_dof_x,
            # loaded_el_nodes_dof_y,
            loaded_el_nodes_dof_z,
        ])

        # 2. Fixed DOFs
        # fixed_el_nodes = np.array([node_TF_TL, node_TF_BL, node_BF_TL, node_BF_BL])
        # fixed_el_nodes = np.array([node_BF_TL, node_BF_BL, node_BF_TR, node_BF_BR]) # Rings BCs
        # fixed_el_nodes = np.array([node_BF_TL]) # Single Fixed Corner
        # fixed_el_nodes = np.array([node_BF_TL, node_BF_TL+1, node_BF_TL+2]) # Multi Fixed Corner
        fixed_el_nodes = nodes_BF
        # fixed_el_nodes = np.array([node_BF_TL, node_BF_BL])

        # fixed_el_nodes_dof_x = (3 * fixed_el_nodes) + 0
        # fixed_el_nodes_dof_y = (3 * fixed_el_nodes) + 1
        fixed_el_nodes_dof_z = (3 * fixed_el_nodes) + 2

        fixed_el_nodes_xy = node_BF_TL
        fixed_el_nodes_xy_dof_x = (3 * fixed_el_nodes_xy) + 0
        fixed_el_nodes_xy_dof_y = (3 * fixed_el_nodes_xy) + 1

        fixed_el_dof = np.hstack([
            fixed_el_nodes_xy_dof_x,
            fixed_el_nodes_xy_dof_y,

            # fixed_el_nodes_dof_x,
            # fixed_el_nodes_dof_y,
            fixed_el_nodes_dof_z,
        ])

        # # 1. Loaded DOFs
        # mechanical_load = 10000.0 / len(nodes_BF.tolist()) # Newtons
        # # loaded_el_nodes = np.array([node_TF_BR])
        # loaded_el_nodes = nodes_BF
        # loaded_el_nodes_dof_x = (3 * loaded_el_nodes) + 0
        # loaded_el_nodes_dof_y = (3 * loaded_el_nodes) + 1
        # loaded_el_nodes_dof_z = (3 * loaded_el_nodes) + 2
        # loaded_el_dof = np.hstack([
        #     # loaded_el_nodes_dof_x,
        #     loaded_el_nodes_dof_y,
        #     # loaded_el_nodes_dof_z,
        # ])
        #
        # # 2. Fixed DOFs
        # fixed_el_nodes = nodes_TF
        # fixed_el_nodes_dof_x = (3 * fixed_el_nodes) + 0
        # fixed_el_nodes_dof_y = (3 * fixed_el_nodes) + 1
        # fixed_el_nodes_dof_z = (3 * fixed_el_nodes) + 2
        # fixed_el_dof = np.hstack([
        #     fixed_el_nodes_dof_x,
        #     fixed_el_nodes_dof_y,
        #     fixed_el_nodes_dof_z,
        # ])

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

            'fixed_el_nodes_xy': fixed_el_nodes_xy,
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

        numpy_file = '/Users/gapaza/repos/ideal/structural-thermal-3d/D3/v2/designs/voxel_design.npz'
        data = np.load(numpy_file)
        design = data['design']
        # design = (design >= 0.5).astype(float)
        self.x = np.reshape(design, (-1))

        # For the solid block
        self.x = np.ones_like(self.x)

        for k in range(max_iter):

            print('Initial design shape:', self.x.shape)

            t0 = time.time()

            # 1. Physics
            U1, T1, F_mech_ext, Ft, K_mech, K_therm, C_coup = self.solve_physics(self.x)
            return self.plot_physics(U1)


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
        # U1 = U1 * 0.001

        # Get even elements of U1 (x displacements)
        Ux = U1[0::3]
        Uy = U1[1::3]
        Uz = U1[2::3]

        node_TF_CENTER_E1 = ((self.nely + 1) * (self.nelx // 2)) + (self.nely // 2)
        node_TF_CENTER_E2 = node_TF_CENTER_E1 + 1
        node_TF_CENTER_E3 = node_TF_CENTER_E1 + 2
        # print('Center Node Displacement E1 (mm):', Ux[node_TF_CENTER_E1], Uy[node_TF_CENTER_E1], Uz[node_TF_CENTER_E1])
        # print('Center Node Displacement E2 (mm):', Ux[node_TF_CENTER_E2], Uy[node_TF_CENTER_E2], Uz[node_TF_CENTER_E2])
        # print('Center Node Displacement E3 (mm):', Ux[node_TF_CENTER_E3], Uy[node_TF_CENTER_E3], Uz[node_TF_CENTER_E3])

        Ux = np.reshape(Ux, (self.nelz+1, self.nelx+1, self.nely+1))
        Uy = np.reshape(Uy, (self.nelz+1, self.nelx+1, self.nely+1))
        Uz = np.reshape(Uz, (self.nelz+1, self.nelx+1, self.nely+1))

        # print('UX min/max/mean (mm):', np.min(Ux), np.max(Ux), np.mean(Ux))
        # print('UY min/max/mean (mm):', np.min(Uy), np.max(Uy), np.mean(Uy))
        # print('UZ min/max/mean (mm):', np.min(Uz), np.max(Uz), np.mean(Uz))

        return np.max(Uz)



        # disp = np.sqrt(Ux**2 + Uy**2 + Uz**2)
        #
        # boundary_conditions = self._build_boundary_conditions()
        #
        # fixed_el_nodes = np.zeros(self.ndof_therm)
        # fixed_el_nodes[boundary_conditions['fixed_el_nodes']] = 1
        # fixed_el_nodes = np.reshape(fixed_el_nodes, (self.nelz + 1, self.nelx + 1, self.nely + 1))
        #
        # fixed_el_nodes_xy = np.zeros(self.ndof_therm)
        # fixed_el_nodes_xy[boundary_conditions['fixed_el_nodes_xy']] = 1
        # fixed_el_nodes_xy = np.reshape(fixed_el_nodes_xy, (self.nelz + 1, self.nelx + 1, self.nely + 1))
        #
        # loaded_el_nodes = np.zeros(self.ndof_therm)
        # loaded_el_nodes[boundary_conditions['loaded_el_nodes']] = 1
        # loaded_el_nodes = np.reshape(loaded_el_nodes, (self.nelz + 1, self.nelx + 1, self.nely + 1))
        #
        # import napari
        # viewer = napari.Viewer()
        # viewer.add_image(design, name='rho', rendering='attenuated_mip')
        # viewer.add_image(fixed_el_nodes, name='fixed_elements', rendering='attenuated_mip', visible=False, colormap='green')
        # viewer.add_image(loaded_el_nodes, name='force_elements', rendering='attenuated_mip', visible=False, colormap='fire')
        # viewer.add_image(fixed_el_nodes_xy, name='fixed_xy_elements', rendering='attenuated_mip', visible=False, colormap='cyan')
        #
        # viewer.add_image(Ux, name='Ux', rendering='mip', visible=False, colormap='blue')
        # viewer.add_image(Uy, name='Uy', rendering='mip', visible=False, colormap='green')
        # viewer.add_image(Uz, name='Uz', rendering='mip', visible=False, colormap='red')
        # viewer.add_image(disp, name='Displacement Magnitude', rendering='attenuated_mip', visible=False, colormap='yellow')
        #
        #
        # viewer.dims.ndisplay = 3  # switch to 3D view
        # viewer.axes.visible = True
        # napari.run()
        # exit(0)





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

    import json
    # forces = [0.101, 0.121, 0.125, 0.149, 0.159, 0.182, 0.2, 0.21, 0.244, 0.245, 0.296, 0.288, 0.348, 0.332, 0.385, 0.378, 0.429, 0.43, 0.48, 0.489, 0.514, 0.54, 0.559, 0.627, 0.631, 0.695, 0.685, 0.77, 0.749, 0.83, 0.824, 0.889, 0.892, 0.959, 0.973, 1.009, 1.04, 1.057, 1.137, 1.149, 1.231, 1.227, 1.364, 1.338, 1.441, 1.435, 1.537, 1.55, 1.64, 1.665, 1.736, 1.775, 1.824, 1.901, 1.919, 2.057, 2.057, 2.22, 2.182, 2.338, 2.311, 2.454, 2.457, 2.575, 2.613, 2.688, 2.735, 2.805, 2.901, 2.93, 3.074, 3.074, 3.261, 3.239, 3.399, 3.386, 3.545, 3.554, 3.715, 3.752, 3.845, 3.931, 4.002, 4.119, 4.167, 4.334, 4.329, 4.546, 4.525, 4.7, 4.686, 4.843, 4.87, 5.005, 5.049, 5.154, 5.24, 5.289, 5.432, 5.464, 5.649, 5.655, 5.872, 5.85, 6.034, 6.032, 6.211, 6.231, 6.403, 6.442, 6.589, 6.658, 6.751, 6.876, 6.919, 7.131, 7.15, 7.398, 7.376, 7.581, 7.564, 7.759, 7.777, 7.953, 8.013, 8.149, 8.211, 8.302, 8.42, 8.483, 8.695, 8.702, 8.957, 8.936, 9.16, 9.143, 9.357, 9.376, 9.581, 9.639, 9.784, 9.897, 9.998, 10.17, 10.234, 10.438, 10.437, 10.7, 10.686, 10.917, 10.92, 11.13, 11.149, 11.343, 11.399, 11.534, 11.63, 11.699, 11.877, 11.923, 12.138, 12.149, 12.423, 12.403, 12.628, 12.631, 12.851, 12.876, 13.085, 13.143, 13.317, 13.388, 13.514, 13.664, 13.725, 13.981, 14.012, 14.302, 14.277, 14.534, 14.514, 14.757, 14.774, 14.98, 15.046, 15.217, 15.28, 15.398, 15.548, 15.618, 15.854, 15.865, 16.159, 16.138, 16.393, 16.382, 16.628, 16.65, 16.888, 16.939, 17.116, 17.245, 17.354, 17.526, 17.594, 17.829, 17.831, 18.137, 18.121, 18.363, 18.367, 18.611, 18.628, 18.841, 18.896, 19.039, 19.133, 19.221, 19.41, 19.469, 19.701, 19.715]
    forces = [0.033, 0.045, 0.124, 0.126, 0.179, 0.211, 0.263, 0.259, 0.319, 0.352, 0.347, 0.446, 0.446, 0.441, 0.558, 0.593, 0.587, 0.584, 0.619, 0.88, 0.876, 0.869, 0.864, 0.859, 0.861, 0.865, 0.857, 1.016, 1.038, 1.036, 1.035, 1.033, 1.042, 1.186, 1.191, 1.189, 1.189, 1.207, 1.369, 1.376, 1.375, 1.371, 1.369, 1.48, 1.544, 1.545, 1.543, 1.541, 1.565, 1.694, 1.701, 1.698, 1.698, 1.722, 1.935, 2.065, 2.07, 2.068, 2.066, 2.063, 2.068, 2.061, 2.068, 2.061, 2.058, 2.243, 2.388, 2.399, 2.4, 2.398, 2.394, 2.393, 2.394, 2.395, 2.392, 2.414, 2.5, 2.506, 2.506, 2.757, 2.895, 2.896, 2.891, 2.891, 2.885, 2.893, 2.896, 2.898, 2.896, 2.893, 2.887, 2.97, 3.205, 3.219, 3.217, 3.216, 3.214, 3.224, 3.219, 3.223, 3.218, 3.203, 3.653, 3.771, 3.772, 3.772, 3.77, 3.765, 3.766, 3.77, 3.772, 3.772, 3.769, 3.763, 3.759, 3.754, 3.751, 3.743, 3.781, 3.877, 3.885, 3.886, 3.885, 4.065, 4.078, 4.079, 4.079, 4.078, 4.08, 4.261, 4.291, 4.293, 4.295, 4.291, 4.293, 4.289, 4.446, 4.462, 4.466, 4.464, 4.464, 4.565, 4.66, 4.665, 4.668, 4.665, 4.668, 4.67, 5.015, 5.046, 5.056, 5.057, 5.055, 5.054, 5.056, 5.061, 5.055, 5.051, 5.044, 5.116, 5.227, 5.235, 5.236, 5.237, 5.237, 5.255, 5.421, 5.439, 5.439, 5.44, 5.44, 5.447, 5.601, 5.848, 5.852, 5.854, 5.851, 5.851, 5.849, 5.853, 5.856, 5.855, 5.852, 5.851, 5.841, 5.982, 5.997, 6.002, 6.0, 6.001, 6.15, 6.168, 6.172, 6.172, 6.171, 6.303, 6.407, 6.413, 6.415, 6.416, 6.417, 6.418, 6.418, 6.705, 6.865, 6.87, 6.872, 6.87, 6.872, 6.867, 6.877, 6.875, 6.869, 6.863, 6.862, 6.856, 6.846, 7.087, 7.251, 7.258, 7.26, 7.258, 7.26, 7.26, 7.265, 7.268, 7.262, 7.258, 7.257, 7.32, 7.451, 7.46, 7.461, 7.463, 7.464, 7.462, 7.59, 7.625, 7.629, 7.631, 7.63, 7.781, 8.016, 8.03, 8.033, 8.033, 8.033, 8.035, 8.041, 8.045, 8.037, 8.038, 8.035, 8.034, 8.166, 8.215, 8.216, 8.222, 8.219, 8.226, 8.449, 8.624, 8.636, 8.642, 8.64, 8.641, 8.643, 8.644, 8.644, 8.636, 8.632, 8.633, 8.648, 8.759, 8.774, 8.779, 8.78, 8.781, 8.948, 8.963, 8.967, 8.969, 8.969, 8.977, 9.281, 9.298, 9.302, 9.304, 9.305, 9.304, 9.317, 9.323, 9.322, 9.318, 9.359, 9.509, 9.519, 9.52, 9.522, 9.521, 9.527, 9.715, 9.894, 9.903, 9.908, 9.908, 9.909, 9.907, 9.919, 9.922, 9.925, 9.922, 9.92, 9.906, 10.059, 10.083, 10.091, 10.09, 10.094, 10.231, 10.292, 10.3, 10.302, 10.302, 10.303, 10.313, 10.486, 10.515, 10.519, 10.522, 10.524, 10.527, 10.528, 10.672, 10.697, 10.703, 10.705, 10.704, 10.714, 10.882, 10.901, 10.91, 10.911, 10.914, 10.912, 11.009, 11.087, 11.1, 11.1, 11.103, 11.104, 11.217, 11.515, 11.541, 11.544, 11.548, 11.546, 11.55, 11.556, 11.556, 11.557, 11.557, 11.554, 11.554, 11.551, 11.653, 11.708, 11.716, 11.721, 11.72, 11.719, 12.127, 12.141, 12.144, 12.143, 12.144, 12.15, 12.159, 12.164, 12.172, 12.172, 12.167, 12.167, 12.165, 12.162, 12.427, 12.574, 12.581, 12.586, 12.586, 12.588, 12.601, 12.601, 12.604, 12.604, 12.601, 12.6, 12.597, 12.671, 13.207, 13.234, 13.236, 13.239, 13.238, 13.245, 13.247, 13.25, 13.256, 13.253, 13.25, 13.25, 13.247, 13.246, 13.243, 13.243, 13.24, 13.238, 13.236, 13.242, 13.535, 13.652, 13.659, 13.663, 13.663, 13.666, 13.675, 13.681, 13.691, 13.686, 13.686, 13.684, 13.684, 13.665, 13.863, 13.878, 13.886, 13.885, 13.888, 13.889, 13.956, 14.055, 14.063, 14.065, 14.069, 14.067, 14.242, 14.278, 14.29, 14.29, 14.294, 14.295, 14.301, 14.357, 14.487, 14.496, 14.504, 14.503, 14.507, 14.511, 14.743, 14.911, 14.923, 14.927, 14.931, 14.93, 14.934, 14.938, 14.942, 14.943, 14.941, 14.941, 14.939, 14.974, 15.103, 15.106, 15.106, 15.107, 15.104, 15.276, 15.289, 15.294, 15.294, 15.298, 15.301, 15.481, 15.654, 15.665, 15.667, 15.67, 15.67, 15.674, 15.673, 15.677, 15.675, 15.676, 15.679, 15.854, 15.879, 15.889, 15.887, 15.895, 15.905, 15.909, 16.087, 16.254, 16.263, 16.268, 16.27, 16.27, 16.279, 16.287, 16.286, 16.297, 16.291, 16.291, 16.371, 16.473, 16.486, 16.489, 16.492, 16.493, 16.491, 16.653, 16.682, 16.687, 16.69, 16.69, 16.694, 16.857, 16.897, 16.903, 16.907, 16.907, 16.907, 16.915, 17.107, 17.14, 17.149, 17.149, 17.153, 17.152, 17.16, 17.153, 17.324, 17.344, 17.349, 17.353, 17.353, 17.351, 17.509, 17.531, 17.541, 17.542, 17.544, 17.546, 17.908, 17.97, 17.981, 17.984, 17.986, 17.988, 17.987, 17.99, 18.0, 17.996, 17.996, 17.993, 17.993, 17.99, 18.113, 18.168, 18.174, 18.179, 18.178, 18.183, 18.3, 18.385, 18.394, 18.399, 18.401, 18.406, 18.42, 18.461, 18.849, 18.872, 18.873, 18.875, 18.874, 18.88, 18.883, 18.896, 18.902, 18.902, 18.899, 18.897, 18.895, 18.895, 18.926, 18.698]
    forces_used = []

    disp_real = [0.0, 0.001, 0.005, 0.005, 0.006, 0.008, 0.008, 0.008, 0.009, 0.009, 0.009, 0.011, 0.011, 0.011, 0.012, 0.012, 0.012, 0.012, 0.014, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.016, 0.017, 0.017, 0.017, 0.017, 0.017, 0.019, 0.019, 0.019, 0.019, 0.019, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.022, 0.022, 0.022, 0.022, 0.022, 0.023, 0.023, 0.023, 0.023, 0.023, 0.025, 0.026, 0.026, 0.026, 0.026, 0.026, 0.026, 0.026, 0.026, 0.026, 0.026, 0.026, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.031, 0.031, 0.031, 0.031, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.034, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.039, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.042, 0.044, 0.044, 0.044, 0.044, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.047, 0.047, 0.047, 0.047, 0.047, 0.047, 0.048, 0.048, 0.048, 0.048, 0.048, 0.048, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.051, 0.053, 0.053, 0.053, 0.053, 0.053, 0.053, 0.053, 0.053, 0.053, 0.053, 0.053, 0.055, 0.055, 0.055, 0.055, 0.055, 0.055, 0.056, 0.056, 0.056, 0.056, 0.056, 0.056, 0.056, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.059, 0.061, 0.061, 0.061, 0.061, 0.061, 0.062, 0.062, 0.062, 0.062, 0.062, 0.062, 0.064, 0.064, 0.064, 0.064, 0.064, 0.064, 0.064, 0.064, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.067, 0.069, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.072, 0.073, 0.073, 0.073, 0.073, 0.073, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.076, 0.078, 0.078, 0.078, 0.078, 0.078, 0.078, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.081, 0.083, 0.083, 0.083, 0.083, 0.083, 0.084, 0.084, 0.084, 0.084, 0.084, 0.084, 0.086, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.089, 0.089, 0.089, 0.089, 0.089, 0.089, 0.089, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.092, 0.094, 0.094, 0.094, 0.094, 0.094, 0.094, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097, 0.097, 0.098, 0.098, 0.098, 0.098, 0.098, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.101, 0.101, 0.101, 0.101, 0.101, 0.101, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.105, 0.106, 0.106, 0.106, 0.106, 0.106, 0.108, 0.109, 0.109, 0.109, 0.109, 0.109, 0.109, 0.109, 0.109, 0.109, 0.109, 0.109, 0.109, 0.109, 0.111, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.112, 0.116, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.119, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.122, 0.122, 0.122, 0.122, 0.122, 0.122, 0.122, 0.123, 0.123, 0.123, 0.123, 0.123, 0.123, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.126, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.13, 0.131, 0.131, 0.131, 0.131, 0.131, 0.133, 0.133, 0.133, 0.133, 0.133, 0.133, 0.133, 0.136, 0.136, 0.136, 0.136, 0.136, 0.136, 0.136, 0.136, 0.136, 0.136, 0.136, 0.137, 0.137, 0.137, 0.137, 0.137, 0.137, 0.137, 0.137, 0.141, 0.141, 0.141, 0.141, 0.141, 0.141, 0.141, 0.141, 0.141, 0.141, 0.141, 0.141, 0.142, 0.142, 0.142, 0.142, 0.142, 0.142, 0.144, 0.144, 0.144, 0.144, 0.144, 0.144, 0.144, 0.145, 0.145, 0.145, 0.145, 0.145, 0.145, 0.145, 0.147, 0.147, 0.147, 0.147, 0.147, 0.147, 0.147, 0.148, 0.148, 0.148, 0.148, 0.148, 0.148, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.153, 0.153, 0.153, 0.153, 0.153, 0.153, 0.153, 0.153, 0.153, 0.153, 0.153, 0.153, 0.153, 0.153, 0.155, 0.155, 0.155, 0.155, 0.155, 0.155, 0.156, 0.156, 0.156, 0.156, 0.156, 0.156, 0.156, 0.158, 0.159, 0.159, 0.159, 0.159, 0.159, 0.159, 0.159, 0.159, 0.159, 0.159, 0.159, 0.159, 0.159, 0.159, 0.161, 0.151]
    disp_sim = []

    results = []
    for f in forces:

        if f not in forces_used:
            forces_used.append(f)
            # Ansys Testing Steup
            nelx, nely, nelz = 100, 40, 16
            volfrac = 1.0
            penal = 3.0
            rmin = 1.5
            el_weight = 0.5
            fname = 'test_design.npz'
            plot = True

            l_ele = 0.15  # Element length in mm
            opt = ThermoelasticTopologyOptimization3D(
                nelx, nely, nelz,
                volfrac, penal, rmin,
                lx=l_ele, ly=l_ele, lz=l_ele,
                iter_solve=True,
                fname=fname,
                el_weight=el_weight,
                plot=plot,
                f=f
            )

            disp = opt.optimize(max_iter=200)
            print('Force Displacement (mm) for load', f, 'is:', disp)
        else:
            f_idx = forces_used.index(f)
            disp = disp_sim[f_idx]
            print('Using previous result for load', f, 'Displacement (mm):', disp)


        results.append([f, disp])
        disp_sim.append(disp)

        # Save results to json file
        with open('force_displacement_results_block.json', 'w') as json_file:
            json.dump(results, json_file)

        n_disps = len(disp_sim)
        # plot the real vs simulated displacements
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(forces[:n_disps], disp_real[:n_disps], label='Real Displacement', marker='o')
        plt.plot(forces[:n_disps], disp_sim, label='Simulated Displacement', marker='x')
        plt.xlabel('Force (N)')
        plt.ylabel('Displacement (mm)')
        plt.title('Force vs Displacement Comparison')
        plt.legend()
        plt.grid()
        plt.show()
        plt.close()



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










