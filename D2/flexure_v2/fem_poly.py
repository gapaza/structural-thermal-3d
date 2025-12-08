import numpy as np
from scipy.sparse import coo_matrix, linalg
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from numpy.typing import NDArray
import copy
import time
from D2.flexure_v2.utils import plot_design
from D2.flexure_v2.fem_model import ThermalFlexureTopologyOptimization, fe_melthm


class PolypropyleneOptimization(ThermalFlexureTopologyOptimization):
    def __init__(self, nelx, nely, volfrac, penal, rmin):
        super().__init__(nelx, nely, volfrac, penal, rmin)

        # --- 1. Material Properties (Polypropylene) ---
        # Dimensions: Assuming a 2cm x 2cm domain for physical realism
        dim_x = 0.02  # m
        dim_y = 0.02  # m
        self.elem_size = dim_x / nelx  # Assumed square elements

        # Mechanical
        self.E0 = 1.1e9  # 1.1 GPa
        self.Emin = 1e-6 * self.E0
        self.nu = 0.45

        # Thermal
        self.rho0 = 930.0  # kg/m3
        self.k0 = 0.22  # W/(m K)
        self.kmin = 1e-3 * self.k0
        self.Cp = 1900.0  # J/(kg K)
        self.alpha = 8.0e-5  # 1/K

        # --- 2. Re-Scaling Matrices for Physical Size ---
        # The FE helper assumes unit size (1x1). We must scale by element size.
        # Stiffness (2D): Independent of size? No, K ~ E * thickness.
        # For 2D Plane Stress with unit thickness, K is independent of element size L.
        # Conductivity: K ~ k * thickness. Independent of L.
        # Capacity: C = rho * Cp * Area. Area = L^2.

        area = self.elem_size ** 2
        # Scale the unit matrices pre-calculated in parent class
        self.Ce0 = self.Ce0 * (area * self.rho0 * self.Cp)  # Scale by mass

        # Note: Ke0 and Kt0 in 2D square elements are theoretically independent
        # of element size L, but depend on material props.
        # We re-calculate them here with real material props to be safe.
        self.Ke0, self.Kt0, self.Cethm0 = fe_melthm(self.nu, self.E0, self.k0, self.alpha)

        # IMPORTANT: Normalize Stiffness to prevent tiny numbers
        # We will solve K_scaled * U = F_scaled.
        # This keeps numbers close to 1.0 for the solver.
        self.Ke0_norm = self.Ke0 / self.E0

    def solve_physics(self, x):
        """
        Updated for Fixed Temperature BC (Dirichlet)
        """
        # Assemble matrices (using normalized stiffness)
        x_penal = x ** self.penal

        # Mechanical K (Normalized)
        sK = (self.Emin / self.E0 + x_penal * (1.0 - self.Emin / self.E0)).flatten()
        vals_K = np.kron(sK, self.Ke0_norm.flatten())
        K_mech = coo_matrix((vals_K, (self.iK_mech, self.jK_mech)),
                            shape=(self.ndof_mech, self.ndof_mech)).tocsc()

        # Thermal Conductivity (Real units)
        sKt = (self.kmin + x_penal * (self.k0 - self.kmin)).flatten()
        vals_Kt = np.kron(sKt, self.Kt0.flatten())
        K_therm = coo_matrix((vals_Kt, (self.iK_therm, self.jK_therm)),
                             shape=(self.ndof_therm, self.ndof_therm)).tocsc()

        # Heat Capacity (Real units)
        sC = (0.01 + x * 0.99).flatten()  # Linear mass
        vals_C = np.kron(sC, self.Ce0.flatten())
        C_mat = coo_matrix((vals_C, (self.iK_therm, self.jK_therm)),
                           shape=(self.ndof_therm, self.ndof_therm)).tocsc()

        # --- A. Transient Heat Transfer (Dirichlet BC) ---
        # BC: Left Edge (x=0) is fixed at Delta_T = 80 (100C - 20C)
        fixed_temp_nodes = np.arange(15, self.nely + 1)
        # fixed_temp_nodes = np.arange(0, 12)
        fixed_temp_val = 80.0

        # Initial Condition
        Phi = np.zeros(self.ndof_therm)
        Phi_history = [Phi.copy()]

        # System: (C + dt*Kt) Phi_new = C * Phi_old
        # Note: No Pt (heat source) vector anymore
        A_step = C_mat + self.dt * K_therm

        # Apply Dirichlet BC via Penalty Method for Time Stepping
        # Add large number to diagonal of fixed nodes, and Large * Value to RHS
        penalty = 1e12 * A_step.diagonal().mean()

        # Modify A_step for BCs
        # (For efficiency, this should be done once outside loop if x fixed,
        # but here we do it inside for clarity or if x changes)
        # We need A_step in LIL or CSR to modify
        A_step = A_step.tolil()

        # --- OLD BROKEN LINE ---
        # A_step[fixed_temp_nodes, fixed_temp_nodes] += penalty

        # --- NEW WORKING FIX ---
        for node in fixed_temp_nodes:
            A_step[node, node] += penalty

        A_step = A_step.tocsc()
        solve_transient = linalg.factorized(A_step)

        for t in range(self.num_time_steps):
            RHS = C_mat @ Phi
            # Apply BC to RHS
            RHS[fixed_temp_nodes] += penalty * fixed_temp_val

            Phi = solve_transient(RHS)
            Phi_history.append(Phi.copy())

        Phi_final = Phi_history[-1]

        # --- B. Mechanical Equilibrium ---

        # 1. Thermal Load
        Ft_global = np.zeros(self.ndof_mech)

        # Vectorized assembly of Ft
        # F_e = Integral( B^T E alpha Phi )
        # Note: We used Ke0_norm (divided by E0). We must consistency scale force.
        # Or, we solve K_norm * U = F_norm. F_norm = F_real / E0.
        # Cethm0 contains E0. So we divide Cethm0 by E0 to get Cethm_norm.
        Cethm_norm = self.Cethm0 / self.E0

        for i in range(self.num_elems):
            idx_m = self.edofMat_mech[i]
            idx_t = self.edofMat_therm[i]
            phi_e = Phi_final[idx_t]
            f_e = x_penal[i] * (Cethm_norm @ phi_e)
            Ft_global[idx_m] += f_e

        # 2. Boundary Conditions (Fix Bottom Left and Bottom Right corners)
        # This allows expansion but prevents rigid body motion
        n_BL = self.nely
        n_BR = self.nely + (self.nelx) * (self.nely + 1)
        fixed_dofs = np.array([2 * n_BL, 2 * n_BL + 1, 2 * n_BR + 1])  # Fix BL(x,y) and BR(y)
        free_dofs = np.setdiff1d(np.arange(self.ndof_mech), fixed_dofs)

        K_free = K_mech[free_dofs, :][:, free_dofs]

        # Solve U1 (Thermal)
        U1 = np.zeros(self.ndof_mech)
        U1[free_dofs] = spsolve(K_free, Ft_global[free_dofs])

        # Solve U2 (Dummy Output - Top Middle +Y)
        out_node = self.nelx * (self.nely + 1)  # Top Middle
        out_dof = 2 * out_node
        P2 = np.zeros(self.ndof_mech)
        P2[out_dof] = 1.0 / self.E0  # Normalize force
        U2 = np.zeros(self.ndof_mech)
        U2[free_dofs] = spsolve(K_free, P2[free_dofs])

        # Solve U3 (Input Stiffness - Same point)
        P3 = np.zeros(self.ndof_mech)
        P3[out_dof] = -1.0 / self.E0
        U3 = np.zeros(self.ndof_mech)
        U3[free_dofs] = spsolve(K_free, P3[free_dofs])

        # Store for sensitivity
        # Note: Pt vector is now None or 0, as we use Dirichlet
        return U1, U2, U3, Phi_history, K_mech, K_therm, C_mat, K_free, free_dofs, None


import matplotlib.pyplot as plt


def run_thermal_calibration(optimizer, max_time_seconds=200, steps=100):
    """
    Runs only the thermal part to find the optimal actuation time tf.
    """
    print("Running Thermal Calibration...")

    # Temporarily override time settings
    optimizer.dt = max_time_seconds / steps
    optimizer.num_time_steps = steps

    # Solid material (x=1)
    x_solid = np.ones(optimizer.num_elems)

    # Run physics (We only care about Phi_history)
    # We catch the output but only use Phi_history
    _, _, _, Phi_hist, _, _, _, _, _, _ = optimizer.solve_physics(x_solid)

    variances = []
    times = []

    for t in range(len(Phi_hist)):
        phi = Phi_hist[t]
        # Calculate standard deviation of temperature field
        # High deviation = High gradient = Good for bending
        var = np.std(phi)
        variances.append(var)
        times.append(t * optimizer.dt)

    # Find peak
    peak_idx = np.argmax(variances)
    best_time = times[peak_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(times, variances, label='Temp Deviation ($\sigma$)')
    plt.axvline(best_time, color='r', linestyle='--', label=f'Optimal tf ~ {best_time:.1f}s')
    plt.title("Thermal Calibration: Finding Sweet Spot")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature Diversity (Std Dev)")
    plt.legend()
    plt.grid(True)
    plt.show()

    return best_time


# if __name__ == "__main__":
#     nelx, nely = 32, 32
#     volfrac = 0.4
#     penal = 3.0
#     rmin = 1.5
#
#     optimizer = PolypropyleneOptimization(nelx, nely, volfrac, penal, rmin)
#     optimizer.optimize(max_iter=1000)  # Reduced iterations for demo


if __name__ == "__main__":
    # 1. Setup Problem (Coarse Mesh 20x20 as requested)
    nelx, nely = 64, 64
    volfrac = 0.3
    penal = 3.0
    rmin = 1.2

    # Instantiate Polypropylene Solver
    opt = PolypropyleneOptimization(nelx, nely, volfrac, penal, rmin)

    # 2. Run Calibration
    # We guess a range. Plastic is slow. Let's try up to 200 seconds.
    best_tf = run_thermal_calibration(opt, max_time_seconds=20, steps=100)

    print(f"Calibration complete. Best time found: {best_tf}s")

    # 3. Update Optimizer with Calibrated Time
    opt.num_time_steps = 10
    opt.dt = best_tf / opt.num_time_steps
    print(f"Optimization set to: {opt.num_time_steps} steps of {opt.dt:.2f}s")

    # 4. Run Optimization
    opt.optimize(max_iter=1000)




