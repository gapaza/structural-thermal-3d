import numpy as np
from scipy.sparse import coo_matrix, linalg
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from numpy.typing import NDArray
import copy
import time
from D2.flexure_v2.utils import plot_design


# ==========================================
# 1. MMA Interface (User Provided Standards)
# ==========================================

from D2.flexure_v2.mma_subroutine import mmasub, MMAInputs

# ==========================================
# 2. Finite Element Helper (User Provided)
# ==========================================

def fe_melthm(nu: float, e: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds element stiffness, conductivity, and coupling matrices."""
    # Construct element stiffness matrix
    kel = np.array(
        [
            1 / 2 - nu / 6,
            1 / 8 + nu / 8,
            -1 / 4 - nu / 12,
            -1 / 8 + 3 * nu / 8,
            -1 / 4 + nu / 12,
            -1 / 8 - nu / 8,
            nu / 6,
            1 / 8 - 3 * nu / 8,
        ]
    )

    indices = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 7, 6, 5, 4, 3, 2],
        [2, 7, 0, 5, 6, 3, 4, 1],
        [3, 6, 5, 0, 7, 2, 1, 4],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 4, 3, 2, 1, 0, 7, 6],
        [6, 3, 4, 1, 2, 7, 0, 5],
        [7, 2, 1, 4, 3, 6, 5, 0],
    ]

    ke = (e / (1 - nu ** 2)) * kel[indices]

    # Construct element conductivity matrix
    k_eth = (k / 6) * np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]])

    # Element coupling matrix (thermal expansion)
    # Maps Nodal Temperatures (4x1) to Thermal Forces (8x1)
    c_ethm = (e * alpha / (6 * (1 - nu))) * np.array(
        [
            [-2, -2, -1, -1],
            [-2, -1, -1, -2],
            [2, 2, 1, 1],
            [-1, -2, -2, -1],
            [1, 1, 2, 2],
            [1, 2, 2, 1],
            [-1, -1, -2, -2],
            [2, 1, 1, 2],
        ]
    )

    return ke, k_eth, c_ethm


# ==========================================
# 3. Optimization Class
# ==========================================

from D2.flexure_v2.flex_plot import plot_thermal_actuation

class ThermalFlexureTopologyOptimization:
    def __init__(self, nelx, nely, volfrac, penal, rmin):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin

        # Material Properties (Arbitrary units for example, polypropylene units in comments)
        self.E0 = 1.0      # Young's Modulus (1.1e9 -> 1.1 GPa)
        self.Emin = 1e-9   # Void stiffness
        self.nu = 0.3      # Poisson ratio (0.45)
        self.k0 = 3.0      # Thermal Conductivity (0.22 J/(smK))
        self.kmin = 1e-9
        self.alpha = 0.01  # Expansion coeff (8.0e-5 /K)
        self.rho0 = 1.0    # Density (930 kg/m^3)
        self.Cp = 4.0      # Heat Capacity (1.9e3 J/(kgK))
        self.Cmin = 1e-9   # Add this

        # Time integration settings
        self.dt = 0.5
        self.num_time_steps = 100
        self.tf = self.dt * self.num_time_steps

        # Geometry
        self.ndof_mech = 2 * (nelx + 1) * (nely + 1)
        self.ndof_therm = (nelx + 1) * (nely + 1)
        self.num_elems = nelx * nely

        # Build Element Matrices (Unit Material)
        self.Ke0, self.Kt0, self.Cethm0 = fe_melthm(self.nu, self.E0, self.k0, self.alpha)
        # Heat capacity matrix (lumped or consistent - using simplified proportional to volume)
        # For a square element, integral of shape functions N^T N is related to area
        self.Ce0 = self.rho0 * self.Cp * (1.0 / (self.nelx * self.nely)) * np.eye(4) / 4.0  # Simplified lumped

        # DOF Mappings
        self._build_edof_matrices()

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


    def _build_edof_matrices(self):
        """Precompute DOF indices for efficiency."""
        self.edofMat_mech = np.zeros((self.num_elems, 8), dtype=int)
        self.edofMat_therm = np.zeros((self.num_elems, 4), dtype=int)

        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                # Thermal Nodes (Scalar)
                # n1 = ely + elx * (self.nely + 1)
                # n2 = (ely + 1) + elx * (self.nely + 1)
                # n3 = (ely + 1) + (elx + 1) * (self.nely + 1)
                # n4 = ely + (elx + 1) * (self.nely + 1)
                n1 = (ely + 1) + elx * (self.nely + 1)  # Bottom Left
                n2 = (ely + 1) + (elx + 1) * (self.nely + 1)  # Bottom Right
                n3 = ely + (elx + 1) * (self.nely + 1)  # Top Right
                n4 = ely + elx * (self.nely + 1)  # Top Left
                self.edofMat_therm[el, :] = [n1, n2, n3, n4]

                # Mechanical Nodes (Vector x,y)
                self.edofMat_mech[el, :] = [2 * n1, 2 * n1 + 1, 2 * n2, 2 * n2 + 1,
                                            2 * n3, 2 * n3 + 1, 2 * n4, 2 * n4 + 1]

        self.iK_mech = np.kron(self.edofMat_mech, np.ones((8, 1))).flatten()
        self.jK_mech = np.kron(self.edofMat_mech, np.ones((1, 8))).flatten()
        self.iK_therm = np.kron(self.edofMat_therm, np.ones((4, 1))).flatten()
        self.jK_therm = np.kron(self.edofMat_therm, np.ones((1, 4))).flatten()

    def _assemble_system(self, x):
        """Assembles Global K (mech), Kt (thermal), C (heat cap), and A (coupling)."""
        # Penalization
        # Stiffness and Conductivity: SIMP p=3
        # Density/Capacity: Linear p=1

        x_penal = x ** self.penal
        x_linear = x

        # 1. Mechanical Stiffness K
        sK = (self.Emin + x_penal * (self.E0 - self.Emin)).flatten()
        # Scale unit matrix Ke0 by sK for every element
        vals_K = np.kron(sK, self.Ke0.flatten())
        K_mech = coo_matrix((vals_K, (self.iK_mech, self.jK_mech)),
                            shape=(self.ndof_mech, self.ndof_mech)).tocsc()

        # 2. Thermal Conductivity Kt
        sKt = (self.kmin + x_penal * (self.k0 - self.kmin)).flatten()
        vals_Kt = np.kron(sKt, self.Kt0.flatten())
        K_therm = coo_matrix((vals_Kt, (self.iK_therm, self.jK_therm)),
                             shape=(self.ndof_therm, self.ndof_therm)).tocsc()

        # 3. Heat Capacity C
        # Note: Usually linear for mass constraint consistency
        # sC = x_linear.flatten()
        sC = (self.Cmin + x_linear * (self.rho0 * self.Cp - self.Cmin)).flatten()
        vals_C = np.kron(sC, self.Ce0.flatten())
        C_mat = coo_matrix((vals_C, (self.iK_therm, self.jK_therm)),
                           shape=(self.ndof_therm, self.ndof_therm)).tocsc()

        return K_mech, K_therm, C_mat

    def solve_physics(self, x):
        """
        Solves:
        A. Transient Heat Transfer -> Phi_history
        B. Mechanical Equilibrium (Thermal Load) -> U1
        C. Mechanical Equilibrium (Dummy & Resistive) -> U2, U3
        """
        K_mech, K_therm, C_mat = self._assemble_system(x)

        # --- A. Transient Heat Transfer ---
        # Eq: C * dPhi/dt + Kt * Phi = Pt
        # Backward Euler: (C + dt*Kt) * Phi_{t} = C * Phi_{t-1} + dt * Pt

        # Define Thermal BCs (e.g., Left side fixed temp 0, Right side heat flux)
        # New BC: Heat source at Bottom Right
        # Node Index for (x=nelx, y=nely)
        # node_TL = 0
        node_BR = self.nely + ((self.nely+1) * self.nelx)
        # node_BR = self.nely + self.nelx * (self.nely + 1)
        Pt = np.zeros(self.ndof_therm)
        Pt[node_BR] = 100.0

        # Initial Condition
        Phi = np.zeros(self.ndof_therm)
        Phi_history = [Phi.copy()]  # Store t=0

        # System Matrix for Time Stepping
        A_transient = C_mat + self.dt * K_therm

        # Pre-factorize for speed
        solve_transient = linalg.factorized(A_transient)

        for t in range(self.num_time_steps):
            RHS = C_mat @ Phi + self.dt * Pt
            # Apply BCs to RHS/Matrix if Dirichlet (omitted for brevity, assuming natural or handled via penalty)
            Phi = solve_transient(RHS)
            Phi_history.append(Phi.copy())

        Phi_final = Phi_history[-1]

        # --- B. Mechanical Equilibrium (Thermal Load F_t) ---
        # Ft^e = Integral( B^T E^H alpha Phi_e )
        # Using Cethm0 (8x4) which maps Element Temp -> Element Force

        Ft_global = np.zeros(self.ndof_mech)
        x_penal = x ** self.penal

        # Vectorized assembly of Ft
        # Scale coupling matrix by density/stiffness design variable
        # Note: User formulation: E^H alpha. If alpha constant, scales with E^H (x^p)

        for i in range(self.num_elems):
            idx_m = self.edofMat_mech[i]
            idx_t = self.edofMat_therm[i]
            phi_e = Phi_final[idx_t]

            # Element Thermal Force
            # F_e = (x_e^p) * Cethm0 * phi_e
            f_e = x_penal[i] * (self.Cethm0 @ phi_e)
            Ft_global[idx_m] += f_e

        # BCs for Mechanical
        # New BCs: Fixed Bottom Left and Bottom Middle
        node_BL = self.nely  # x=0, y=nely
        node_BM = self.nely + int(self.nelx / 2) * (self.nely + 1)  # x=half, y=nely
        # node_BM = 0

        fixed_nodes = np.array([node_BL, node_BM])
        fixed_dofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])

        free_dofs = np.setdiff1d(np.arange(self.ndof_mech), fixed_dofs)

        # Solve K U1 = Ft
        # Reduced integration for BCs
        K_free = K_mech[free_dofs, :][:, free_dofs]
        U1 = np.zeros(self.ndof_mech)
        U1[free_dofs] = spsolve(K_free, Ft_global[free_dofs])

        # --- C. Mechanical Equilibrium (Dummy & Resistive) ---
        # Case 2: Dummy load P2 (Output) -> Top Right, Direction Right (+x)
        # Case 3: Resistive load P3 (Stiffness) -> Top Right, Direction Left (-x)

        node_TR = self.nelx * (self.nely + 1)  # x=nelx, y=0
        dof_TR_x = 2 * node_TR  # X-direction

        P2 = np.zeros(self.ndof_mech)
        P2[dof_TR_x] = 1.0  # Unit dummy load (+x)

        P3 = np.zeros(self.ndof_mech)
        P3[dof_TR_x] = -1.0  # Resistive load (-x)

        U2 = np.zeros(self.ndof_mech)
        U2[free_dofs] = spsolve(K_free, P2[free_dofs])

        U3 = np.zeros(self.ndof_mech)
        U3[free_dofs] = spsolve(K_free, P3[free_dofs])

        return U1, U2, U3, Phi_history, K_mech, K_therm, C_mat, K_free, free_dofs, Pt

    def sensitivity_analysis(self, x, U1, U2, U3, Phi_history, K_mech, K_therm, C_mat, K_free, free_dofs, Pt):
        """
        Calculates gradients using Adjoint Method.
        Objective f = (U1.K.U2) / (U3.K.U3)
        """
        # Mutual Compliance: MC = U1.K.U2
        # Mean Compliance: C = U3.K.U3

        # 1. Calculate terms directly
        # Note: U^T K U can be calculated via Dot Product of F and U
        MC = np.dot(U1, (K_mech @ U2))
        C_val = np.dot(U3, (K_mech @ U3))
        f = MC / C_val
        # f = C_val

        # --- Sensitivity of Mean Compliance a(3,3) ---
        # da33/dx = -U3^T * dK/dx * U3
        # dK/dx_e = p * x^(p-1) * Ke0
        dc_dx = np.zeros(self.num_elems)
        xp_deriv = self.penal * (x ** (self.penal - 1))

        for i in range(self.num_elems):
            idx = self.edofMat_mech[i]
            u3_e = U3[idx]
            dKe = xp_deriv[i] * self.Ke0
            dc_dx[i] = -1.0 * (u3_e.T @ dKe @ u3_e)

        # --- Sensitivity of Mutual Compliance a(1,2) ---
        # term1: Stiffness sensitivity = -U1^T * dK/dx * U2
        # term2: Force sensitivity = U2^T * dFt/dx (Requires Adjoint)

        dmc_dx_stiffness = np.zeros(self.num_elems)
        for i in range(self.num_elems):
            idx = self.edofMat_mech[i]
            u1_e = U1[idx]
            u2_e = U2[idx]
            dKe = xp_deriv[i] * self.Ke0
            dmc_dx_stiffness[i] = -1.0 * (u1_e.T @ dKe @ u2_e)

        # Adjoint Solver for Thermal Part
        # Terminal condition: C * Lambda(tf) = A^T * U2
        # Here A is the assembly of coupling matrices Cethm.
        # So RHS = Sum_elements (Cethm_e^T * u2_e)

        # Construct RHS for Adjoint Terminal Condition
        RHS_adj_term = np.zeros(self.ndof_therm)
        x_penal = x ** self.penal

        for i in range(self.num_elems):
            idx_m = self.edofMat_mech[i]
            idx_t = self.edofMat_therm[i]
            u2_e = U2[idx_m]
            # A matrix contains x_penal scaling
            # Local contribution: (x^p * Cethm)^T * u2
            term = x_penal[i] * (self.Cethm0.T @ u2_e)
            RHS_adj_term[idx_t] += term

        # Solve for Lambda(tf) -> C * Lambda = RHS
        # Using the C_mat assembled earlier
        # Note: If C_mat is singular (voids), might need damping or reduced set
        # Assuming C_mat is well conditioned or constrained
        Lambda_tf = spsolve(C_mat, RHS_adj_term)

        # Solve Adjoint backward in time
        # -C * dLambda/dt + Kt * Lambda = 0
        # Discrete Backward: (C + dt*Kt)^T * Lambda_{t-1} = C^T * Lambda_{t}
        # Note: Since C and Kt are symmetric usually, Transpose might be same, but explicit T is safer.

        A_adj = (C_mat + self.dt * K_therm).T
        solve_adj = linalg.factorized(A_adj)

        Lambda_history = [None] * (self.num_time_steps + 1)
        Lambda_history[-1] = Lambda_tf

        for t in range(self.num_time_steps - 1, -1, -1):
            rhs = C_mat.T @ Lambda_history[t + 1]
            Lambda_history[t] = solve_adj(rhs)

        # Calculate Force Sensitivity Integral
        # U2^T * dFt/dx
        dmc_dx_force = np.zeros(self.num_elems)

        # dC/dx = Ce0 (linear x)
        # dKt/dx = p*x^(p-1) * Kt0
        # dPt/dx = 0 (assuming load independent of design)
        # dE^H/dx = p*x^(p-1)

        for i in range(self.num_elems):
            idx_t = self.edofMat_therm[i]
            idx_m = self.edofMat_mech[i]

            # Precompute derivatives of matrices
            dCe = self.Ce0  # Linear x
            dKte = xp_deriv[i] * self.Kt0
            dCethm_e = xp_deriv[i] * self.Cethm0

            # Integral term (Sum over time steps)
            integral_sum = 0.0

            # Loop over time intervals [0, tf]
            # Approximating integral via summation consistent with Backward Euler
            for t in range(1, self.num_time_steps + 1):
                Lam = Lambda_history[t][idx_t]
                Phi_curr = Phi_history[t][idx_t]
                Phi_prev = Phi_history[t - 1][idx_t]
                Phi_dot = (Phi_curr - Phi_prev) / self.dt

                # Term: Lambda^T * (-dC/dx * Phi_dot - dKt/dx * Phi)
                term = -1.0 * (Lam.T @ dCe @ Phi_dot) - (Lam.T @ dKte @ Phi_curr)
                integral_sum += term * self.dt

            # Direct coupling term at tf
            # Term: B^T * dE^H/dx * alpha * Phi(tf) * [1,1,0] . U2
            # This corresponds to: u2_e^T * (dCethm_e * Phi(tf))
            Phi_final_e = Phi_history[-1][idx_t]
            u2_e = U2[idx_m]

            direct_term = u2_e.T @ (dCethm_e @ Phi_final_e)

            dmc_dx_force[i] = integral_sum + direct_term

        dmc_dx = dmc_dx_stiffness + dmc_dx_force

        # Total Sensitivity via Quotient Rule
        # f = a12 / a33
        # f' = (a12' * a33 - a12 * a33') / (a33)^2

        df_dx = (dmc_dx * C_val - MC * dc_dx) / (C_val ** 2)
        # df_dx = dc_dx

        return f, df_dx, MC, C_val

    def filter_sensitivity(self, x, dfdx):
        """Mesh independence filter."""
        dfdx_filtered = np.zeros_like(dfdx)
        for i in range(self.num_elems):
            sum_w = 0.0
            sum_val = 0.0

            # Find neighbors (simplified grid logic)
            # In production, use cKDTree or precomputed neighbor lists
            elx, ely = divmod(i, self.nely)

            # Search window
            r_int = int(np.ceil(self.rmin))
            for dx in range(-r_int, r_int + 1):
                for dy in range(-r_int, r_int + 1):
                    nx, ny = elx + dx, ely + dy
                    if 0 <= nx < self.nelx and 0 <= ny < self.nely:
                        dist = np.sqrt(dx ** 2 + dy ** 2)
                        if dist <= self.rmin:
                            w = self.rmin - dist
                            neighbor_idx = ny + nx * self.nely
                            sum_w += w
                            sum_val += w * x[neighbor_idx] * dfdx[neighbor_idx]

            # dfdx_filtered[i] = sum_val / (x[i] * sum_w) if x[i] > 1e-3 else 0
            denominator = max(x[i] * sum_w, 1e-6)
            dfdx_filtered[i] = sum_val / denominator

        return dfdx_filtered

    def optimize(self, max_iter=50):
        print(f"Starting Optimization: Grid {self.nelx}x{self.nely}")

        for k in range(max_iter):
            # 1. Physics
            U1, U2, U3, Phi_hist, Km, Kt, Cm, Kfree, fdofs, Pt_vec = self.solve_physics(self.x)

            # 2. Sensitivity
            f_val, dfdx, MC, C33 = self.sensitivity_analysis(self.x, U1, U2, U3, Phi_hist, Km, Kt, Cm, Kfree, fdofs, Pt_vec)

            # 3. Filtering
            dfdx_filt = self.filter_sensitivity(self.x, dfdx)

            # 4. Constraints
            # Volume constraint: sum(x) <= volfrac * num_elems
            # g = (sum(x) / (volfrac * N)) - 1 <= 0
            current_vol = np.sum(self.x)
            g_val = (current_vol / (self.volfrac * self.num_elems)) - 1.0
            dgdx = np.ones(self.num_elems) / (self.volfrac * self.num_elems)

            # 5. MMA Step
            # Map to MMA Inputs
            inputs = MMAInputs(
                m=1,
                n=self.num_elems,
                iterr=k,
                xval=self.x,
                xmin=0.0,
                xmax=1.0,
                xold1=self.xold1,
                xold2=self.xold2,
                df0dx=-dfdx_filt,  # Negative because MMA minimizes, we maximize f
                fval=np.array([g_val]),
                dfdx=dgdx.reshape(1, -1),
                low=self.low,
                upp=self.upp,
                a0=1.0,
                a=self.a_mma,
                c=self.c_mma,
                d=self.d_mma,
                f0val=-f_val
            )

            # Call MMA
            x_new, low_new, upp_new = mmasub(inputs)

            # Update history
            self.xold2 = self.xold1.copy()
            self.xold1 = self.x.copy()
            self.low = low_new
            self.upp = upp_new

            change = np.max(np.abs(x_new - self.x))
            self.x = x_new


            # x_formatted = np.reshape(x_new, (self.nelx, self.nely))
            # plot_design(x_formatted, open_plot=True)
            # time.sleep(1)

            # ============================================================
            # VISUALIZATION INSERTION
            # ============================================================
            # Pass the final temperature field (Phi_hist[-1])
            if k % 5 == 0:
                plot_thermal_actuation(self, U1, Phi_hist[-1], title=f"Iteration {k}")
            # ============================================================
            # time.sleep(5)



            print(f"Iter: {k} | Obj: {f_val:.4e} | Vol: {current_vol / (self.num_elems):.3f} | Change: {change:.4f}")

            if change < 0.001 and k > 5:
                print("Convergence reached.")
                break


# ==========================================
# 4. Thermal Calibration
# ==========================================

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
    x_solid = np.ones(optimizer.num_elems) * 1.0
    # x_noise = np.random.normal(loc=0, scale=0.3, size=x_solid.shape)
    # x_solid += x_noise

    # Run physics (We only care about Phi_history)
    # We catch the output but only use Phi_history
    _, _, _, Phi_hist, _, _, _, _, _, _ = optimizer.solve_physics(x_solid)

    variances = []
    times = []
    print(len(Phi_hist))
    for t in range(len(Phi_hist)):
        phi = Phi_hist[t]
        # Calculate standard deviation of temperature field
        # High deviation = High gradient = Good for bending
        var = np.std(phi)
        variances.append(var)
        times.append(t * optimizer.dt)

        # if t % 5 == 0:
        #     print(phi.shape)
        #     phi_re = np.reshape(phi, (optimizer.nelx+1, optimizer.nely+1))
        #     plt.figure(figsize=(8, 5))
        #     im4 = plt.imshow(phi_re)
        #     # add a color bar
        #     plt.colorbar(im4)
        #     plt.show()
        #     time.sleep(0.5)


    # exit(0)

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

    # exit(0)

    return best_time


# ==========================================
# 5. Main Execution Block
# ==========================================

if __name__ == "__main__":
    # Example Setup
    nelx, nely = 128, 128
    volfrac = 0.3
    penal = 3.0
    rmin = 3.0

    opt = ThermalFlexureTopologyOptimization(nelx, nely, volfrac, penal, rmin)
    # opt.optimize(max_iter=1000)  # Reduced iterations for demo

    # 2. Run Calibration
    # We guess a range. Plastic is slow. Let's try up to 200 seconds.
    best_tf = run_thermal_calibration(opt, max_time_seconds=200, steps=100)

    print(f"Calibration complete. Best time found: {best_tf}s")

    # 3. Update Optimizer with Calibrated Time
    opt.num_time_steps = 10
    opt.dt = best_tf / opt.num_time_steps
    print(f"Optimization set to: {opt.num_time_steps} steps of {opt.dt:.2f}s")

    # 4. Run Optimization
    opt.optimize(max_iter=1000)











