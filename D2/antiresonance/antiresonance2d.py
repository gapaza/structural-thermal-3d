import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, linalg
from scipy.sparse.linalg import spsolve, eigsh
from scipy.signal import find_peaks
from D3.utils.linear_solver import solve_spd_with_amg

from D2.antiresonance.plotting import plot_eigenmodes, plot_design, plot_harmonic_sweep
from D2.antiresonance.xu_filter import XuProjectionFilter

# ==========================================
# 1. MMA Interface (User Provided Standards)
# ==========================================

from D2.flexure_v2.mma_subroutine import mmasub, MMAInputs

# ==========================================
# 2. Finite Element Helper (User Provided)
# ==========================================

# Local element matrics
from element.physics.elasticity_2d import get_mechanical_stiffness
from element.physics.mass_2d import get_mechanical_mass

# DOF Helpers
from element.edof_2d import build_edof_matrices_generic


# ==========================================
# 3. Optimization Class
# ==========================================


class AntiresonanceTopologyOptimization2D:

    def __init__(self, nelx, nely, volfrac, penal, rmin, iter_solve=False, vmin=0.3, vmax=0.7):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.vmin = vmin
        self.vmax = vmax

        if iter_solve is False:
            self.sparse_solver = spsolve
        else:
            self.sparse_solver = solve_spd_with_amg

        # Material properties
        self.E0 = 3.2516e8     # Young's Modulus (1.1e9 -> 1.1 GPa)
        self.Emin = 1e-9  # Void stiffness
        self.nu = 0.33  # Poisson ratio (0.45)
        self.rho0 = 1222.2     # Density (930 kg/m^3)
        self.rho_min = 1e-9  # Void density

        # # Material properties
        # self.E0 = 1.0  # Young's Modulus (1.1e9 -> 1.1 GPa)
        # self.Emin = 1e-9  # Void stiffness
        # self.nu = 0.3  # Poisson ratio (0.45)
        # self.rho0 = 1.0  # Density (930 kg/m^3)
        # self.rho_min = 1e-9  # Void density

        # Geometry
        self.ndof_mech = 2 * (nelx + 1) * (nely + 1)
        self.num_elems = nelx * nely

        # Build Element Matrices (Unit Material)
        self.lx = 1.0
        self.ly = 1.0
        self.thick = 1.0
        self.Ke0 = get_mechanical_stiffness(self.nu, self.E0, lx=self.lx, ly=self.ly, thick=self.thick)
        self.Km0 = get_mechanical_mass(self.rho0, lx=self.lx, ly=self.ly, thick=self.thick)

        # DOF Mappings
        self.iK_mech, self.jK_mech, self.edof_mech = build_edof_matrices_generic(self.nelx, self.nely, dof_per_node=2)

        # MMA Initialization
        self.iter = 0

        # self.x = self.volfrac * np.ones(self.num_elems)
        self.x = 0.5 * np.ones(self.num_elems)
        # self.x[self.num_elems // 4:] = 0.8
        # add random noise to x
        # self.x += 0.1 * np.random.rand(self.num_elems)

        self.xold1 = self.x.copy()
        self.xold2 = self.x.copy()
        self.low = np.zeros_like(self.x)
        self.upp = np.ones_like(self.x)
        self.a0 = 1.0
        self.a_mma = np.zeros(1)  # m=1 constraint
        self.c_mma = 1000.0 * np.ones(1)
        self.d_mma = np.zeros(1)

        # Filter Initialization
        self.xu_filter = XuProjectionFilter(nelx, nely, rmin)



    def _build_boundary_nodes(self):

        # Individual Boundary Nodes
        node_TL = 0
        node_ML = self.nely // 2
        node_BL = self.nely
        node_TR = self.nelx * (self.nely + 1)
        node_MR = ((self.nely + 1) * self.nelx) + node_ML
        node_BR = ((self.nely + 1) * self.nelx) + node_BL

        # Node sets
        nodes_R = np.arange(node_TR, node_TR+16)
        nodes_L = np.arange(node_TL, node_BL)

        # Constrained Nodes
        # fixed_nodes = np.array([node_ML, node_BL])
        fixed_nodes = nodes_L

        # Antiresonance Nodes (also the force application nodes)
        antires_nodes = np.array([node_BR, node_MR, node_TR])
        # antires_nodes = np.array([node_TR, node_TR+1, node_TR+2])
        # antires_nodes = nodes_R

        # Force Nodes
        force_nodes = antires_nodes
        force_dofs_x = 2 * force_nodes
        force_dofs_y = 2 * force_nodes + 1
        # force_dofs = np.concatenate([force_dofs_x, force_dofs_y])
        force_dofs = force_dofs_y
        force_val = 10.0

        # Oscillatory force
        target_freq_hz = 150.0  # f_T: Defined by user/problem
        target_omega = 2 * np.pi * target_freq_hz

        # Eigenvalue Analysis
        num_modes = 1000

        # Harmonic Analysis
        freq_hz_start = 1.0
        freq_hz_end = 1000.0
        freq_steps = 100
        freqs_scan = np.linspace(freq_hz_start, freq_hz_end, freq_steps)


        return {
            'fixed_nodes': fixed_nodes,
            'antires_nodes': antires_nodes,
            'force_nodes': force_nodes,

            'force_dofs': force_dofs,
            'force_val': force_val,

            'target_freq_hz': target_freq_hz,
            'target_omega': target_omega,


            'num_modes': num_modes,
            'freqs_scan': freqs_scan
        }

    def _assemble_system(self, x):
        """Assembles Global Ke (stiffness), Me (mass), and Ce (damping) matrices."""
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

        # 2. Mass Matrix M
        sM = (self.rho_min + x_linear * (self.rho0 - self.rho_min)).flatten()
        vals_M = np.kron(sM, self.Km0.flatten())
        M_mech = coo_matrix((vals_M, (self.iK_mech, self.jK_mech)),
                            shape=(self.ndof_mech, self.ndof_mech)).tocsc()

        # 3. Damping Matrix C (Rayleigh Damping: C = alpha*M + beta*K)
        # alpha = 0.01
        # beta = 0.01
        alpha = 1e-5  # Low mass damping
        beta = 2e-6  # Low stiffness damping (Calculated for ~1500Hz)
        C_mech = alpha * M_mech + beta * K_mech

        return K_mech, M_mech, C_mech

    def solve_physics(self, x):
        """
        Uses:
        The Antiresonance Principle - When a structure is forced at an antiresonance frequency, the displacement at the forcing point drops to zero.
            Physically, this looks identical to the structure vibrating freely as if that forcing point were mechanically fixed (constrained).

        Solves:
        A. Eigenfrequency Problem (Constrained at Antiresonance node)
        B. Harmonic Analysis Problem (Free at Antiresonance node)
        C. MAC Selection
        """
        boundary_conds = self._build_boundary_nodes()

        K_mech, M_mech, C_mech = self._assemble_system(x)


        # Add oscillatory force definition here
        target_freq_hz = boundary_conds['target_freq_hz']
        target_omega = boundary_conds['target_omega']


        # --- A. Eigenfrequency Problem (Constrained Target) ---
        # If we do the eigenfrequency analysis without the constraint at the target node,
        #  we get a normal resonance analysis.
        # Instead, we want to FIX the target node, getting the ways in which the structure can
        #  vibrate while holding the target point still (i.e., antiresonance modes).
        # This gives us the eigenmodes of the structure that have zero displacement at the target point.
        # Problem: we don't know which mode corresponds to the antiresonance at the target frequency.

        # 1. Boundary Conditions

        # Nodes
        fixed_nodes = boundary_conds['fixed_nodes']
        antires_nodes = boundary_conds['antires_nodes']

        # DOFs
        fixed_dofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
        antires_dofs = np.concatenate([2 * antires_nodes, 2 * antires_nodes + 1])

        constrained_dofs = np.union1d(fixed_dofs, antires_dofs)
        all_dofs = np.arange(self.ndof_mech)
        free_dofs_eig = np.setdiff1d(all_dofs, constrained_dofs)

        # Extract sub-matrices
        K_eig = K_mech[free_dofs_eig, :][:, free_dofs_eig]
        M_eig = M_mech[free_dofs_eig, :][:, free_dofs_eig]

        # 2. Solve Generalized Eigenvalue Problem
        num_modes = boundary_conds['num_modes']  # Calculate a few to ensure we capture the relevant one
        eigenvalues, eigenvectors_sub = eigsh(
            A=K_eig,
            M=M_eig,
            k=num_modes,
            sigma=(target_omega ** 2),  # Shift-invert near target freq
            which='LM'
        )

        # 3. Post-Process Results

        # Sort output (eigsh doesn't guarantee order)
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors_sub = eigenvectors_sub[:, idx]

        # Convert Eigenvalues (lambda = omega^2) to Natural Frequencies (Hz)
        # Note: We take absolute value to handle tiny negative numerical noise from shift
        natural_frequencies_rad = np.sqrt(np.abs(eigenvalues))
        natural_frequencies_hz = natural_frequencies_rad / (2 * np.pi)

        # 4. Reconstruct Full Mode Shapes
        # Insert zeros back into the fixed DOF locations
        modes = np.zeros((self.ndof_mech, num_modes))
        modes[free_dofs_eig, :] = eigenvectors_sub

        # 5. Visualize Modes for Debugging
        # plot_eigenmodes(self, x, natural_frequencies_hz, modes)

        # --- B. Harmonic Sweep (Unconstrained) ---
        # We need to find the CURRENT antiresonance, which might not be at target_freq_hz.

        # 1. Define Sweep Range (e.g., +/- 50% of target)
        freqs_scan = boundary_conds['freqs_scan']

        # 2. Prepare Unconstrained Matrices (Fixed Base, Free Target)
        free_dofs_harm = np.setdiff1d(all_dofs, fixed_dofs)

        K_red = K_mech[free_dofs_harm, :][:, free_dofs_harm]
        M_red = M_mech[free_dofs_harm, :][:, free_dofs_harm]
        C_red = C_mech[free_dofs_harm, :][:, free_dofs_harm]

        # 3. Force Vector
        F = np.zeros(self.ndof_mech, dtype=complex)
        force_dofs = boundary_conds['force_dofs']
        F[force_dofs] = boundary_conds['force_val']
        F_red = F[free_dofs_harm]

        # 4. Run Sweep
        frf_amplitudes = []
        displacement_snapshots = []  # Store u for every freq (memory intensive but safest)

        for f_hz in freqs_scan:
            w = 2 * np.pi * f_hz
            # Dynamic Stiffness
            Kd = K_red + 1j * w * C_red - (w ** 2) * M_red

            # Solve
            u_red = self.sparse_solver(Kd, F_red)

            # Reconstruct full vector
            u_full = np.zeros(self.ndof_mech, dtype=complex)
            u_full[free_dofs_harm] = u_red

            displacement_snapshots.append(u_full)

            # Record Amplitude at Load Node (for Peak finding)
            u_load_mag = np.mean(np.abs(u_full[boundary_conds['force_dofs']]))
            frf_amplitudes.append(u_load_mag)

        # Pack sweep data into a dictionary to pass to the selection routine
        harmonic_data = {
            'freqs': freqs_scan,
            'frf_amps': np.array(frf_amplitudes),
            'displacements': displacement_snapshots
        }

        return harmonic_data, eigenvalues, natural_frequencies_hz, modes, M_mech




    def _filter_localized_modes(self, mode_vectors, mode_freqs, rho, threshold=1e3):
        """
        Filters out modes that are localized in void regions (low density).
        Returns indices of 'valid' modes.
        """
        print('finding valid modes...')
        boundary_conds = self._build_boundary_nodes()

        valid_indices = []

        # Define thresholds per paper
        rho_hard_thresh = 0.7
        rho_soft_thresh = 0.1

        # Identify Elements
        hard_elem_indices = np.where(rho > rho_hard_thresh)[0]
        soft_elem_indices = np.where(rho < rho_soft_thresh)[0]

        # Handle edge case: If structure is fully solid, all modes are valid
        if len(soft_elem_indices) == 0:
            return np.arange(mode_vectors.shape[1])

        # Map Elements to Nodes to get Nodal Displacements
        # We assume if any element connected to a node is Hard, the node is 'Hard'
        # (This is a heuristic to categorize nodes)
        # edof_mech shape: (num_elems, 8) -> flattened node indices

        # 1. Build Node Masks
        # Flatten edof indices for hard/soft elements
        hard_dofs = np.unique(self.edof_mech[hard_elem_indices, :].flatten())
        soft_dofs = np.unique(self.edof_mech[soft_elem_indices, :].flatten())

        # Build DOFs to ignore (constrained DOFs)
        fixed_nodes = boundary_conds['fixed_nodes']
        antires_nodes = boundary_conds['antires_nodes']
        fixed_dofs = np.concatenate([2 * fixed_nodes, 2 * fixed_nodes + 1])
        antires_dofs = np.concatenate([2 * antires_nodes, 2 * antires_nodes + 1])
        constrained_dofs = np.union1d(fixed_dofs, antires_dofs)

        # Remove constrained dofs from hard/soft lists
        hard_dofs = np.setdiff1d(hard_dofs, constrained_dofs)
        soft_dofs = np.setdiff1d(soft_dofs, constrained_dofs)

        # For each eigenvector, extract displacements and compute metrics
        for i in range(mode_vectors.shape[1]):
            phi = mode_vectors[:, i]

            # Extract displacements
            u_hard = np.abs(phi[hard_dofs])
            u_soft = np.abs(phi[soft_dofs])

            # Avoid division by zero
            if len(u_hard) == 0 or np.max(u_hard) == 0:
                # Suspicious mode (no movement in solid parts), likely localized
                continue

            # Compute Ratios (Step 3 metrics)
            # 1. PC90
            pc90_soft = np.percentile(u_soft, 90)
            pc90_hard = np.percentile(u_hard, 90)

            # Avoid divide by zero
            # r_pc90 = pc90_soft / pc90_hard if pc90_hard > 1e-12 else 1e6
            r_pc90 = pc90_soft / pc90_hard

            # 2. Max
            r_max = np.max(u_soft) / np.max(u_hard)

            # 3. Mean
            r_mean = np.mean(u_soft) / np.mean(u_hard)

            # Compound Ratio
            ratio_compound = r_pc90 + r_max + r_mean

            if ratio_compound < threshold:
                valid_indices.append(i)

        return np.array(valid_indices)

    def select_antiresonance_mode(self, x, harmonic_data, natural_frequencies_hz, modes):
        """
        Selects the eigenmode that best matches the physical antiresonance behavior.

        Args:
            harmonic_data: frequencies, amplitudes, displacements from harmonic sweep
            natural_frequencies_hz: Array of eigenfrequencies
            modes: Matrix of eigenvectors (ndof x num_modes)

        Returns:
            selected_freq: The eigenvalue (f_A) to be minimized
            selected_mode_derivs: Data needed for sensitivity analysis (the eigenvector)
        """
        boundary_conds = self._build_boundary_nodes()
        target_freq = boundary_conds['target_freq_hz']

        # --- Step 1 & 2: Rigid Body Filter ---
        hz_threshold = 1.0
        valid_indices = np.where(natural_frequencies_hz > hz_threshold)[0]
        if len(valid_indices) == 0:
            print("Warning: All modes are rigid body or < 1Hz.")
            return None, None  # Handle gracefully
        filtered_freqs = natural_frequencies_hz[valid_indices]
        filtered_modes = modes[:, valid_indices]
        # print("Filtered Frequencies:", filtered_freqs.flatten())

        # --- Step 3: Localized Mode Filter ---
        physically_valid_idx = self._filter_localized_modes(filtered_modes, filtered_freqs, x, threshold=1e4)

        if len(physically_valid_idx) == 0:
            print("Warning: All modes identified as localized artifacts.")
            # Fallback: revert to rigid-body filtered set
            candidate_freqs = filtered_freqs
            candidate_modes = filtered_modes
        else:
            candidate_freqs = filtered_freqs[physically_valid_idx]
            candidate_modes = filtered_modes[:, physically_valid_idx]

        # --- Step 4 & 5: Harmonic Sweep & Peak Identification ---

        # 1. Unpack Sweep Data
        freqs_scan = harmonic_data['freqs']
        frf = harmonic_data['frf_amps']
        displacements = harmonic_data['displacements']

        # 2. Find Peaks in INVERTED FRF (Minima in displacement = Maxima in 1/disp)
        inv_frf = 1.0 / (frf + 1e-16)
        peaks, properties = find_peaks(inv_frf, prominence=0.1, width=0.1)

        # --- DEBUG PLOTTING ---
        # if self.iter % 10 == 0:
        plot_harmonic_sweep(self, freqs_scan, frf, inv_frf)
        # ----------------------

        if len(peaks) == 0:
            print("Warning: No antiresonance peaks found in FRF.")
            best_peak_idx = np.argmax(inv_frf)
        else:
            # Score Peaks
            best_score = -np.inf
            best_peak_idx = -1
            for i, idx in enumerate(peaks):
                # print the keys of properties
                amp = properties['peak_heights'][i] if 'peak_heights' in properties else inv_frf[idx]
                prom = properties['prominences'][i]
                width = properties['widths'][
                    i]  # This is in samples, ideally convert to Hz but this works for relative comparison

                peak_freq = freqs_scan[idx]
                proximity = abs(target_freq - peak_freq)

                # Evaluation Factor
                # Note: Amplitude and Proximity likely have vastly different scales.
                # If selection is erratic, normalize these terms.
                factor = amp + prom + width - proximity

                if factor > best_score:
                    best_score = factor
                    best_peak_idx = idx

        identified_freq_Af, x_H = freqs_scan[best_peak_idx], displacements[best_peak_idx]
        print(f"  Physical Antiresonance detected at: {identified_freq_Af:.2f} Hz")

        # --- Step 6: MAC Selection ---
        # Compare physical response x_H with candidate eigenmodes Phi_A

        best_mac = -1.0
        best_mode_idx = -1

        # x_H is complex, modes are real.
        # We compare magnitude/shape correlation.

        for i in range(candidate_modes.shape[1]):
            phi_A = candidate_modes[:, i]

            # Standard MAC Formula:
            # MAC = |phi.T * x|^2 / ((phi.T * phi) * (x.T * x))

            dot_prod = np.vdot(phi_A, x_H)  # Hermitian inner product
            numerator = np.abs(dot_prod) ** 2

            denom_phi = np.vdot(phi_A, phi_A).real
            denom_x = np.vdot(x_H, x_H).real

            denominator = denom_phi * denom_x

            # print('Denominator:', denominator)
            # print('Numerator:', numerator)
            # if denominator < 1e-12:
            #     mac = 0.0
            # else:
            #     mac = numerator / denominator
            mac = numerator / denominator

            if mac > best_mac:
                best_mac = mac
                best_mode_idx = i

        selected_freq = candidate_freqs[best_mode_idx]
        selected_mode = candidate_modes[:, best_mode_idx]

        print(f"  Selected Mode: {selected_freq:.2f} Hz (MAC: {best_mac:.4f})")

        return selected_freq, selected_mode

    def sensitivity_analysis(self, x, selected_freq, selected_mode, M_mech):
        """
        Calculates the gradient of the objective function with respect to density x.
        Using the chain rule derived from Eq (5) in the paper.

        Args:
            x: Current density array (nelem,)
            selected_freq: The current antiresonance frequency f_A (scalar, Hz)
            selected_mode: The corresponding eigenvector Phi_A (ndof,)

        Returns:
            grad: The sensitivity vector dJ/dx (nelem,)
        """
        boundary_conds = self._build_boundary_nodes()
        target_freq = boundary_conds['target_freq_hz']

        # Compute Objective Function First
        obj = ((selected_freq - target_freq) / target_freq) ** 2

        # 1. Constants and Scale Factors
        # ---------------------------------------------------------
        # Lambda = (2*pi*f)^2
        lam_A = (2 * np.pi * selected_freq) ** 2

        # Scale Factor derived from Chain Rule (matches Paper Eq. 5)
        # dJ/dlam = (f_A - f_T) / (4 * pi^2 * f_T^2 * f_A)
        # Note: Avoid division by zero if f_A is weirdly 0
        scale_factor = (selected_freq - target_freq) / (4 * (np.pi ** 2) * (target_freq ** 2) * selected_freq)

        # 2. Vectorized Element Energy Calculations
        # ---------------------------------------------------------
        # We need u_e^T * K_e * u_e for every element.
        # Doing this in a loop is too slow. We vectorize using the edof matrix.

        # Get displacements for every element: (nelem, 8)
        u_ele = selected_mode[self.edof_mech]

        # Compute Kinetic Energy term: u_e^T * M_e0 * u_e
        # M_e0 is (8,8). Result is (nelem,)
        # (u_ele @ M0) performs matrix mult on last dimension
        # sum( ... * u_ele, axis=1) completes the dot product u^T * (M u)
        m_e0_prods = (u_ele @ self.Km0) * u_ele
        energy_kinetic = m_e0_prods.sum(axis=1)

        # Compute Strain Energy term: u_e^T * K_e0 * u_e
        k_e0_prods = (u_ele @ self.Ke0) * u_ele
        energy_strain = k_e0_prods.sum(axis=1)

        # 3. Material Derivatives (SIMP / Linear)
        # ---------------------------------------------------------
        # dK/dx = p * x^(p-1) * (E0 - Emin) * K_e0
        # dM/dx = 1 * (rho0 - rhomin) * M_e0

        # Stiffness sensitivity factor
        dE_dx = self.penal * (x ** (self.penal - 1)) * (self.E0 - self.Emin)

        # Mass sensitivity factor
        dRho_dx = (self.rho0 - self.rho_min) * np.ones_like(x)

        # 4. Eigenvalue Sensitivity (dLam / dx)
        # ---------------------------------------------------------
        # Formula: (phi^T * (dK/dx - lam * dM/dx) * phi) / (phi^T * M * phi)

        # Numerator element-wise
        # (dE_dx * uKu) - lam * (dRho_dx * uMu)
        sens_num = (dE_dx * energy_strain) - (lam_A * dRho_dx * energy_kinetic)

        # Denominator (Mass Norm) - Global Scalar
        # We need the global M matrix for this, or we can sum element contributions
        # Global way is safer to ensure BCs are handled if M_mech was modified

        # Note: selected_mode might be full size.
        # The M_mech must match the size.
        # phi^T * M * phi
        denom = selected_mode.T @ (M_mech @ selected_mode)

        dlam_dx = sens_num / denom

        # 5. Final Gradient
        # ---------------------------------------------------------
        grad = scale_factor * dlam_dx

        return obj, grad

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

    def filter_design(self, x):
        """Applies the Xu Projection Filter to the design variables."""
        x_filtered, eta = self.xu_filter.apply_filter(x, beta=5.0)
        return x_filtered


    def optimize(self, max_iter=50):
        print(f"Starting Optimization: Grid {self.nelx}x{self.nely}")

        for k in range(max_iter):

            self.x = self.filter_design(self.x)

            # 1. Physics
            u_harmonic, eigenvalues, natural_frequencies_hz, modes, M_mech = self.solve_physics(self.x)

            # 2. Antiresonance Eigenmode Selection
            selected_freq, selected_mode = self.select_antiresonance_mode(self.x, u_harmonic, natural_frequencies_hz, modes)

            # 3. Sensitivity Analysis
            obj, grad = self.sensitivity_analysis(self.x, selected_freq, selected_mode, M_mech)
            # grad = self.filter_sensitivity(self.x, grad)

            # 4. Constraints
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
                xmin=1e-6,
                xmax=1.0,
                xold1=self.xold1,
                xold2=self.xold2,
                df0dx=grad,
                fval=np.array([g_val]),
                dfdx=dgdx.reshape(1, -1),
                low=self.low,
                upp=self.upp,
                a0=1.0,
                a=self.a_mma,
                c=self.c_mma,
                d=self.d_mma,
                f0val=obj
            )



            # Call MMA
            x_new, low_new, upp_new = mmasub(inputs)
            x_new = np.reshape(x_new, (-1,))

            # Update history
            self.xold2 = self.xold1.copy()
            self.xold1 = self.x.copy()
            self.low = low_new
            self.upp = upp_new

            change = np.max(np.abs(x_new - self.x))
            self.x = x_new
            self.iter += 1

            # if k % 10 == 0:
            plot_design(self, self.x)

            print(f"Iter: {k} | Obj: {obj:.4e} | Vol: {current_vol / (self.num_elems):.3f} | Change: {change:.4f}")
            if change < 0.001 and k > 5:
                print("Convergence reached.")
                break


# ==========================================
# 4. Main Execution Block
# ==========================================


if __name__ == "__main__":
    # Example Setup
    nelx, nely = 64, 32
    volfrac = 0.9
    penal = 3.0
    rmin = 1.5

    opt = AntiresonanceTopologyOptimization2D(nelx, nely, volfrac, penal, rmin)
    opt.optimize(max_iter=1000)










