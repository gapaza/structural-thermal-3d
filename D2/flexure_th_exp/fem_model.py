"""This module contains the Python implementation of the thermoelastic 2D problem."""

from math import ceil
from math import hypot
import time
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from D2.flexure_th_exp.fem_matrix_builder import fe_melthm
from D2.flexure_th_exp.fem_forward import fe_mthm_bc
from D2.flexure_th_exp.mma_subroutine import MMAInputs
from D2.flexure_th_exp.mma_subroutine import mmasub

from D2.flexure_th_exp.utils import get_res_bounds
from D2.flexure_th_exp.utils import plot_multi_physics

SECOND_ITERATION_THRESHOLD = 2
FIRST_ITERATION_THRESHOLD = 1
MIN_ITERATIONS = 10
MAX_ITERATIONS = 120
UPDATE_THRESHOLD = 0.01


class FeaModel:
    """Finite Element Analysis (FEA) model for coupled 2D thermoelastic topology optimization."""

    def __init__(self, *, plot: bool = False, eval_only: bool | None = False) -> None:
        """Instantiates a new model for the thermoelastic 2D problem.

        Args:
            plot: (bool, optional): If True, the updated design will be plotted at each iteration.
            eval_only: (bool, optional): If True, the model will only evaluate the design and return the objective values.
        """
        self.plot = plot
        self.eval_only = eval_only

    def get_initial_design(self, volume_fraction: float, nelx: int, nely: int) -> np.ndarray:
        """Generates the initial design variable field for the optimization process.

        Args:
            volume_fraction (float): The initial volume fraction for the material distribution.
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.

        Returns:
            np.ndarray: A 2D NumPy array of shape (nely, nelx) initialized with the given volume fraction.
        """
        return volume_fraction * np.ones((nely, nelx))

    def get_matricies(self, nu: float, e: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes and returns the element matrices required for the structural-thermal analysis.

        Args:
            nu (float): Poisson's ratio.
            e (float): Young's modulus (modulus of elasticity).
            k (float): Thermal conductivity.
            alpha (float): Coefficient of thermal expansion.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - The stiffness matrix for mechanical analysis.
                - The thermal stiffness matrix.
                - The coupling matrix for thermal expansion effects.
        """
        return fe_melthm(nu, e, k, alpha)

    def get_filter(self, nelx: int, nely: int, rmin: float) -> tuple[coo_matrix, np.ndarray]:
        """Constructs a sensitivity filtering matrix to smoothen the design variables.

        The filter helps mitigate checkerboarding issues in topology optimization by averaging
        sensitivities over neighboring elements.

        Args:
            nelx (int): Number of elements in the x-direction.
            nely (int): Number of elements in the y-direction.
            rmin (float): Minimum filter radius.

        Returns:
            Tuple[csr_matrix, np.ndarray]: A tuple containing:
                - `h` (csr_matrix): A sparse matrix that represents the filtering operation.
                - `hs` (np.ndarray): A normalization factor for the filtering.
        """
        i_h = []
        j_h = []
        s_h = []

        for i1 in range(nelx):
            i2_min = max(i1 - (ceil(rmin) - 1), 0)
            for j1 in range(nely):
                e1 = i1 * nely + j1
                i2_max = min(i1 + (ceil(rmin) - 1), nelx - 1)
                for i2 in range(i2_min, i2_max + 1):
                    j2_min = max(j1 - (ceil(rmin) - 1), 0)
                    j2_max = min(j1 + (ceil(rmin) - 1), nely - 1)
                    for j2 in range(j2_min, j2_max + 1):
                        e2 = i2 * nely + j2
                        i_h.append(e1)
                        j_h.append(e2)
                        s_h.append(max(0, rmin - hypot(i1 - i2, j1 - j2)))

        h = coo_matrix((s_h, (i_h, j_h)), shape=(nelx * nely, nelx * nely)).tocsr()
        hs = np.array(h.sum(axis=1)).flatten()

        return h, hs

    def run(self, bcs: dict[str, Any], x_init: np.ndarray | None = None) -> dict[str, Any]:  # noqa: PLR0915
        """Run the optimization algorithm for the coupled structural-thermal problem.

        This method performs an iterative optimization procedure that adjusts the design
        variables for a coupled structural-thermal problem until convergence criteria are met.
        The algorithm utilizes finite element analysis (FEA), sensitivity analysis, and the Method
        of Moving Asymptotes (MMA) to update the design.

        Args:
            bcs (dict[str, any]): A dictionary containing boundary conditions and problem parameters.
                Expected keys include:
                    - 'volfrac' (float): Target volume fraction.
                    - 'fixed_elements' (np.ndarray): NxN binary array encoding the location of fixed elements.
                    - 'force_elements_x' (np.ndarray): NxN binary array encoding the location of loaded elements in the x direction.
                    - 'force_elements_y' (np.ndarray): NxN binary array encoding the location of loaded elements in the y direction.
                    - 'heatsink_elements' (np.ndarray): NxN binary array encoding the location of heatsink elements.
                    - 'weight' (float, optional): Weighting factor between structural and thermal objectives.
            x_init (Optional[np.ndarray]): Initial design variable array. If None, the design is generated
                using the get_initial_design method.

        Returns:
            Dict[str, Any]: A dictionary containing the optimization results. The dictionary includes:
                - 'design' (np.ndarray): Final design layout.
                - 'bcs' (Dict[str, Any]): The input boundary conditions.
                - 'sc' (float): Structural cost component.
                - 'tc' (float): Thermal cost component.
                - 'vf' (float): Volume fraction error.
            If self.eval_only is True, returns a dictionary with keys 'sc', 'tc', and 'vf' only.
        """
        # WEIGHTING
        w1 = bcs.get("weight", 0.5)
        w2 = 1.0 - w1

        fixed_elements = bcs["fixed_elements"]
        fe_h, fe_w = fixed_elements.shape
        nelx = fe_h - 1
        nely = fe_w - 1

        volfrac = bcs["volfrac"]
        n = nely * nelx  # Total number of elements

        # OptiSteps records
        opti_steps = []

        # 1. Initial Design
        x = self.get_initial_design(volfrac, nelx, nely) if x_init is None else x_init

        # 2. Parameters
        penal = 3  # Penalty term
        rmin = 1.1  # Filter's radius
        e = 1.0  # Modulus of elasticity
        nu = 0.3  # Poisson's ratio
        k = 1.0  # Conductivity
        alpha = 5e-4  # Coefficient of thermal expansion (CTE)
        tref = 9.267e-4  # Reference Temperature
        change = 1.0  # Density change criterion
        m = 1  # Number of constraints (volume constraints)
        iterr = 0  # Number of iterations
        xmin = 1e-3  # Densities' Lower bound
        xmax = 1.0  # Densities' Upper bound
        low = xmin
        upp = xmax
        xold1 = x.reshape(n, 1)
        xold2 = x.reshape(n, 1)
        a0 = 1
        a = np.zeros((m, 1))
        c = 10000 * np.ones((m, 1))
        d = np.zeros((m, 1))

        upp_vec = np.ones((n,)) * upp
        low_vec = np.ones((n,)) * low

        # 3. Matrices
        ke, k_eth, c_ethm = self.get_matricies(nu, e, k, alpha)

        # 4. Filter
        h, hs = self.get_filter(nelx, nely, rmin)

        # 5. Optimization Loop
        change_evol = []
        obj = []

        f0valm = 0.0
        f0valt = 0.0

        while change > UPDATE_THRESHOLD or iterr < MIN_ITERATIONS:
            iterr += 1
            s_time = time.time()
            curr_time = time.time()

            # FE-ANALYSIS
            results = fe_mthm_bc(nely, nelx, penal, x, ke, k_eth, c_ethm, tref, bcs)
            km = results.km
            kth = results.kth
            um = results.um
            uth = results.uth
            fm = results.fm
            fth = results.fth
            d_cthm = results.d_cthm
            fixeddofsm = results.fixeddofsm
            freedofsm = results.freedofsm
            fixeddofsth = results.fixeddofsth
            fp = results.fp
            feps = results.feps



            t_forward = time.time() - curr_time
            curr_time = time.time()

            # Plot design update
            if self.plot is True:
                plot_multi_physics(x, fixeddofsm, fixeddofsth, nelx, nely, fp, um, uth, open_plot=True)


            # ---------------------------------

            # ndofm = 2 * (nely + 1) * (nelx + 1)
            #
            # # Flatten the force matrix once
            # flam = fm.flatten()
            #
            # # --- Solve for free degrees of freedom ---
            # km_ff = km[freedofsm, :][:, freedofsm].tocsc()
            # flam_freedofs = flam[freedofsm]
            # lamm_freedofs = -spsolve(km_ff, flam_freedofs)
            # lamm = np.zeros(ndofm)
            # lamm[freedofsm] = lamm_freedofs
            #
            # # --- Sensitivity analysis or second solve ---
            # temp = lamm.T @ d_cthm - um.T @ d_cthm - fth.T
            # temp = np.asarray(temp).flatten()
            # kth_sparse = kth.tocsc()
            # lamth = spsolve(kth_sparse, temp)
            #
            # # ------- END OPTIMIZED CODE ------- #
            #
            # # PREPARE SENSITIVITY ANALYSIS
            # t_sensitivity = time.time() - curr_time
            # curr_time = time.time()
            # f0val = 0
            # f0valm = 0
            # f0valt = 0
            # df0dx_mat = np.zeros((nely, nelx))
            #
            # df0dx_m = np.zeros((nely, nelx))
            # df0dx_t = np.zeros((nely, nelx))
            #
            # xval = x.reshape(n, 1)
            # # DEFINE CONSTRAINTS
            # volconst = np.sum(x) / (volfrac * n) - 1  # shape ()
            # fval = volconst  # Column vector of size (1xm)
            # dfdx = np.ones((1, n)) / (volfrac * n)  # shape (1, n)
            #
            # # CALCULATE SENSITIVITIES
            # for elx in range(nelx):
            #     for ely in range(nely):
            #         n1 = (nely + 1) * elx + ely
            #         n2 = (nely + 1) * (elx + 1) + ely
            #         edof4 = np.array([n1 + 1, n2 + 1, n2, n1], dtype=int)
            #         edof8 = np.array(
            #             [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3, 2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1], dtype=int
            #         )
            #
            #         ume = um[edof8].flatten()
            #         uthe = uth[edof4].flatten()
            #         lamm[edof8].flatten()
            #         lamthe = lamth[edof4].flatten()
            #         x_p = x[ely, elx] ** penal
            #         x_p_minus1 = penal * x[ely, elx] ** (penal - 1)
            #
            #         # Separate
            #         f0valm += x_p * ume.T @ ke @ ume
            #         f0valt += x_p * uthe.T @ k_eth @ uthe
            #
            #         df0dx_m[ely, elx] = -x_p_minus1 * ume.T @ ke @ ume
            #         df0dx_t[ely, elx] = lamthe.T @ (x_p_minus1 * k_eth @ uthe)
            #         df0dx_mat[ely, elx] = (df0dx_m[ely, elx] * w1) + (df0dx_t[ely, elx] * w2)  # Weighted sensitivity
            #
            # f0val = (f0valm * w1) + (f0valt * w2)
            #
            # if self.eval_only is True:
            #     vf_error = np.abs(np.mean(x) - volfrac)
            #     return {
            #         "structural_compliance": f0valm,
            #         "thermal_compliance": f0valt,
            #         "volume_fraction": vf_error,
            #     }
            # vf_error = np.abs(np.mean(x) - volfrac)
            # obj_values = np.array([f0valm, f0valt, vf_error])
            # opti_step = None
            # opti_steps.append(opti_step)
            #
            #
            # df0dx = df0dx_mat.reshape(nely * nelx, 1)
            # df0dx = (h @ (xval * df0dx)) / hs[:, None] / np.maximum(1e-3, xval)  # Filtered sensitivity

            # # ------------------------------------------------------------------
            # # NEW OBJECTIVE & SENSITIVITY: energy-based thermoelastic formulation
            # # ------------------------------------------------------------------
            # ndofm = 2 * (nely + 1) * (nelx + 1)
            # nn = (nelx + 1) * (nely + 1)
            #
            # # Mechanical and thermal states
            # u = um  # mechanical displacement vector (size ndofm)
            # T = uth  # temperature vector (size nn)
            # f_th = results.feps  # mechanical thermal load vector (size ndofm)
            #
            # # Mechanical strain energy: E = 0.5 * u^T K u
            # # Use sparse matrix operations; km is CSR
            # E_mech = 0.5 * float(u.T @ (km @ u))
            #
            # # Thermal work: W_th = u^T f_th
            # W_th = float(u.T @ f_th)
            #
            # # Primary objective: maximize stiffness -> minimize -E_mech
            # f0val = -E_mech
            #
            # # --- Constraints ---
            # # Volume constraint as before
            # volconst = np.sum(x) / (volfrac * n) - 1.0
            # # You currently have m=1 in MMA; this is constraint index 0
            # fval = np.array([volconst])  # shape (1,)
            #
            # # Gradient of volume constraint
            # dfdx = np.ones((1, n)) / (volfrac * n)
            #
            # # If you want to additionally enforce an energy feasibility constraint
            # # Phi = E_mech - W_th <= 0 (i.e. thermal work can at least supply the stored energy),
            # # you can extend to m=2 and add its gradient as a second row in dfdx.
            # # For now, we'll keep only the volume constraint in MMA to keep structure identical.
            #
            #
            # # ---------------------------
            # # Sensitivity of objective
            # # ---------------------------
            # # We use element-wise expressions consistent with:
            # # dE/dx_e = 0.5 * penal * x_e^(penal-1) * u_e^T ke u_e
            # # plus thermal terms if you decide to incorporate E - W_th directly.
            # #
            # # We'll build df0dx_elem (nely x nelx) and then filter it as before.
            #
            # df0dx_elem = np.zeros((nely, nelx))
            #
            # # Build thermal adjoint for the Phi = E - W_th expression,
            # # even if we initially only use the mechanical part in objective.
            # # K_eth is d_cthm: mapping DeltaT -> f_th (global)
            # K_eth = results.d_cthm.tocsr()
            # # Right-hand side: psi = K_eth^T * u
            # psi = K_eth.T @ u
            # # Thermal adjoint: kth * eta = psi
            # kth_sparse = kth.tocsr()
            # eta = spsolve(kth_sparse, psi)
            #
            # # Loop over elements and build element contributions
            # for elx in range(nelx):
            #     for ely in range(nely):
            #         n1 = (nely + 1) * elx + ely
            #         n2 = (nely + 1) * (elx + 1) + ely
            #         edof4 = np.array([n1 + 1, n2 + 1, n2, n1], dtype=int)
            #         edof8 = np.array(
            #             [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3,
            #              2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1],
            #             dtype=int
            #         )
            #
            #         # Local fields
            #         u_e = u[edof8].flatten()  # mechanical DOFs
            #         T_e = T[edof4].flatten()  # thermal DOFs
            #         eta_e = eta[edof4].flatten()  # thermal adjoint DOFs
            #
            #         x_e = x[ely, elx]
            #         x_p = x_e ** penal
            #         x_p_minus1 = penal * x_e ** (penal - 1)
            #
            #         # Thermal conduction interpolation uses (2e-2 + x)^penal in your code
            #         kx = 2e-2 + x_e
            #         kx_p_minus1 = penal * kx ** (penal - 1)
            #
            #         # Element strain energy part: u_e^T ke u_e
            #         e_mech_elem = float(u_e.T @ (ke @ u_e))
            #
            #         # Element thermal “load” parts:
            #         diff_T = T_e - tref  # same as in fe_mthm_bc
            #         # K_eth element block acts as c_ethm on (DeltaT - tref)
            #         # we reuse c_ethm and k_eth here
            #         # u^T dK/dx u term for E_mech:
            #         dE_dx_mech = 0.5 * x_p_minus1 * e_mech_elem
            #
            #         # If you later want Phi = E - W_th, the additional terms per element are:
            #         #
            #         # term2 = - x_p_minus1 * (u_e^T (c_ethm @ diff_T))
            #         # term3 = + kx_p_minus1 * (eta_e^T (k_eth @ T_e))
            #         #
            #         # so that:
            #         #
            #         # dPhi_dx_e = -dE_dx_mech + term2 + term3
            #         #
            #         # For now, we keep the objective purely mechanical (maximize E),
            #         # so df0/dx_e = -dE/dx_e.
            #
            #         df0dx_elem[ely, elx] = -dE_dx_mech
            #
            # # Flatten objective gradient
            # df0dx = df0dx_elem.reshape(nely * nelx, 1)
            #
            # # Apply density filter to gradient (same as before)
            # xval = x.reshape(n, 1)
            # df0dx = (h @ (xval * df0dx)) / hs[:, None] / np.maximum(1e-3, xval)
            #
            # # If you later add Phi as a constraint, build its gradient as above
            # # and stack into dfdx as an extra row.
            # # (Don't forget to set m = 2 and extend MMAInputs accordingly.)

            # -------------------

            # -------------------------------
            # NEW OBJECTIVE & CONSTRAINTS
            # -------------------------------
            ndofm = 2 * (nely + 1) * (nelx + 1)
            nn = (nelx + 1) * (nely + 1)

            # Mechanical and thermal state variables
            u = um  # mechanical DOFs
            T = uth  # thermal DOFs
            f_th = feps  # thermal mechanical load (from expansion)

            # Mechanical strain energy: E = 0.5 * u^T K u
            E_mech = 0.5 * float(u.T @ (km @ u))

            # "Thermal work" measure: W_th = u^T f_th
            W_th = float(u.T @ f_th)

            # Primary objective: maximize stiffness -> minimize -E_mech
            f0val = -E_mech

            # -------------------
            # Constraint values
            # -------------------
            # 1) Volume constraint: sum(x)/(volfrac*n) - 1 <= 0
            volconst = np.sum(x) / (volfrac * n) - 1.0

            # 2) Energy feasibility-like constraint:
            #    Phi(x) = E_mech - W_th <= 0
            Phi = E_mech - W_th

            # Scale Phi for numerical stability
            phi_scale = max(abs(Phi), 1.0)
            g2 = Phi / phi_scale

            # fval shape: (m,) with m=2
            # fval = np.array([volconst, g2])
            fval = g2

            # Gradient of volume constraint
            dfdx_vol = np.ones((1, n)) / (volfrac * n)

            # ---------------------------
            # Gradient of Phi (adjoint)
            # ---------------------------
            # We use:
            # dPhi/dx_e = -0.5 u^T (dK_m/dx_e) u
            #             - u^T (dK_eth/dx_e) T
            #             + eta^T (dK_th/dx_e) T
            #
            # with:
            #   K_eth ~ d_cthm          (global mapping T->f_th)
            #   K_th  ~ kth
            #   eta solves:  K_th * eta = K_eth^T * u

            # K_eth operator
            K_eth = d_cthm.tocsr()  # shape (ndofm, nn)

            # Thermal adjoint: K_th * eta = K_eth^T * u
            psi = K_eth.T @ u  # size nn
            kth_sparse = kth.tocsr()
            eta = spsolve(kth_sparse, psi)  # size nn

            # Element-wise gradient of Phi
            dPhi_dx_elem = np.zeros((nely, nelx))

            for elx in range(nelx):
                for ely in range(nely):
                    n1 = (nely + 1) * elx + ely
                    n2 = (nely + 1) * (elx + 1) + ely
                    edof4 = np.array([n1 + 1, n2 + 1, n2, n1], dtype=int)
                    edof8 = np.array(
                        [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3,
                         2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1],
                        dtype=int
                    )

                    # Local fields
                    u_e = u[edof8].flatten()  # mechanical
                    T_e = T[edof4].flatten()  # thermal
                    eta_e = eta[edof4].flatten()  # adjoint thermal

                    x_e = x[ely, elx]
                    x_p = x_e ** penal
                    x_p_minus1 = penal * x_e ** (penal - 1)

                    # Thermal conductivity interpolation used in fe_mthm_bc:
                    # (2e-2 + x)^penal
                    kx = 2e-2 + x_e
                    kx_p_minus1 = penal * (kx ** (penal - 1))

                    # 1) dK_m/dx_e contribution
                    # K_m element contribution: x_e^penal * ke
                    e_mech_elem = float(u_e.T @ (ke @ u_e))
                    term1 = -0.5 * x_p_minus1 * e_mech_elem

                    # 2) dK_eth/dx_e contribution
                    # f_th element: x_e^penal * c_ethm @ (T_e - tref)
                    diff_T = T_e - tref
                    term2 = - x_p_minus1 * float(u_e.T @ (c_ethm @ diff_T))

                    # 3) dK_th/dx_e contribution
                    # K_th element: (2e-2 + x_e)^penal * k_eth
                    term3 = kx_p_minus1 * float(eta_e.T @ (k_eth @ T_e))

                    dPhi_dx_elem[ely, elx] = term1 + term2 + term3

                # Constraint gradient for g2 = Phi/phi_scale
            dPhi_dx = dPhi_dx_elem.reshape(n, 1) / phi_scale

            # Filter gradient of Phi using same density filter
            xval = x.reshape(n, 1)
            dPhi_dx_filt = (h @ (xval * dPhi_dx)) / hs[:, None] / np.maximum(1e-3, xval)

            # Stack constraint gradients: shape (2, n)
            # dfdx = np.vstack([
            #     dfdx_vol,
            #     dPhi_dx_filt.reshape(1, n),
            # ])
            dfdx = dPhi_dx_filt.reshape(1, n)

            # ---------------------------
            # Objective gradient
            # ---------------------------
            # For objective f0 = -E_mech, we use:
            #   E_mech = 0.5 sum_e x_e^p u_e^T ke u_e
            #   dE/dx_e = 0.5 * penal * x_e^(penal-1) * u_e^T ke u_e
            # so df0/dx_e = -dE/dx_e
            df0dx_elem = np.zeros((nely, nelx))

            for elx in range(nelx):
                for ely in range(nely):
                    n1 = (nely + 1) * elx + ely
                    n2 = (nely + 1) * (elx + 1) + ely
                    edof8 = np.array(
                        [2 * n1 + 2, 2 * n1 + 3, 2 * n2 + 2, 2 * n2 + 3,
                         2 * n2, 2 * n2 + 1, 2 * n1, 2 * n1 + 1],
                        dtype=int
                    )

                    u_e = u[edof8].flatten()
                    x_e = x[ely, elx]
                    x_p_minus1 = penal * x_e ** (penal - 1)
                    e_mech_elem = float(u_e.T @ (ke @ u_e))

                    dE_dx_mech = 0.5 * x_p_minus1 * e_mech_elem
                    df0dx_elem[ely, elx] = -dE_dx_mech

            df0dx = df0dx_elem.reshape(n, 1)
            df0dx = (h @ (xval * df0dx)) / hs[:, None] / np.maximum(1e-3, xval)




            t_sensitivity_calc = time.time() - curr_time
            curr_time = time.time()


            # UPDATE DESIGN VARIABLES USING MMA
            mmainputs = MMAInputs(
                m=m,
                n=n,
                iterr=iterr,
                xval=xval[:, 0],  # selecting appropriate column
                xmin=xmin,
                xmax=xmax,
                xold1=xold1,
                xold2=xold2,
                df0dx=df0dx[:, 0],  # selecting appropriate column
                fval=fval,  # Constraint values
                dfdx=dfdx,
                low=low_vec,
                upp=upp_vec,
                a0=a0,
                a=a[0],
                c=c[0],
                d=d[0],
                f0val=f0val,
            )
            xmma, low_vec, upp_vec = mmasub(mmainputs)
            t_mma = time.time() - curr_time

            # reshape low_vec to (-1,)
            low_vec = np.squeeze(low_vec)
            upp_vec = np.squeeze(upp_vec)

            # Store previous density fields
            if iterr > SECOND_ITERATION_THRESHOLD:
                xold2 = xold1
                xold1 = xval
            elif iterr > FIRST_ITERATION_THRESHOLD:
                xold1 = xval

            x = xmma.reshape(nely, nelx)

            # Print results
            change = np.max(np.abs(xmma - xold1))
            change_evol.append(change)
            obj.append(f0val)
            t_total = time.time() - s_time
            print(
                f" It.: {iterr:4d} Obj.: {f0val:10.4f} Vol.: {np.sum(x) / (nelx * nely):6.3f} ch.: {change:6.3f} || t_forward:{t_forward:6.3f} + t_sens_calc:{t_sensitivity_calc:6.3f} + t_mma: {t_mma:6.3f} = {t_total:6.3f}"
            )

            if iterr > MAX_ITERATIONS:
                break

        print("Optimization finished...")
        vf_error = np.abs(np.mean(x) - volfrac)

        return {
            "design": x,
            "bcs": bcs,
            "structural_compliance": f0valm,
            "thermal_compliance": f0valt,
            "volume_fraction": vf_error,
            "opti_steps": opti_steps,
        }


if __name__ == "__main__":
    nelx = 64
    nely = 64

    client = FeaModel(plot=True)

    lci, tri, rci, bri = get_res_bounds(nelx + 1, nely + 1)

    fixed_elements = np.zeros((nelx + 1, nely + 1))
    fixed_elements[0, 0] = 1
    fixed_elements[-1, 0] = 1

    force_elements_x = np.zeros((nelx + 1, nely + 1))
    force_elements_x[-1, -1] = 1

    force_elements_y = np.zeros((nelx + 1, nely + 1))
    force_elements_y[-1, -1] = 1

    heatsink_elements = np.zeros((nelx + 1, nely + 1))
    heatsink_elements[32, 0] = 1


    bcs = {
        "nelx": nelx,
        "nely": nely,
        "fixed_elements": fixed_elements,
        "force_elements_x": force_elements_x,
        "force_elements_y": force_elements_y,
        "heatsink_elements": heatsink_elements,
        "volfrac": 0.2,
        "rmin": 1.5,
        "weight": 0.0,  # 1.0 for pure structural, 0.0 for pure thermal
    }

    result = client.run(bcs)