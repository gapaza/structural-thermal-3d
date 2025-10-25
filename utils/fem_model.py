from math import ceil
from math import hypot
import time
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# Reuse your existing MMA wrapper
# from engibench.core import OptiStep
# from engibench.problems.thermoelastic2d.model.mma_subroutine import MMAInputs, mmasub
from utils.mma_subroutine import MMAInputs, mmasub

# The new 3D element builder and 3D assembly/BC routine you now have
# (adjust imports to your actual module locations)
# from engibench.problems.thermoelastic3d.model.fem_matrix_builder import fe_melthm_3d
# from engibench.problems.thermoelastic3d.model.fem_setup import fe_mthm_bc_3d
from utils.fem_matrix_builder import fe_melthm_3d
from utils.fem_setup import fe_mthm_bc_3d

# For this snippet assume they are in scope:
# def fe_melthm_3d(nu: float, E: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...
# def fe_mthm_bc_3d(...): ...


from utils.fem_plotting import plot_fem_3d

from utils.linear_solver import solve_spd_with_amg


SECOND_ITERATION_THRESHOLD = 2
FIRST_ITERATION_THRESHOLD = 1
MIN_ITERATIONS = 10
MAX_ITERATIONS = 600
UPDATE_THRESHOLD = 0.01


class FeaModel3D:
    """Finite Element Analysis (FEA) model for coupled 3D thermoelastic topology optimization."""

    def __init__(self, *, plot: bool = False, eval_only: bool | None = False) -> None:
        """Instantiates a new 3D thermoelastic model.

        Args:
            plot: If True, you can hook in your own plotting / volume rendering each iteration.
            eval_only: If True, evaluate the given design once and return objective components only.
        """
        self.plot = plot
        self.eval_only = eval_only

    # -----------------------------
    # Initial design (3D voxel grid)
    # -----------------------------
    def get_initial_design(self, volume_fraction: float, nelx: int, nely: int, nelz: int) -> np.ndarray:
        """Initial (nely, nelx, nelz) density field."""
        return volume_fraction * np.ones((nely, nelx, nelz), dtype=float)

    # -----------------------------
    # Element matrices (Hex8)
    # -----------------------------
    def get_matrices(self, nu: float, E: float, k: float, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (ke 24x24, k_eth 8x8, c_ethm 24x8) for Hex8."""
        return fe_melthm_3d(nu, E, k, alpha)

    # -----------------------------
    # 3D sensitivity filter
    # -----------------------------
    def get_filter(self, nelx: int, nely: int, nelz: int, rmin: float) -> tuple[coo_matrix, np.ndarray]:
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
        return H, hs

    # -----------------------------
    # Main optimization loop
    # -----------------------------
    def run(self, bcs: dict[str, Any], x_init: np.ndarray | None = None) -> dict[str, Any]:  # noqa: PLR0915
        """Run 3D thermoelastic topology optimization.

        BC expectations (node-based 3D boolean masks of shape (nelx+1, nely+1, nelz+1)):
            - 'fixed_elements'
            - 'heatsink_elements'
            - optional 'force_elements_x', 'force_elements_y', 'force_elements_z'
        """
        # Weighting
        w1 = bcs.get("weight", 0.5)  # structural
        w2 = 1.0 - w1                # thermal

        # Derive mesh sizes from fixed_elements mask (shape: (nelx+1, nely+1, nelz+1))
        fixed_nodes_mask = np.asarray(bcs["fixed_elements"], dtype=bool)
        if fixed_nodes_mask.ndim != 3:
            raise ValueError("3D FeaModel3D expects node masks of shape (nelx+1, nely+1, nelz+1).")

        nxp, nyp, nzp = fixed_nodes_mask.shape  # nodes-per-direction
        nelx, nely, nelz = nxp - 1, nyp - 1, nzp - 1
        n = nelx * nely * nelz  # number of elements

        volfrac = bcs["volfrac"]

        # Optimization history (optional)
        # opti_steps: list[OptiStep] = []

        # 1) Initial design
        x = self.get_initial_design(volfrac, nelx, nely, nelz) if x_init is None else x_init.copy()


        # 2) Parameters
        penal = 3.0
        rmin = bcs.get("rmin", 1.1)
        E = 1.0
        nu = 0.3
        k = 1.0
        alpha = 5e-4
        tref = 9.267e-4

        change = 1.0
        iterr = 0
        xmin, xmax = 1e-3, 1.0

        # MMA scaffolding
        xold1 = x.reshape(n, 1)
        xold2 = x.reshape(n, 1)
        m = 1  # volume constraint
        a0 = 1.0
        a = np.zeros((m, 1))
        c = 10000.0 * np.ones((m, 1))
        d = np.zeros((m, 1))
        low = xmin
        upp = xmax

        # 3) Element matrices
        ke, k_eth, c_ethm = self.get_matrices(nu, E, k, alpha)

        # 4) 3D filter
        H, hs = self.get_filter(nelx, nely, nelz, rmin)

        print("Starting 3D optimization...")

        # 5) Loop
        f0valm = 0.0
        f0valt = 0.0
        change_evol = []
        obj_evol = []

        while change > UPDATE_THRESHOLD or iterr < MIN_ITERATIONS:
            iterr += 1
            t0 = time.time()
            tcur = t0

            # Forward FEA with BCs & assembly (3D)
            res = fe_mthm_bc_3d(nely, nelx, nelz, penal, x, ke, k_eth, c_ethm, tref, bcs)

            km = res.km
            kth = res.kth
            um = res.um
            uth = res.uth
            fm = res.fm
            fth = res.fth
            d_cthm = res.d_cthm
            fixeddofsm = res.fixeddofsm
            freedofsm = res.freedofsm
            fixeddofsth = res.fixeddofsth
            fp = res.fp

            if self.plot is True and (iterr % 50 == 0):
                plot_fem_3d(bcs, x)


            t_forward = time.time() - tcur
            tcur = time.time()

            # Mechanical adjoint: solve K_m * lambda_m = -f_m on free dofs
            ndofm = um.size
            flam = fm  # already flat
            km_ff = km[freedofsm, :][:, freedofsm].tocsc()
            rhs_m = -flam[freedofsm]

            # Recomputing lamm directly (only if force depends on design, which it does not here) TODO: solver change here
            # lamm_free = spsolve(km_ff, rhs_m)
            # lamm = np.zeros(ndofm, dtype=float)
            # lamm[freedofsm] = lamm_free
            lamm = -um

            # Thermal adjoint: K_th * lambda_th = (lamm^T - um^T) * d_cthm - f_th TODO: solver change here
            # (vector on thermal dofs)
            # temp = (lamm @ d_cthm - um @ d_cthm) - fth
            # lamth = spsolve(kth.tocsc(), temp)
            lamth = solve_spd_with_amg(kth.tocsr(), (lamm @ d_cthm - um @ d_cthm) - fth)

            t_adjoints = time.time() - tcur
            tcur = time.time()

            # Sensitivities & objective
            f0val = 0.0
            f0valm = 0.0
            f0valt = 0.0

            df0dx_m = np.zeros_like(x)   # (nely, nelx, nelz)
            df0dx_t = np.zeros_like(x)
            df0dx_mat = np.zeros_like(x)

            # Element DOF helper consistent with fe_mthm_bc_3d
            def elem_dofs(elx: int, ely: int, elz: int):
                # Build element node ids (same Hex8 order)
                def node_id(ix, iy, iz):
                    return (nely + 1) * (nelz + 1) * ix + (nelz + 1) * iy + iz

                n000 = node_id(elx,     ely,     elz)
                n100 = node_id(elx + 1, ely,     elz)
                n110 = node_id(elx + 1, ely + 1, elz)
                n010 = node_id(elx,     ely + 1, elz)
                n001 = node_id(elx,     ely,     elz + 1)
                n101 = node_id(elx + 1, ely,     elz + 1)
                n111 = node_id(elx + 1, ely + 1, elz + 1)
                n011 = node_id(elx,     ely + 1, elz + 1)

                edof8 = np.array([n000, n100, n110, n010, n001, n101, n111, n011], dtype=int)
                edof24 = np.empty(24, dtype=int)
                edof24[0::3] = 3 * edof8 + 0
                edof24[1::3] = 3 * edof8 + 1
                edof24[2::3] = 3 * edof8 + 2
                return edof8, edof24

            # Loop elements (clear & readable; vectorization is possible later)
            for elx in range(nelx):
                for ely in range(nely):
                    for elz in range(nelz):
                        edof8, edof24 = elem_dofs(elx, ely, elz)

                        ue = um[edof24]          # (24,)
                        the = uth[edof8]         # (8,)
                        lamthe = lamth[edof8]    # (8,)

                        x_e = x[ely, elx, elz]
                        x_p = x_e ** penal
                        x_p_minus1 = penal * (x_e ** (penal - 1))

                        # Element contributions
                        f0valm += x_p * (ue @ (ke @ ue))
                        f0valt += x_p * (the @ (k_eth @ the))

                        # Sensitivities (weighted later)
                        df0dx_m[ely, elx, elz] = -x_p_minus1 * (ue @ (ke @ ue))
                        df0dx_t[ely, elx, elz] = lamthe @ (x_p_minus1 * (k_eth @ the))

                        # (df0dx_mat is formed after weighting)
            f0val = w1 * f0valm + w2 * f0valt
            df0dx_mat = w1 * df0dx_m + w2 * df0dx_t

            if self.eval_only:
                vf_error = abs(np.mean(x) - volfrac)
                return {
                    "structural_compliance": float(f0valm),
                    "thermal_compliance": float(f0valt),
                    "volume_fraction": vf_error,
                }

            # Constraint: volume
            xval = x.reshape(n, 1)
            volconst = np.sum(x) / (volfrac * n) - 1.0
            fval = volconst  # scalar
            dfdx = np.ones((1, n), dtype=float) / (volfrac * n)

            # Apply 3D filter to sensitivities
            df0dx_vec = df0dx_mat.reshape(n, 1)
            df0dx_filt = (H @ (xval * df0dx_vec)) / hs[:, None] / np.maximum(1e-3, xval)

            t_sens = time.time() - tcur
            tcur = time.time()

            # MMA update
            upp_vec = np.ones((n,), dtype=float) * upp
            low_vec = np.ones((n,), dtype=float) * low

            mmainputs = MMAInputs(
                m=1,
                n=n,
                iterr=iterr,
                xval=xval[:, 0],
                xmin=xmin,
                xmax=xmax,
                xold1=xold1,
                xold2=xold2,
                df0dx=df0dx_filt[:, 0],
                fval=fval,
                dfdx=dfdx,
                low=low_vec,
                upp=upp_vec,
                a0=a0,
                a=a[0],
                c=c[0],
                d=d[0],
                f0val=f0val,
            )
            xmma = mmasub(mmainputs)

            # Shift history
            if iterr > SECOND_ITERATION_THRESHOLD:
                xold2 = xold1
                xold1 = xval
            elif iterr > FIRST_ITERATION_THRESHOLD:
                xold1 = xval

            # Update design
            x = xmma.reshape(nely, nelx, nelz)

            # Progress
            change = np.max(np.abs(xmma - xold1))
            change_evol.append(change)
            obj_evol.append(f0val)

            t_mma = time.time() - tcur
            t_total = time.time() - t0
            print(
                f" It.: {iterr:4d} Obj.: {f0val:10.4f} "
                f"Vol.: {np.sum(x)/(nelx*nely*nelz):6.3f} ch.: {change:6.3f} "
                f"|| t_forward:{t_forward:6.3f} + t_adj:{t_adjoints:6.3f} + t_sens:{t_sens:6.3f} + t_mma:{t_mma:6.3f} = {t_total:6.3f}"
            )

            if iterr > MAX_ITERATIONS:
                break

        print("3D optimization finished.")
        vf_error = abs(np.mean(x) - volfrac)

        # Record last step
        # opti_steps.append(OptiStep(obj_values=np.array([f0valm, f0valt, vf_error]), step=iterr))

        return {
            "design": x,
            "bcs": bcs,
            "structural_compliance": float(f0valm),
            "thermal_compliance": float(f0valt),
            "volume_fraction": vf_error,
            # "opti_steps": opti_steps,
        }







def indices_to_binary_matrix(indices: list[int], nelx: int, nely: int, nelz: int) :
    """Converts a list of indices to a binary matrix of a specific size.

    Args:
        indices (list[int]): The list of indices to set to 1 in the binary matrix.
        nelx (int): Number of elements in the x-direction.
        nely (int): Number of elements in the y-direction.

    Returns:
        npt.NDArray: The binary matrix.
    """
    flat_matrix = np.zeros((nelx * nely * nelz,), dtype=int)
    flat_matrix[indices] = 1
    matrix = flat_matrix.reshape((nelx, nely, nelz))
    return matrix



if __name__ == '__main__':

    nelx = 48
    nely = 48
    nelz = 48

    fixed_elements_matrix = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=int)
    fixed_elements_matrix[0, 0, 0] = 1
    fixed_elements_matrix[0, -1, -1] = 1
    fixed_elements_matrix[0, -1, 0] = 1

    force_elements_x_matrix = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=int)
    force_elements_x_matrix[-1, -1, -1] = 1

    force_elements_y_matrix = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=int)
    force_elements_y_matrix[-1, -1, -1] = 1

    foce_elements_z_matrix = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=int)
    force_elements_y_matrix[-1, -1, -1] = 1

    heatsink_elements_matrix = np.zeros((nelx + 1, nely + 1, nelz + 1), dtype=int)
    heatsink_elements_matrix[-1, -1, 0] = 1

    conditions: tuple[tuple[str, Any], ...] = (
        ("fixed_elements", fixed_elements_matrix),
        ("force_elements_x", force_elements_x_matrix),
        ("force_elements_y", force_elements_y_matrix),
        ("force_elements_z", foce_elements_z_matrix),
        ("heatsink_elements", heatsink_elements_matrix),

        # Elastic
        ("volfrac", 0.1),
        ("rmin", 1.5),
        ("weight", 1.0),  # 1.0 for pure structural, 0.0 for pure thermal

        # # Thermal
        # ("volfrac", 0.2),
        # ("rmin", 1.5),
        # ("weight", 0.0),  # 1.0 for pure structural, 0.0 for pure thermal

        # # Thermo-elastic
        # ("volfrac", 0.2),
        # ("rmin", 1.5),
        # ("weight", 0.5),  # 1.0 for pure structural, 0.0 for pure thermal

    )
    conditions = dict(conditions)

    starting_point = conditions['volfrac'] * np.ones((nelx, nely, nelz), dtype=float)

    results = FeaModel3D(plot=True, eval_only=False).run(conditions, x_init=starting_point)














