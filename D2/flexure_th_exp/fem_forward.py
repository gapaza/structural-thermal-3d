from math import ceil, hypot
import time
from typing import Any, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

from D2.thermoelastic.fem_matrix_builder import fe_melthm
from D2.thermoelastic.mma_subroutine import MMAInputs, mmasub
from D2.flexure.utils import plot_design

FIRST_ITERATION_THRESHOLD = 1
SECOND_ITERATION_THRESHOLD = 2
MIN_ITERATIONS = 10
MAX_ITERATIONS = 800
UPDATE_THRESHOLD = 5e-3


def build_filter(nelx: int, nely: int, rmin: float) -> tuple[csr_matrix, np.ndarray]:
    """Density/sensitivity filter (cone weights), in sparse form."""
    ih: list[int] = []
    jh: list[int] = []
    sh: list[float] = []
    for ex in range(nelx):
        ex_min = max(ex - (ceil(rmin) - 1), 0)
        ex_max = min(ex + (ceil(rmin) - 1), nelx - 1)
        for ey in range(nely):
            e1 = ex * nely + ey
            ey_min = max(ey - (ceil(rmin) - 1), 0)
            ey_max = min(ey + (ceil(rmin) - 1), nely - 1)
            for ex2 in range(ex_min, ex_max + 1):
                for ey2 in range(ey_min, ey_max + 1):
                    e2 = ex2 * nely + ey2
                    ih.append(e1)
                    jh.append(e2)
                    sh.append(max(0.0, rmin - hypot(ex - ex2, ey - ey2)))
    H = coo_matrix((sh, (ih, jh)), shape=(nelx * nely, nelx * nely)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).ravel()
    return H, Hs


def edof_mappings(nelx: int, nely: int) -> np.ndarray:
    """Element-to-DOF map (edof8 for mechanics) vectorized over all elements."""
    # Element indices on a meshgrid
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    ex = ex.ravel()
    ey = ey.ravel()
    # Node numbers (0-based)
    n1 = (nely + 1) * ex + ey        # bottom-left node index
    n2 = (nely + 1) * (ex + 1) + ey  # bottom-right
    n3 = n2 + 1                      # top-right
    n4 = n1 + 1                      # top-left

    # 8 mechanical dofs per element (ux, uy per node)
    edof8 = np.stack(
        [
            2 * n1,     2 * n1 + 1,   # node 1 (bottom-left)
            2 * n2,     2 * n2 + 1,   # node 2 (bottom-right)
            2 * n3,     2 * n3 + 1,   # node 3 (top-right)
            2 * n4,     2 * n4 + 1,   # node 4 (top-left)
        ],
        axis=1,
    )
    return edof8.astype(int)


def thermal_edof_mappings(nelx: int, nely: int) -> np.ndarray:
    """Element-to-node map (edof4 for thermal problem), 4 temperature DOFs per element."""
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    ex = ex.ravel()
    ey = ey.ravel()
    n1 = (nely + 1) * ex + ey        # bottom-left node
    n2 = (nely + 1) * (ex + 1) + ey  # bottom-right
    n3 = n2 + 1                      # top-right
    n4 = n1 + 1                      # top-left
    edof4 = np.stack([n1, n2, n3, n4], axis=1)
    return edof4.astype(int)


def prescribed_u_matrix(
    nelx: int, nely: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the prescribed displacement matrix U[:,3] for tx, ty, rz and fixed/free sets.

    Returns:
        U       : (ndof, 3) prescribed displacements for tx,ty,rz
        fixed   : indices of fixed mechanical DOFs (top + bottom)
        free    : indices of free mechanical DOFs
        top_nodes    : node indices of the top edge (for reference)
        bottom_nodes : node indices of the bottom edge (for thermal BCs)
    """
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Node indexing on rect grid:
    # y-index 0..nely, x-index 0..nelx
    def node_id(ix: int, iy: int) -> int:
        return (nely + 1) * ix + iy

    # Top (iy = 0) and bottom (iy = nely)
    bottom_nodes = [node_id(ix, nely) for ix in range(nelx + 1)]
    top_nodes = [node_id(ix, 0) for ix in range(nelx + 1)]

    # DOF indices (0-based): ux=2*n, uy=2*n+1
    top_ux = np.array([2 * n for n in top_nodes], dtype=int)
    top_uy = top_ux + 1
    bot_ux = np.array([2 * n for n in bottom_nodes], dtype=int)
    bot_uy = bot_ux + 1

    fixed = np.concatenate([top_ux, top_uy, bot_ux, bot_uy])
    all_dofs = np.arange(ndof, dtype=int)
    free = np.setdiff1d(all_dofs, fixed, assume_unique=False)

    # U for three mechanism cases: 0->tx, 1->ty, 2->rz
    U = np.zeros((ndof, 3), dtype=float)

    # ty: vertical unit translation at top
    U[top_uy, 1] = 1.0
    # tx: horizontal unit translation at top
    U[top_ux, 0] = 1.0
    # rz: approximate small rotation: ux=1 (uniform), uy varies linearly across x
    U[top_ux, 2] = 1.0
    U[top_uy, 2] = np.linspace(1.0, -1.0, nelx + 1)  # center of rotation mid-span in x

    return U, fixed, free, top_nodes, bottom_nodes


class FlexureThermoModel:
    """
    2D flexure synthesis via topology optimization with thermo-elastic coupling.

    - Weakly coupled 2D thermo-elasticity
    - Steady-state heat conduction with uniform volumetric heat generation
    - Thermal expansion induces forces; prescribed interface displacements define mechanism degrees
    - DOCs: stiffness (strain energy) maximized (via objective)
    - DOFs: energy balance constraint E - W_th <= 0 (E = strain energy, W_th = thermal work)
    """

    def __init__(self, *, plot: bool = False, eval_only: bool | None = False) -> None:
        self.plot = plot
        self.eval_only = eval_only

    @staticmethod
    def get_initial_design(volfrac_init: float, nelx: int, nely: int) -> np.ndarray:
        return volfrac_init * np.ones((nely, nelx), dtype=float)

    def run(self, bcs: dict[str, Any], x_init: np.ndarray | None = None) -> dict[str, Any]:  # noqa: PLR0915
        """
        Required keys in bcs:
            - nelx, nely : ints
            - doc : str or list[str] subset of {"tx","ty","rz"}
            - dof : str or list[str] subset of {"tx","ty","rz"} disjoint from doc

        Thermal / material parameters:
            - E0   : float, base Young's modulus (default 1.0)
            - k0   : float, base thermal conductivity (default 1.0)
            - alpha: float, thermal expansion coefficient (default 1.0)
            - q    : float, uniform volumetric heat generation (default 1.0)

        Constraint scaling:
            - emax : float or list[float], scaling for energy balance constraints
                     (g_j = (E_j - W_th_j) / emax_j <= 0)

        Optional:
            - rmin (float) default 2.0
            - penal (float) default 3.0
            - volfrac_init (float) default 0.2
        """
        nelx: int = int(bcs["nelx"])
        nely: int = int(bcs["nely"])

        # Degrees parsing
        all_deg = ["tx", "ty", "rz"]

        def to_mask(s: str | Sequence[str]) -> np.ndarray:
            if isinstance(s, str):
                s = [s]
            return np.array([d in set(s) for d in all_deg], dtype=bool)

        mdoc = to_mask(bcs["doc"])  # DOC mask
        mdof = to_mask(bcs["dof"])  # DOF mask
        assert not np.any(mdoc & mdof), "DOC and DOF sets must be disjoint"
        deg_mask = mdoc | mdof
        deg_idx = np.where(deg_mask)[0]  # active mechanism indices (0..2)
        ldoc = int(mdoc.sum())
        ldof = int(mdof.sum())
        assert 1 <= ldoc <= 2, "DOC must have 1 or 2 entries"
        assert 1 <= ldof <= 2, "DOF must have 1 or 2 entries"

        # emax handling (per DOF) – used as scaling for (E - W_th)
        emax_in = bcs["emax"]
        if isinstance(emax_in, (float, int)):
            emax = np.array([float(emax_in)] * ldof, dtype=float)
        else:
            emax = np.array([float(v) for v in emax_in], dtype=float)
            assert len(emax) == ldof, "Length of emax must match number of DOFs"

        # Parameters
        penal = float(bcs.get("penal", 3.0))
        rmin = float(bcs.get("rmin", 2.0))
        volfrac_init = float(bcs.get("volfrac_init", 0.2))

        # Material / thermal parameters
        E0 = float(bcs.get("E0", 1.0))
        k0 = float(bcs.get("k0", 1.0))
        alpha = float(bcs.get("alpha", 1.0))
        q0 = float(bcs.get("q", 1.0))  # uniform volumetric heat generation (dimensionless here)

        nu = 0.3
        eps = 1e-9
        eps_th = 1e-3  # minimum thermal conductivity
        penal_th = penal  # can be chosen differently if desired

        # Design & filter
        n = nelx * nely
        x = self.get_initial_design(volfrac_init, nelx, nely) if x_init is None else x_init.copy()
        H, Hs = build_filter(nelx, nely, rmin)

        # Element matrices (base, normalized: E=1, k=1, alpha=1)
        ke_base, kth_base, cethm_base = fe_melthm(nu, 1.0, 1.0, 1.0)

        # Global mappings
        edof8 = edof_mappings(nelx, nely)        # mechanical 8 dofs/element
        edof4 = thermal_edof_mappings(nelx, nely)  # thermal 4 nodes/element

        nnode = (nelx + 1) * (nely + 1)   # number of temperature DOFs
        ndof = 2 * nnode                  # number of mechanical DOFs

        U_presc, fixed, free, top_nodes, bottom_nodes = prescribed_u_matrix(nelx, nely)

        # Thermal boundary conditions: fix T=0 at bottom edge
        th_fixed = np.array(bottom_nodes, dtype=int)
        th_all = np.arange(nnode, dtype=int)
        th_free = np.setdiff1d(th_all, th_fixed, assume_unique=False)

        # Pre-build global stiffness sparsity patterns
        n_elem = n

        # Mechanical stiffness K_mech: K = Σ_e (E_e * ke_base)
        K_rows = np.repeat(edof8, 8, axis=1).ravel()   # 8x8 per element → 64 entries
        K_cols = np.tile(edof8, 8).ravel()
        KE_flat = ke_base.reshape(1, -1)               # shape (1, 64)

        # Thermal stiffness K_th: K_th = Σ_e (k_e * kth_base)
        Kth_rows = np.repeat(edof4, 4, axis=1).ravel()  # 4x4 → 16 entries
        Kth_cols = np.tile(edof4, 4).ravel()
        Kth_flat = kth_base.reshape(1, -1)              # shape (1, 16)

        # Thermo-elastic coupling K_eth: f_th = Σ_e (γ_e * cethm_base * ΔT_e)
        # cethm_base is 8x4; global shape K_eth: (ndof, nnode)
        Keth_rows = np.repeat(edof8, 4, axis=1).ravel()  # 8x4 → 32 entries
        Keth_cols = np.tile(edof4, 8).ravel()
        Ceth_flat = cethm_base.reshape(1, -1)            # shape (1, 32)

        # Global thermal load vector from uniform volumetric heat generation
        Q_th = np.zeros(nnode, dtype=float)
        # Consistent nodal heat loads: q0 * area * (1/4) per node (area=1 here)
        for e in range(n_elem):
            nodes_e = edof4[e]
            Q_th[nodes_e] += q0 / 4.0

        # MMA setup (no volume constraint by default; only energy-balance constraints)
        m = ldof  # number of inequality constraints
        xmin = 1e-3
        xmax = 1.0
        xold1 = x.reshape(n, 1)
        xold2 = x.reshape(n, 1)
        a0 = 1.0
        a = np.zeros((m, 1))
        c = 10000 * np.ones((m, 1))
        d = np.zeros((m, 1))
        low_vec = xmin * np.ones((n,))
        upp_vec = xmax * np.ones((n,))

        it = 0
        change = 1.0
        alpha_scale = np.ones(3, dtype=float)  # energy scaling per degree (set at iter 1)

        energies_hist: list[np.ndarray] = []

        while change > UPDATE_THRESHOLD or it < MIN_ITERATIONS:
            it += 1
            t0 = time.time()

            # --- Symmetry (optional) ---
            # x = 0.5 * (x + np.flipud(x))
            # x = 0.5 * (x + np.fliplr(x))

            # --- Density filter (x -> xPhys) ---
            xval = x.reshape(n, 1)
            xPhys = (H @ xval) / Hs[:, None]
            xPhys = xPhys.reshape(nely, nelx)

            # --- SIMP interpolation ---
            x_flat = xPhys.ravel()

            # Mechanical modulus per element
            Ee = eps + (E0 - eps) * x_flat**penal
            # Thermal conductivity per element
            k_e = eps_th + (k0 - eps_th) * x_flat**penal_th
            # Coupling scale γ_e ~ E * alpha
            gamma_e = alpha * Ee

            # --- Assemble mechanical stiffness K_mech ---
            K_data = (Ee[:, None] * KE_flat).ravel()
            K = coo_matrix((K_data, (K_rows, K_cols)), shape=(ndof, ndof)).tocsr()
            K = 0.5 * (K + K.T)  # enforce symmetry

            # --- Assemble thermal stiffness K_th ---
            Kth_data = (k_e[:, None] * Kth_flat).ravel()
            K_th = coo_matrix((Kth_data, (Kth_rows, Kth_cols)), shape=(nnode, nnode)).tocsr()
            K_th = 0.5 * (K_th + K_th.T)

            # --- Assemble thermo-elastic coupling K_eth ---
            Keth_data = (gamma_e[:, None] * Ceth_flat).ravel()
            K_eth = coo_matrix((Keth_data, (Keth_rows, Keth_cols)), shape=(ndof, nnode)).tocsr()

            # --- Solve thermal problem: K_th * ΔT = Q_th with T=0 on bottom ---
            T = np.zeros(nnode, dtype=float)
            rhs_th = Q_th[th_free]  # no prescribed nonzero T on fixed nodes
            T_free = spsolve(K_th[np.ix_(th_free, th_free)], rhs_th)
            T[th_free] = T_free
            # ΔT relative to reference 0:
            dT = T  # reference temperature = 0

            # --- Thermal forces: f_th = K_eth * ΔT ---
            f_th = K_eth @ dT  # (ndof,)

            # --- Solve mechanical equilibrium for each active mechanism case ---
            U = np.zeros((ndof, 3), dtype=float)
            energies = np.zeros(3, dtype=float)
            W_th = np.zeros(3, dtype=float)

            for ii in deg_idx:
                # Partition mechanical equilibrium:
                # K_ff * U_f = f_th_f - K_fp * U_p
                rhs_mech = f_th[free] - K[np.ix_(free, fixed)] @ U_presc[fixed, ii]
                U_free = spsolve(K[np.ix_(free, free)], rhs_mech)

                U[:, ii] = 0.0
                U[fixed, ii] = U_presc[fixed, ii]
                U[free, ii] = U_free

                Ku = K @ U[:, ii]
                energies[ii] = 0.5 * float(U[:, ii].T @ Ku)
                W_th[ii] = float(U[:, ii].T @ f_th)

            # --- Scaling factors alpha (first iteration) ---
            if it == 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha_scale = np.zeros(3, dtype=float)
                    nz = energies > 0
                    alpha_scale[nz] = 1.0 / energies[nz]
                    # Empirical doubling for rotation "rz" (index 2), if used
                    alpha_scale[2] *= 2.0

            # --- Objective and constraints ---
            # Objective: maximize stiffness on DOCs (strain energy), same style as MATLAB
            f0val = -np.mean(alpha_scale[mdoc] * energies[mdoc])

            # DOF constraints: energy balance Φ_i = E_i - W_th_i <= 0 (scaled)
            gscale = 0.000001
            Phi = energies - W_th
            Phi_dof = Phi[mdof]
            fval = gscale * (Phi_dof / emax)  # shape (ldof,)

            if self.eval_only:
                return {
                    "design": x,
                    "energies": energies,
                    "thermal_work": W_th,
                    "objective": f0val,
                    "constraints": fval,
                }

            # --- Sensitivities (approximate: only dE/dx, ignoring dW_th/dx) ---
            # Elemental energies per mechanism: ce_i(e) = 0.5 * u_e^T ke_base u_e
            dc_i = np.zeros((nely, nelx, 3), dtype=float)
            for kdeg in deg_idx:
                Uk = U[:, kdeg]
                Ue = Uk[edof8]  # (n_elem, 8)

                # Element strain energy (using base ke; scaling via Ee moved to SIMP derivative)
                ce = 0.5 * np.einsum("ij,ij->i", Ue @ ke_base, Ue).reshape(nely, nelx)

                # Local sensitivity w.r.t. xPhys:
                # dE/dxPhys ≈ penal * (1-eps) * xPhys^(penal-1) * ce
                dc_loc = penal * (1.0 - eps) * (xPhys ** (penal - 1.0)) * ce

                # Filter sensitivities back to design x:
                dc_vec = dc_loc.ravel()
                dc_filtered_vec = H.T @ (dc_vec / Hs)
                dc_i[:, :, kdeg] = dc_filtered_vec.reshape(nely, nelx)

            # Objective gradient: df = -mean over DOCs of (alpha_i * dE_i/dx)
            df0dx = np.zeros((nely, nelx), dtype=float)
            if ldoc > 0:
                ii_doc = np.where(mdoc)[0]
                A = alpha_scale[ii_doc]
                dstack = dc_i[:, :, ii_doc]  # (nely, nelx, ldoc)
                df0dx = -np.mean(A[None, None, :] * dstack, axis=2)

            # Constraints: dg_j ≈ (1/emax_j) * dE_dof_j/dx  (ignoring dW_th/dx)
            dgdx = []
            if ldof > 0:
                jj = np.where(mdof)[0]
                for j, ej in enumerate(jj):
                    dgdx.append(dc_i[:, :, ej] / emax[j])
            dgdx = np.stack(dgdx, axis=0) if ldof > 0 else np.zeros((0, nely, nelx))

            # reshape for MMA: (n,) vectors/rows
            df0dx_vec = df0dx.ravel()[:, None]
            dfdx_mat = gscale * dgdx.reshape(ldof, -1)  # (m, n)

            print(f"iter {it}: max |fval| = {np.max(np.abs(fval)):.3e}")
            print(f"iter {it}: min(upp - x) = {np.min(upp_vec - xval[:, 0]):.3e}, "
                  f"min(x - low) = {np.min(xval[:, 0] - low_vec):.3e}")
            print(f"iter {it}: max |df0dx| = {np.max(np.abs(df0dx_vec)):.3e}, "
                  f"max |dfdx| = {np.max(np.abs(dfdx_mat)):.3e}")
            # time.sleep(2)

            # --- MMA update ---
            xmma, low_vec, upp_vec = mmasub(
                MMAInputs(
                    m=ldof,
                    n=n,
                    iterr=it,
                    xval=xval[:, 0],
                    xmin=xmin,
                    xmax=xmax,
                    xold1=xold1,
                    xold2=xold2,
                    df0dx=df0dx_vec[:, 0],
                    fval=fval,          # shape (m,)
                    dfdx=dfdx_mat,      # shape (m, n)
                    low=low_vec,
                    upp=upp_vec,
                    a0=a0,
                    a=np.zeros((ldof,)),  # per MMAInputs signature
                    c=(10000.0 * np.ones((ldof,))),
                    d=np.zeros((ldof,)),
                    f0val=f0val,
                )
            )
            low_vec = np.squeeze(low_vec)
            upp_vec = np.squeeze(upp_vec)

            # Update history and state
            if it > SECOND_ITERATION_THRESHOLD:
                xold2 = xold1
                xold1 = xval
            elif it > FIRST_ITERATION_THRESHOLD:
                xold1 = xval

            change = float(np.max(np.abs(xmma - xval[:, 0])))
            x = xmma.reshape(nely, nelx)
            energies_hist.append(energies.copy())

            plot_design(x, open_plot=True)

            ttot = time.time() - t0
            print(
                f" it {it:3d}  obj: {-f0val:8.3e}  "
                f"E[tx,ty,rz]=[{energies[0]:.3e},{energies[1]:.3e},{energies[2]:.3e}]  "
                f"Wth[tx,ty,rz]=[{W_th[0]:.3e},{W_th[1]:.3e},{W_th[2]:.3e}]  "
                f"g={fval}  ch={change:6.3f}  t={ttot:5.2f}s"
            )

            if it > MAX_ITERATIONS:
                break

        return {
            "design": x,
            "energies": energies_hist[-1] if energies_hist else None,
            "thermal_work": W_th,
            "objective": f0val,
            "constraints": fval,
            "energies_history": np.array(energies_hist) if energies_hist else None,
        }


if __name__ == "__main__":
    # Example: stiff in ty, compliant in tx under thermal actuation
    nelx, nely = 64, 64
    model = FlexureThermoModel(plot=False)
    bcs = {
        "nelx": nelx,
        "nely": nely,
        "doc": ["ty"],         # stiff in tx
        "dof": ["tx"],         # thermal actuation in ty; energy balance constraint
        "emax": [1.0],         # scaling for E - W_th constraint
        "rmin": 2.0,
        "penal": 3.0,
        "volfrac_init": 0.2,
        "E0": 1.0,
        "k0": 1.0,
        "alpha": 1.0,
        "q": 1.0,              # uniform heat generation
    }
    out = model.run(bcs)
    print("Final energies:", out["energies"])
    print("Final thermal work:", out["thermal_work"])
