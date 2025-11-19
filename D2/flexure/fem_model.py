# flexure_opt.py

from math import ceil, hypot
import time
from typing import Any, Sequence

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Reuse your element stiffness builder; we only need 'ke'
from D2.thermoelastic.fem_matrix_builder import fe_melthm
from D2.thermoelastic.mma_subroutine import MMAInputs, mmasub
from D2.flexure.utils import plot_design

FIRST_ITERATION_THRESHOLD = 1
SECOND_ITERATION_THRESHOLD = 2
MIN_ITERATIONS = 10
MAX_ITERATIONS = 200
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


def edof_mappings(nelx: int, nely: int) -> tuple[np.ndarray, np.ndarray]:
    """Element-to-DOF maps (edof8 for mechanics) vectorized over all elements."""
    # Element indices on a meshgrid
    ex, ey = np.meshgrid(np.arange(nelx), np.arange(nely), indexing="ij")
    ex = ex.ravel()
    ey = ey.ravel()
    # Node numbers (0-based)
    n1 = (nely + 1) * ex + ey        # Maps element to bottom left node global index
    n2 = (nely + 1) * (ex + 1) + ey  # Maps element to bottom right node global index
    # 4-node connectivity (thermal not used; just for reference)
    edof4 = np.stack([n1 + 1, n2 + 1, n2, n1], axis=1)
    # 8 mechanical dofs per element (ux,uy per node)
    edof8 = np.stack(
        [
            2 * n1 + 2, 2 * n1 + 3, # Node 1
            2 * n2 + 2, 2 * n2 + 3, # Node 2
            2 * n2, 2 * n2 + 1,     # Node 3
            2 * n1, 2 * n1 + 1      # Node 4
        ],
        axis=1,
    )  # Maps element to global DOF (8) for that element
    return edof8.astype(int), edof4.astype(int)


def prescribed_u_matrix(nelx: int, nely: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the prescribed displacement matrix U[:,3] for tx,ty,rz and fixed/free sets."""
    ndof = 2 * (nelx + 1) * (nely + 1)

    # Node indexing on a rect grid:
    # y-index 0..nely, x-index 0..nelx
    def node_id(ix: int, iy: int) -> int:
        return (nely + 1) * ix + iy

    # Collect top and bottom node indices
    bot_nodes = [node_id(ix, nely) for ix in range(nelx + 1)]
    top_nodes = [node_id(ix, 0) for ix in range(nelx + 1)]


    # DOF indices (0-based): ux=2*n, uy=2*n+1
    top_ux = np.array([2 * n for n in top_nodes], dtype=int)
    top_uy = top_ux + 1
    bot_ux = np.array([2 * n for n in bot_nodes], dtype=int)
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
    U[top_uy, 2] = np.linspace(1.0, -1.0, nelx + 1)  # center of rotation in the middle across x

    return U, fixed, free, top_nodes


class FlexureModel:
    """2D flexure synthesis via topology optimization (pure structural, prescribed displacements)."""

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
            - emax : float or list[float] (same length/order as dof)
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

        mdoc = to_mask(bcs["doc"]) # [ True False False]
        mdof = to_mask(bcs["dof"]) # [False  True False]
        assert not np.any(mdoc & mdof), "DOC and DOF sets must be disjoint"
        deg_mask = mdoc | mdof
        deg_idx = np.where(deg_mask)[0]  # active mechanism indices (0..2)
        ldoc = int(mdoc.sum())
        ldof = int(mdof.sum())
        assert 1 <= ldoc <= 2, "DOC must have 1 or 2 entries"
        assert 1 <= ldof <= 2, "DOF must have 1 or 2 entries"

        # emax handling (per DOF)
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
        E0 = 1.0
        nu = 0.3
        eps = 1e-9

        # Design & filter
        n = nelx * nely
        x = self.get_initial_design(volfrac_init, nelx, nely) if x_init is None else x_init.copy()
        H, Hs = build_filter(nelx, nely, rmin)

        # Element stiffness (we use only 'ke'; ignore thermal returns)
        ke, _, _ = fe_melthm(nu, E0, 1.0, 0.0)  # k, alpha unused

        # Global mappings
        edof8, _ = edof_mappings(nelx, nely)  # (4096, 8)
        ndof = 2 * (nelx + 1) * (nely + 1)
        U_presc, fixed, free, _ = prescribed_u_matrix(nelx, nely)

        # Pre-build global stiffness sparsity pattern indices
        n_elem = n
        # For vectorized assembly of K = Σ_e (E_e * ke)
        # K_e is an 8x8 matrix, so 64 values
        # Each of the 64 element-specific values are mapped to the global K, which is 50 by 50
        K_rows = np.repeat(edof8, 8, axis=1).ravel()  # (16, 64) before ravel, (1024,) after
        # print(K_rows.shape)
        # exit(0)
        K_cols = np.tile(edof8, 8).ravel()              # (16, 64) before ravel, (1024,) after
        KE_flat = ke.reshape(1, 64)

        # MMA setup (no volume constraint by default; only DOF energy constraints)
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
        alpha = np.ones(3, dtype=float)  # energy scaling per degree (set at iter 1)

        # Bookkeeping
        energies_hist: list[np.ndarray] = []

        while change > UPDATE_THRESHOLD or it < MIN_ITERATIONS:
            it += 1
            t0 = time.time()

            # --- Symmetry (optional): comment out if you don't want enforced symmetry
            # x = 0.5 * (x + np.flipud(x))
            # x = 0.5 * (x + np.fliplr(x))

            # --- Density filter (x -> xPhys)
            xval = x.reshape(n, 1)
            xPhys = (H @ xval) / Hs[:, None]
            xPhys = xPhys.reshape(nely, nelx)

            # --- SIMP interpolation (elasticity)
            Ee = eps + (1.0 - eps) * xPhys.ravel() ** penal  # per element scalar
            # Assemble K
            K_data = (Ee[:, None] * KE_flat).ravel()  # K_data is (16, 64) before ravel, local element stiffness matrix per-element
            K = coo_matrix((K_data, (K_rows, K_cols)), shape=(ndof, ndof)).tocsr()  # Form global stiffness matrix
            K = (K + K.T) * 0.5  # enforce symmetry

            # --- Solve for each active mechanism case (columns)
            U = np.zeros((ndof, 3), dtype=float)
            energies = np.zeros(3, dtype=float)
            for ii in deg_idx:
                # Kff * Uf = −Kfp * Up
                rhs = -K[np.ix_(free, fixed)] @ U_presc[fixed, ii]
                U_free = spsolve(K[np.ix_(free, free)], rhs)
                U[:, ii] = 0.0
                U[fixed, ii] = U_presc[fixed, ii]
                U[free, ii] = U_free
                # Energy E_i = 0.5 * U^T K U
                Ku = K @ U[:, ii]
                energies[ii] = 0.5 * float(U[:, ii].T @ Ku)

            # Set alpha on first iteration (and tweak rotation scaling like MATLAB)
            if it == 1:
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = np.zeros(3, dtype=float)
                    nz = energies > 0
                    alpha[nz] = 1.0 / energies[nz]
                    # Empirical doubling for rotation "rz" (index 2) as in MATLAB
                    alpha[2] *= 2.0

            # --- Objective and constraints
            # f = -mean(alpha * E) over DOCs
            f0val = -np.mean(alpha[mdoc] * energies[mdoc])

            # g_j = E_dof_j / emax_j - 1   (scaled later if desired)
            E_dof = energies[mdof]
            fval = (E_dof / emax) - 1.0  # shape (ldof,)

            if self.eval_only:
                return {
                    "design": x,
                    "energies": energies,
                    "objective": f0val,
                    "constraints": fval,
                }

            # --- Sensitivities (self-adjoint)
            # Elemental energies per mechanism: ce_i(e) = 0.5 * u_e^T ke u_e
            dc_i = np.zeros((nely, nelx, 3), dtype=float)
            for kdeg in deg_idx:
                Uk = U[:, kdeg]
                Ue = Uk[edof8]  # (n_elem, 8)

                # Element strain energy: ce_e = 0.5 * u_e^T ke u_e
                ce = 0.5 * np.einsum("ij,ij->i", Ue @ ke, Ue).reshape(nely, nelx)

                # Local sensitivity w.r.t. xPhys:
                # dE/dxPhys = penal * (1-eps) * xPhys^(penal-1) * ce
                dc_loc = penal * (1.0 - eps) * (xPhys ** (penal - 1.0)) * ce  # (nely, nelx)

                # Filter sensitivities back to design x:
                # dc_filtered = conv2( (dc_loc./Hs), h, 'same' )
                # In matrix form with H: xPhys = (H x)/Hs ⇒ df/dx ≈ H^T ( df/dxPhys ./ Hs ).
                dc_vec = dc_loc.ravel()
                dc_filtered_vec = H.T @ (dc_vec / Hs)
                dc_i[:, :, kdeg] = dc_filtered_vec.reshape(nely, nelx)

            # Combine to get df and dgi
            df0dx = np.zeros((nely, nelx), dtype=float)
            if ldoc > 0:
                # df = -mean over DOCs of (alpha_i * dE_i/dx)
                ii = np.where(mdoc)[0]
                A = alpha[ii]
                dstack = dc_i[:, :, ii]  # (nely, nelx, ldoc)
                df0dx = -np.mean(A[None, None, :] * dstack, axis=2)

            # Constraints: dg_j = (1/emax_j) * dE_dof_j/dx
            dgdx = []
            if ldof > 0:
                jj = np.where(mdof)[0]
                for j, ej in enumerate(jj):
                    dgdx.append(dc_i[:, :, ej] / emax[j])
            dgdx = np.stack(dgdx, axis=0) if ldof > 0 else np.zeros((0, nely, nelx))
            # reshape for MMA: (n,) vectors/rows
            df0dx_vec = df0dx.ravel()[:, None]
            dgdx_mat = dgdx.reshape(ldof, -1)  # (m, n)

            # --- MMA update
            xmma, low_vec, upp_vec = mmasub(
                MMAInputs(
                    m=ldof,
                    n=n,
                    iterr=it,
                    xval=xval[:, 0],
                    xmin=1e-3,
                    xmax=1.0,
                    xold1=xold1,
                    xold2=xold2,
                    df0dx=df0dx_vec[:, 0],
                    fval=fval,  # shape (m,)
                    dfdx=dgdx_mat,  # shape (m, n)
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

            if it % 5 == 0:
                plot_design(x, open_plot=True)

            ttot = time.time() - t0
            print(
                f" it {it:3d}  obj: {-f0val:8.3e}  "
                f"E[tx,ty,rz]=[{energies[0]:.3e},{energies[1]:.3e},{energies[2]:.3e}]  "
                f"g={fval}  ch={change:6.3f}  t={ttot:5.2f}s"
            )

            if it > MAX_ITERATIONS:
                break

        return {
            "design": x,
            "energies": energies_hist[-1] if energies_hist else None,
            "objective": f0val,
            "constraints": fval,
            "energies_history": np.array(energies_hist) if energies_hist else None,
        }


if __name__ == "__main__":
    # Example: stiff in ty, compliant in tx with E_tx <= 1.0
    nelx, nely = 64, 64
    model = FlexureModel(plot=False)
    bcs = {
        "nelx": nelx,
        "nely": nely,
        "doc": ["tx"],  # stiff vertically
        "dof": ["ty"],  # compliant horizontally
        "emax": [0.01],  # E_tx <= 1.0 Nm (unit disp ⇒ E=0.5k)
        "rmin": 2.0,
        "penal": 3.0,
        "volfrac_init": 0.2,
    }
    out = model.run(bcs)
    print("Final energies:", out["energies"])
