from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from D3.utils.linear_solver import solve_spd_with_amg



@dataclass
class FEMthmBCResult3D:
    """All outputs needed downstream for 3D thermo-elastic TO."""
    km: csr_matrix                 # Global mechanical stiffness (ndofm x ndofm)
    kth: csr_matrix                # Global thermal conductivity (nn x nn)
    um: np.ndarray                 # Mechanical displacements (ndofm,)
    uth: np.ndarray                # Temperatures (nn,)
    fm: np.ndarray                 # Total mechanical RHS (including thermal)
    fth: np.ndarray                # Thermal RHS after BCs
    d_cthm: coo_matrix             # Global thermo-mech coupling derivative (ndofm x nn)
    fixeddofsm: np.ndarray         # Fixed mechanical DOFs
    alldofsm: np.ndarray           # All mechanical DOFs
    freedofsm: np.ndarray          # Free mechanical DOFs
    fixeddofsth: np.ndarray        # Fixed thermal DOFs
    alldofsth: np.ndarray          # All thermal DOFs
    freedofsth: np.ndarray         # Free thermal DOFs
    fp: np.ndarray                 # External mechanical load vector (ndofm,)


def fe_mthm_bc_3d(  # noqa: PLR0915, PLR0913
    nely: int,
    nelx: int,
    nelz: int,
    penal: float,
    x: np.ndarray,                # shape: (nely, nelx, nelz)
    ke: np.ndarray,               # (24, 24)  from fe_melthm_3d
    k_eth: np.ndarray,            # (8, 8)    from fe_melthm_3d
    c_ethm: np.ndarray,           # (24, 8)   from fe_melthm_3d
    tref: float,
    bcs: dict[str, Any],
) -> FEMthmBCResult3D:
    """
    Assemble and solve the 3D coupled thermo-elastic FE systems for topology optimization.
    Uses Hex8 elements, 1 thermal DOF/node, 3 mechanical DOF/node.
    """
    # ---------------------------
    # Helpers
    # ---------------------------
    def node_index(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray) -> np.ndarray:
        """Map (ix,iy,iz) to flat node id with z fastest, then y, then x."""
        return (nely + 1) * (nelz + 1) * ix + (nelz + 1) * iy + iz

    def mask_to_indices(mask3d: np.ndarray) -> np.ndarray:
        """Flatten a boolean node mask (nelx+1, nely+1, nelz+1) to node ids."""
        m = np.asarray(mask3d, dtype=bool)
        return np.flatnonzero(m.ravel(order="C")).astype(int)

    # Domain weighting
    weight = bcs.get("weight", 0.5)

    # ---------------------------
    # Dimensions & DOF counts
    # ---------------------------
    nn = (nelx + 1) * (nely + 1) * (nelz + 1)        # thermal DOFs (nodes)
    ndofsm = 3 * nn                                   # mechanical DOFs (3 per node)

    # ---------------------------
    # THERMAL: BCs (Dirichlet sinks)
    # ---------------------------
    alldofsth = np.arange(nn)
    fixeddofsth = mask_to_indices(bcs["heatsink_elements"])
    freedofsth = np.setdiff1d(alldofsth, fixeddofsth, assume_unique=True)

    # ---------------------------
    # Build element connectivity (once)
    # ---------------------------
    # element indices
    ex, ey, ez = np.meshgrid(
        np.arange(nelx), np.arange(nely), np.arange(nelz), indexing="ij"
    )
    ex = ex.ravel()
    ey = ey.ravel()
    ez = ez.ravel()
    nelem = ex.size

    # Corner nodes in Hex8 order: (-,-,-),(+,-,-),(+,+,-),(-,+,-),(-,-,+),(+,-,+),(+,+,+),(-,+,+)
    n000 = node_index(ex,     ey,     ez)
    n100 = node_index(ex + 1, ey,     ez)
    n110 = node_index(ex + 1, ey + 1, ez)
    n010 = node_index(ex,     ey + 1, ez)
    n001 = node_index(ex,     ey,     ez + 1)
    n101 = node_index(ex + 1, ey,     ez + 1)
    n111 = node_index(ex + 1, ey + 1, ez + 1)
    n011 = node_index(ex,     ey + 1, ez + 1)

    # Thermal edof (nelem, 8)
    edof8 = np.stack([n000, n100, n110, n010, n001, n101, n111, n011], axis=1)

    # Mechanical edof (nelem, 24): for each node append [3n,3n+1,3n+2]
    edof24 = np.stack(
        [3 * edof8 + 0, 3 * edof8 + 1, 3 * edof8 + 2], axis=2
    ).reshape(nelem, 24)

    # Penalized factor per element (shape matches your x layout)
    # x is indexed as x[ey, ex, ez] to respect (nely, nelx, nelz)
    penalized = (x[ey, ex, ez] ** penal).astype(np.float64)

    # ---------------------------
    # THERMAL: Assemble Kth
    # ---------------------------
    # For each element, contribute penalized * k_eth into node-pair positions
    kth_row = np.repeat(edof8, 8, axis=1)        # (nelem, 64)
    kth_col = np.tile(edof8, 8)                  # (nelem, 64)
    kth_blk = penalized[:, None, None] * k_eth   # (nelem, 8, 8)
    kth_dat = kth_blk.reshape(nelem, 64).ravel()

    kth = coo_matrix((kth_dat, (kth_row.ravel(), kth_col.ravel())), shape=(nn, nn))
    kth = (kth + kth.T) / 2.0
    kth = kth.tolil()

    # Thermal RHS and Dirichlet sinks
    fth = np.ones(nn) * tref
    tsink = 0.0
    for d in fixeddofsth:
        kth.rows[d] = [d]
        kth.data[d] = [1.0]
        fth[d] = tsink

    # TODO: solver change here
    # kth = kth.tocsr()
    # uth = spsolve(kth, fth)
    uth = solve_spd_with_amg(kth.tocsr(), fth)

    # ---------------------------
    # MECHANICAL: BCs (clamped supports)
    # ---------------------------
    fix_nodes = mask_to_indices(bcs["fixed_elements"])
    fixeddofsm_x = 3 * fix_nodes + 0
    fixeddofsm_y = 3 * fix_nodes + 1
    fixeddofsm_z = 3 * fix_nodes + 2
    fixeddofsm = np.concatenate((fixeddofsm_x, fixeddofsm_y, fixeddofsm_z))
    alldofsm = np.arange(ndofsm)
    freedofsm = np.setdiff1d(alldofsm, fixeddofsm, assume_unique=True)

    # ---------------------------
    # MECHANICAL: Assemble Km and d_cthm
    # ---------------------------
    # Stiffness
    km_row = np.repeat(edof24, 24, axis=1)             # (nelem, 576)
    km_col = np.tile(edof24, 24)                       # (nelem, 576)
    km_blk = penalized[:, None, None] * ke             # (nelem, 24, 24)
    km_dat = km_blk.reshape(nelem, 576).ravel()
    km = coo_matrix((km_dat, (km_row.ravel(), km_col.ravel())), shape=(ndofsm, ndofsm))

    # Coupling derivative (maps θ to mechanical forces)
    dC_row = np.repeat(edof24, 8, axis=1)              # (nelem, 24*8)
    dC_col = np.tile(edof8, 24)                        # (nelem, 24*8)
    dC_blk = penalized[:, None, None] * c_ethm         # (nelem, 24, 8)
    dC_dat = dC_blk.reshape(nelem, 24 * 8).ravel()
    d_cthm = coo_matrix((dC_dat, (dC_row.ravel(), dC_col.ravel())), shape=(ndofsm, nn))

    # Symmetrize mechanical stiffness and convert
    km = (km + km.T) / 2.0
    km = km.tocsr()

    # ---------------------------
    # THERMO-MECHANICAL LOAD (equivalent forces)
    # ---------------------------
    # For each element: diff_e = uth[edof8] - tref  (nelem, 8)
    diff = uth[edof8] - tref
    # thermal_e = penalized * (c_ethm @ diff_e) -> (nelem, 24)
    thermal_e = (c_ethm @ diff.T).T                   # (nelem, 24)
    thermal_e *= penalized[:, None]

    # Accumulate to global vector via bincount
    feps = np.bincount(edof24.ravel(), weights=thermal_e.ravel(), minlength=ndofsm).astype(np.float64)

    # ---------------------------
    # EXTERNAL MECHANICAL LOADS
    # ---------------------------
    fp = np.zeros(ndofsm, dtype=np.float64)

    def add_load(mask_key: str, comp: int, mag: float = 0.5) -> None:
        if mask_key in bcs and bcs[mask_key] is not None:
            nodes = mask_to_indices(bcs[mask_key])
            dofs = 3 * nodes + comp
            fp[dofs] += mag

    add_load("force_elements_x", 0)
    add_load("force_elements_y", 1)
    add_load("force_elements_z", 2)

    # Choose RHS by weight (pure structural / pure thermal / mixed)
    if weight == 1.0:
        fm = fp
    elif weight == 0.0:
        fm = feps
    else:
        fm = fp + feps

    # ---------------------------
    # SOLVES
    # ---------------------------
    um = np.zeros(ndofsm, dtype=np.float64)
    if freedofsm.size > 0:
        # TODO: solver change here
        # um[freedofsm] = spsolve(km[freedofsm, :][:, freedofsm], fm[freedofsm])

        # # Save numpy arrays once to benchmark linear solvers
        # arr_A = km[freedofsm, :][:, freedofsm].tocsr()  # csr_matrix
        # arr_b = fm[freedofsm]  # ndarray
        # from scipy.sparse import save_npz, load_npz
        # save_path = '/Users/gapaza/repos/ideal/structural-thermal-3d/matrix/store/A.npz'
        # save_npz(save_path, arr_A)
        # save_path_b = '/Users/gapaza/repos/ideal/structural-thermal-3d/matrix/store/b.npy'
        # np.save(save_path_b, arr_b)
        # exit(0)

        # f = K u

        um_free = solve_spd_with_amg(km[freedofsm, :][:, freedofsm].tocsr(), fm[freedofsm])
        um = np.zeros(ndofsm)
        um[freedofsm] = um_free
    if fixeddofsm.size > 0:
        um[fixeddofsm] = 0.0

    return FEMthmBCResult3D(
        km=km,
        kth=kth,
        um=um,
        uth=uth,
        fm=fm,
        fth=fth,
        d_cthm=d_cthm,
        fixeddofsm=fixeddofsm,
        alldofsm=alldofsm,
        freedofsm=freedofsm,
        fixeddofsth=fixeddofsth,
        alldofsth=alldofsth,
        freedofsth=freedofsth,
        fp=fp,
    )


def fe_mthm_bc_3d_abacus(  # noqa: PLR0915, PLR0913
    nely: int,
    nelx: int,
    nelz: int,
    penal: float,
    x: np.ndarray,                # shape: (nely, nelx, nelz)
    ke: np.ndarray,               # (24, 24)  from fe_melthm_3d
    k_eth: np.ndarray,            # (8, 8)    from fe_melthm_3d
    c_ethm: np.ndarray,           # (24, 8)   from fe_melthm_3d
    tref: float,
    bcs: dict[str, Any],
) -> FEMthmBCResult3D:
    """
    Assemble and solve the 3D coupled thermo-elastic FE systems for topology optimization.
    Uses Hex8 elements, 1 thermal DOF/node, 3 mechanical DOF/node.
    """
    return FEMthmBCResult3D()




