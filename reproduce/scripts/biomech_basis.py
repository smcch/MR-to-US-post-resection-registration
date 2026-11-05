#!/usr/bin/env python
"""
biomech_basis.py -- Per-case biomechanical modal basis for ReMIND brain-shift registration.

Builds a structured tetrahedral lattice restricted to the SynthSeg brain mask,
assigns per-element linear-elastic materials from SynthSeg labels (+ optional
tumor/cavity segmentation), assembles P1 linear-elasticity stiffness K and lumped
mass M, adds an elastic foundation on the mesh surface (approximate cranial
contact, removes rigid-body modes), and solves the generalized eigenproblem
K phi = lambda M phi for the n_modes lowest-energy modes.

Unit convention (SI internally):
  - node coordinates are stored/saved in WORLD mm, but converted to meters for
    the FEM assembly; E in Pa; density kg/m^3.
  - eigenvalues lambda = omega^2 in (rad/s)^2; modes are M-orthonormal
    (phi^T M phi = I), i.e. units 1/sqrt(kg). Shapes are what matter downstream.

CPU only: numpy/scipy/nibabel. No torch, no TensorFlow, no GPU.

Usage:
  python biomech_basis.py --synthseg seg.nii.gz [--tumor tum.nii.gz] \
      --out basis.npz --h 4.0 --n-modes 30 [--qa-dir qa/]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import nibabel as nib
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------------------
# Label sets (FreeSurfer / SynthSeg)
# ----------------------------------------------------------------------------
LABELS_CSF = frozenset({4, 5, 14, 15, 43, 44, 24})  # ventricles + CSF
LABELS_PARENCHYMA = frozenset({2, 3, 7, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28,
                               41, 42, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60})

# Materials (E in Pa, nu dimensionless)
MAT_CSF = (100.0, 0.49)          # ventricles / CSF: quasi-fluid
MAT_PARENCHYMA = (3000.0, 0.45)  # brain parenchyma / deep GM / cerebellum / brainstem
MAT_TUMOR = (6000.0, 0.45)       # tumor (2x parenchyma, elastography prior)
MAT_CAVITY = (300.0, 0.40)       # resection cavity: degraded material
RHO = 1000.0                     # kg/m^3, all tissues

# Tumor-seg override codes
TUM_NONE, TUM_TUMOR, TUM_CAVITY = 0, 1, 2

# ----------------------------------------------------------------------------
# 5-tet cube decompositions with alternating parity (conforming shared faces).
# Cube corner flat index c -> offset (c&1, (c>>1)&1, (c>>2)&1).
# Parity (i+j+k) even -> TET_EVEN (central tet on even-sum corners 0,3,5,6);
# odd -> TET_ODD (central tet on odd-sum corners 1,2,4,7). Face diagonals of the
# two schemes are mirrored, so neighbouring cells share conforming faces.
# ----------------------------------------------------------------------------
TET_EVEN = np.array([[0, 3, 5, 6],
                     [1, 0, 3, 5],
                     [2, 0, 6, 3],
                     [4, 0, 5, 6],
                     [7, 3, 6, 5]], dtype=np.int64)
TET_ODD = np.array([[1, 2, 4, 7],
                    [0, 1, 2, 4],
                    [3, 1, 2, 7],
                    [5, 1, 4, 7],
                    [6, 2, 4, 7]], dtype=np.int64)
CORNER_OFFSETS = np.array([[c & 1, (c >> 1) & 1, (c >> 2) & 1] for c in range(8)],
                          dtype=np.int64)  # (8,3)


def log(msg):
    print(msg, flush=True)


# ----------------------------------------------------------------------------
# Lattice construction
# ----------------------------------------------------------------------------
def build_lattice(mask, affine, h):
    """Regular world-axis-aligned lattice of spacing h (mm). Keep cubic cells
    whose center lies inside the brain mask dilated by one cell (radius h, via
    Euclidean distance transform in mm). Returns nodes (world mm), tets,
    and grid metadata."""
    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    dist_mm = ndi.distance_transform_edt(~mask, sampling=zooms)

    # World bounding box of the mask, padded by 2 cells
    ijk = np.array(np.nonzero(mask))
    lo, hi = ijk.min(axis=1), ijk.max(axis=1)
    corners = np.array([[x, y, z, 1.0]
                        for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])], dtype=float)
    wcorners = (affine @ corners.T).T[:, :3]
    wmin = wcorners.min(axis=0) - 2.0 * h
    wmax = wcorners.max(axis=0) + 2.0 * h
    origin = wmin
    n_cells = np.ceil((wmax - origin) / h).astype(int)  # cells per axis

    # Cell centers -> voxel space (nearest neighbor)
    ci, cj, ck = np.meshgrid(np.arange(n_cells[0]), np.arange(n_cells[1]),
                             np.arange(n_cells[2]), indexing='ij')
    centers = origin + (np.stack([ci, cj, ck], axis=-1).reshape(-1, 3) + 0.5) * h
    inv = np.linalg.inv(affine)
    vox = (inv[:3, :3] @ centers.T).T + inv[:3, 3]
    vox = np.rint(vox).astype(int)
    inside = np.all((vox >= 0) & (vox < np.array(mask.shape)), axis=1)
    keep_flat = np.zeros(len(vox), dtype=bool)
    vi = vox[inside]
    keep_flat[inside] = dist_mm[vi[:, 0], vi[:, 1], vi[:, 2]] <= h
    keep = keep_flat.reshape(tuple(n_cells))

    kc = np.array(np.nonzero(keep)).T  # (n_kept, 3)
    n_kept = len(kc)
    if n_kept == 0:
        raise RuntimeError("No lattice cells inside the brain mask.")

    # Node grid: corners of kept cells
    ng_shape = tuple(n_cells + 1)
    used = np.zeros(ng_shape, dtype=bool)
    for off in CORNER_OFFSETS:
        used[kc[:, 0] + off[0], kc[:, 1] + off[1], kc[:, 2] + off[2]] = True
    node_id = np.full(ng_shape, -1, dtype=np.int64)
    n_nodes = int(used.sum())
    node_id[used] = np.arange(n_nodes)
    node_ijk = np.argwhere(used)                       # (n_nodes,3), C order == arange order
    nodes_mm = origin + node_ijk * h                   # world mm

    # Per-kept-cell global corner node ids (n_kept, 8)
    cell_corner_nodes = np.empty((n_kept, 8), dtype=np.int64)
    for c in range(8):
        off = CORNER_OFFSETS[c]
        cell_corner_nodes[:, c] = node_id[kc[:, 0] + off[0], kc[:, 1] + off[1],
                                          kc[:, 2] + off[2]]

    # Tetrahedra with alternating parity
    parity = (kc.sum(axis=1) % 2).astype(bool)
    tet_list = []
    for par, scheme in ((False, TET_EVEN), (True, TET_ODD)):
        cells = cell_corner_nodes[parity == par]       # (m, 8)
        if len(cells) == 0:
            continue
        t = cells[:, scheme.ravel()].reshape(-1, 5, 4).reshape(-1, 4)
        tet_list.append(t)
    tets = np.concatenate(tet_list, axis=0).astype(np.int64)

    # Node -> number of incident kept cells (interior nodes have 8)
    cell_count = np.bincount(cell_corner_nodes.ravel(), minlength=n_nodes)

    meta = dict(origin_mm=origin, h_mm=h, n_cells=n_cells,
                node_grid_shape=np.array(ng_shape), keep_cells=keep,
                node_ijk=node_ijk, node_cell_count=cell_count)
    return nodes_mm, tets, meta


def fix_orientation(nodes_mm, tets):
    """Ensure all tets have positive signed volume (swap two nodes if not)."""
    x = nodes_mm[tets]
    v6 = np.einsum('ij,ij->i', x[:, 1] - x[:, 0],
                   np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
    neg = v6 < 0
    if neg.any():
        tets[neg, 0], tets[neg, 1] = tets[neg, 1].copy(), tets[neg, 0].copy()
    x = nodes_mm[tets]
    v6 = np.einsum('ij,ij->i', x[:, 1] - x[:, 0],
                   np.cross(x[:, 2] - x[:, 0], x[:, 3] - x[:, 0]))
    if (v6 <= 0).any():
        raise RuntimeError("Degenerate tetrahedra after orientation fix.")
    return tets


# ----------------------------------------------------------------------------
# Materials
# ----------------------------------------------------------------------------
def sample_nn(points_mm, img_data, affine):
    """Nearest-neighbor sample of a labelmap at world-mm points. Outside FOV -> 0."""
    inv = np.linalg.inv(affine)
    vox = (inv[:3, :3] @ points_mm.T).T + inv[:3, 3]
    vox = np.rint(vox).astype(int)
    inside = np.all((vox >= 0) & (vox < np.array(img_data.shape)), axis=1)
    out = np.zeros(len(points_mm), dtype=img_data.dtype)
    v = vox[inside]
    out[inside] = img_data[v[:, 0], v[:, 1], v[:, 2]]
    return out


def assign_materials(nodes_mm, tets, seg_data, seg_affine, tumor_data, tumor_affine):
    """Per-tet E, nu from the SynthSeg label at the tet centroid; optional
    tumor(1)/cavity(2) override. Background/unknown labels (shell cells from the
    one-cell dilation) get parenchyma properties."""
    centroids = nodes_mm[tets].mean(axis=1)
    labels = sample_nn(centroids, seg_data, seg_affine).astype(np.int32)

    E = np.full(len(tets), MAT_PARENCHYMA[0])
    nu = np.full(len(tets), MAT_PARENCHYMA[1])
    is_csf = np.isin(labels, list(LABELS_CSF))
    E[is_csf], nu[is_csf] = MAT_CSF
    # parenchyma + background/unknown already at parenchyma defaults

    known = np.isin(labels, list(LABELS_PARENCHYMA)) | is_csf
    n_bg = int((~known).sum())

    tum = np.zeros(len(tets), dtype=np.int8)
    if tumor_data is not None:
        t = sample_nn(centroids, tumor_data, tumor_affine).astype(np.int32)
        tum[t == 1] = TUM_TUMOR
        tum[t == 2] = TUM_CAVITY
        E[tum == TUM_TUMOR], nu[tum == TUM_TUMOR] = MAT_TUMOR
        E[tum == TUM_CAVITY], nu[tum == TUM_CAVITY] = MAT_CAVITY

    counts = dict(csf=int(is_csf.sum()),
                  parenchyma=int((known & ~is_csf).sum()),
                  background_as_parenchyma=n_bg,
                  tumor=int((tum == TUM_TUMOR).sum()),
                  cavity=int((tum == TUM_CAVITY).sum()))
    return E, nu, labels, tum, counts


# ----------------------------------------------------------------------------
# FEM assembly (P1 tets, linear elasticity, vectorized)
# ----------------------------------------------------------------------------
def assemble_K_M(nodes_m, tets, E, nu, rho=RHO, chunk=200000):
    """Assemble sparse stiffness K (SI: N/m) and lumped-mass diagonal (kg per DOF)."""
    n_nodes = len(nodes_m)
    ndof = 3 * n_nodes
    n_tets = len(tets)

    rows_all, cols_all, vals_all = [], [], []
    mass_nodes = np.zeros(n_nodes)
    vol_total = 0.0

    for s in range(0, n_tets, chunk):
        sl = slice(s, min(s + chunk, n_tets))
        t = tets[sl]
        x = nodes_m[t]                                  # (n,4,3)
        n = len(t)
        C = np.ones((n, 4, 4))
        C[:, :, 1:] = x
        detC = np.linalg.det(C)
        V = np.abs(detC) / 6.0                          # m^3
        grads = np.linalg.inv(C)[:, 1:4, :]             # (n,3,4): grads[:,:,i] = grad N_i

        B = np.zeros((n, 6, 12))
        for i in range(4):
            gx, gy, gz = grads[:, 0, i], grads[:, 1, i], grads[:, 2, i]
            B[:, 0, 3 * i + 0] = gx
            B[:, 1, 3 * i + 1] = gy
            B[:, 2, 3 * i + 2] = gz
            B[:, 3, 3 * i + 0] = gy
            B[:, 3, 3 * i + 1] = gx
            B[:, 4, 3 * i + 1] = gz
            B[:, 4, 3 * i + 2] = gy
            B[:, 5, 3 * i + 0] = gz
            B[:, 5, 3 * i + 2] = gx

        e, v = E[sl], nu[sl]
        lam = e * v / ((1.0 + v) * (1.0 - 2.0 * v))
        mu = e / (2.0 * (1.0 + v))
        D = np.zeros((n, 6, 6))
        D[:, :3, :3] = lam[:, None, None]
        idx = np.arange(3)
        D[:, idx, idx] += 2.0 * mu[:, None]
        idx = np.arange(3, 6)
        D[:, idx, idx] = mu[:, None]

        DB = np.einsum('nij,njk->nik', D, B)
        Ke = np.einsum('nki,nkj->nij', B, DB) * V[:, None, None]

        edof = (3 * t[:, [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]]
                + np.tile([0, 1, 2], 4)[None, :]).astype(np.int64)  # (n,12)
        rows_all.append(np.broadcast_to(edof[:, :, None], (n, 12, 12)).ravel())
        cols_all.append(np.broadcast_to(edof[:, None, :], (n, 12, 12)).ravel())
        vals_all.append(Ke.ravel())

        np.add.at(mass_nodes, t.ravel(), np.repeat(rho * V / 4.0, 4))
        vol_total += V.sum()

    K = sp.coo_matrix((np.concatenate(vals_all),
                       (np.concatenate(rows_all), np.concatenate(cols_all))),
                      shape=(ndof, ndof)).tocsr()
    mass_dof = np.repeat(mass_nodes, 3)
    return K, mass_dof, vol_total


def add_foundation(K, node_cell_count, E_tet, tets, n_nodes, frac):
    """Diagonal springs on surface nodes (< 8 incident kept cells).
    k_found = frac * median diagonal K entry over parenchyma-adjacent DOFs."""
    diag = K.diagonal()
    par_tets = tets[E_tet == MAT_PARENCHYMA[0]]
    par_nodes = np.unique(par_tets.ravel())
    par_dofs = (3 * par_nodes[:, None] + np.arange(3)[None, :]).ravel()
    k_found = frac * float(np.median(diag[par_dofs]))

    surf_nodes = np.nonzero(node_cell_count < 8)[0]
    surf_dofs = (3 * surf_nodes[:, None] + np.arange(3)[None, :]).ravel()
    d = np.zeros(K.shape[0])
    d[surf_dofs] = k_found
    K = (K + sp.diags(d)).tocsr()
    return K, k_found, surf_nodes


# ----------------------------------------------------------------------------
# Eigensolve
# ----------------------------------------------------------------------------
def solve_modes(K, mass_dof, n_modes, solver='auto'):
    M = sp.diags(mass_dof).tocsr()
    used = None
    if solver in ('auto', 'shift-invert'):
        try:
            # Explicit factorization with a symmetric-pattern fill-reducing
            # ordering: much less fill than the COLAMD default eigsh would use.
            t0 = time.time()
            lu = spla.splu(K.tocsc(), permc_spec='MMD_AT_PLUS_A',
                           options=dict(SymmetricMode=True))
            log(f"[solve] splu(MMD_AT_PLUS_A, SymmetricMode) factorized in "
                f"{time.time() - t0:.1f} s, fill nnz(L+U)={lu.nnz}")
            OPinv = spla.LinearOperator(K.shape, matvec=lu.solve)
            vals, vecs = spla.eigsh(K, k=n_modes, M=M, sigma=0.0, which='LM',
                                    OPinv=OPinv)
            used = 'shift-invert (ARPACK, sigma=0, splu MMD_AT_PLUS_A)'
        except Exception as exc:  # MemoryError, ArpackError, ...
            if solver == 'shift-invert':
                raise
            log(f"[solve] shift-invert failed ({type(exc).__name__}: {exc}); "
                f"falling back to LOBPCG")
    if used is None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((K.shape[0], n_modes))
        dinv = 1.0 / K.diagonal()
        prec = spla.LinearOperator(K.shape, matvec=lambda x: dinv * x)
        vals, vecs = spla.lobpcg(K, X, B=M, M=prec, largest=False,
                                 tol=1e-8, maxiter=2000)
        used = 'LOBPCG (diagonal preconditioner)'

    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    # Enforce exact M-orthonormalization (cheap; guards LOBPCG round-off)
    nrm = np.sqrt(np.einsum('ij,ij->j', vecs, mass_dof[:, None] * vecs))
    vecs = vecs / nrm[None, :]
    return vals, vecs, used


# ----------------------------------------------------------------------------
# QA
# ----------------------------------------------------------------------------
def qa_checks(K, mass_dof, vals, vecs):
    res = {}
    d = (K - K.T).tocoo()
    kmax = np.abs(K.data).max()
    res['K_sym_rel_err'] = float(np.abs(d.data).max() / kmax) if d.nnz else 0.0
    res['K_sym_ok'] = res['K_sym_rel_err'] < 1e-8
    res['eigvals_positive'] = bool(np.all(vals > 0))
    res['eigvals_monotonic'] = bool(np.all(np.diff(vals) >= -1e-12 * vals[-1]))
    G = vecs.T @ (mass_dof[:, None] * vecs)
    off = G - np.diag(np.diag(G))
    res['M_orthonormal_max_offdiag'] = float(np.abs(off).max())
    res['M_orthonormal_max_diag_dev'] = float(np.abs(np.diag(G) - 1.0).max())
    res['M_orthonormal_ok'] = res['M_orthonormal_max_offdiag'] < 1e-6
    res['no_nan'] = bool(np.isfinite(vals).all() and np.isfinite(vecs).all())
    return res


def export_mode_niftis(qa_dir, mode_indices, vecs, n_nodes, meta, seg_img, E_tet=None,
                       tets=None):
    """Sample selected modes (displacement magnitude, trilinear from lattice
    nodes, zero outside mesh) onto the SynthSeg grid; also an E map (nearest cell)."""
    shape = seg_img.shape
    affine = seg_img.affine
    origin = meta['origin_mm']
    h = meta['h_mm']
    keep = meta['keep_cells']
    ng = tuple(meta['node_grid_shape'])
    node_ijk = meta['node_ijk']

    # Precompute, for every voxel, lattice coords + inside-mesh mask (chunked in z)
    n_cells = keep.shape
    written = []

    # Per-component node grids for one mode at a time
    def node_grid(comp_vals):
        g = np.zeros(ng, dtype=np.float32)
        g[node_ijk[:, 0], node_ijk[:, 1], node_ijk[:, 2]] = comp_vals
        return g

    # Optional E map on lattice cells (mean of the 5 tets of each kept cell)
    e_cell = None
    if E_tet is not None:
        e_cell = np.zeros(n_cells, dtype=np.float32)
        kc = np.argwhere(keep)
        # tets were built parity-grouped from kept cells; recompute per-cell E by
        # sampling cell centers is simpler & equivalent for QA -> use tet centroid trick:
        # take the E of the first tet of each cell in build order.
        par = (kc.sum(axis=1) % 2).astype(bool)
        order = np.concatenate([np.nonzero(~par)[0], np.nonzero(par)[0]])
        e_cell[kc[order, 0], kc[order, 1], kc[order, 2]] = \
            E_tet.reshape(-1, 5).mean(axis=1).astype(np.float32)

    out_arrays = {f"mode_{m + 1:03d}_magnitude": np.zeros(shape, dtype=np.float32)
                  for m in mode_indices}
    if e_cell is not None:
        out_arrays['material_E_Pa'] = np.zeros(shape, dtype=np.float32)

    modes_nodes = {m: vecs[:, m].reshape(n_nodes, 3) for m in mode_indices}
    grids = {m: [node_grid(modes_nodes[m][:, c]) for c in range(3)]
             for m in mode_indices}

    zchunk = 16
    for z0 in range(0, shape[2], zchunk):
        z1 = min(z0 + zchunk, shape[2])
        ii, jj, kk = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]),
                                 np.arange(z0, z1), indexing='ij')
        vox = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3).astype(float)
        world = (affine[:3, :3] @ vox.T).T + affine[:3, 3]
        L = (world - origin) / h                       # lattice (node-grid) coords
        cell = np.floor(L).astype(int)
        ok = np.all((cell >= 0) & (cell < np.array(n_cells)), axis=1)
        okc = cell[ok]
        inside = np.zeros(len(vox), dtype=bool)
        inside[ok] = keep[okc[:, 0], okc[:, 1], okc[:, 2]]

        if not inside.any():
            continue
        Li = L[inside]
        i0 = np.floor(Li).astype(int)
        f = (Li - i0).astype(np.float32)
        i0 = np.clip(i0, 0, np.array(ng) - 2)
        w = []
        for c in range(8):
            o = CORNER_OFFSETS[c]
            wc = np.prod(np.where(o[None, :] == 1, f, 1.0 - f), axis=1)
            w.append((o, wc))

        sl_ijk = np.stack([ii, jj, kk], axis=-1).reshape(-1, 3)[inside]
        for m in mode_indices:
            comps = []
            for c in range(3):
                g = grids[m][c]
                val = np.zeros(len(Li), dtype=np.float32)
                for o, wc in w:
                    val += wc * g[i0[:, 0] + o[0], i0[:, 1] + o[1], i0[:, 2] + o[2]]
                comps.append(val)
            mag = np.sqrt(comps[0] ** 2 + comps[1] ** 2 + comps[2] ** 2)
            out_arrays[f"mode_{m + 1:03d}_magnitude"][sl_ijk[:, 0], sl_ijk[:, 1],
                                                      sl_ijk[:, 2]] = mag
        if e_cell is not None:
            ci_in = cell[inside]
            out_arrays['material_E_Pa'][sl_ijk[:, 0], sl_ijk[:, 1], sl_ijk[:, 2]] = \
                e_cell[ci_in[:, 0], ci_in[:, 1], ci_in[:, 2]]

    for name, arr in out_arrays.items():
        p = os.path.join(qa_dir, f"{name}.nii.gz")
        nib.save(nib.Nifti1Image(arr, affine), p)
        written.append(p)
    return written


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--synthseg', required=True, help='SynthSeg labelmap NIfTI')
    ap.add_argument('--tumor', default=None,
                    help='tumor/cavity seg NIfTI (1=tumor, 2=cavity), any grid')
    ap.add_argument('--out', required=True, help='output basis .npz')
    ap.add_argument('--h', type=float, default=4.0, help='lattice spacing mm')
    ap.add_argument('--n-modes', type=int, default=30)
    ap.add_argument('--qa-dir', default=None)
    ap.add_argument('--k-found-frac', type=float, default=0.02,
                    help='foundation stiffness as fraction of median parenchyma '
                         'diagonal K entry (1-5%% recommended)')
    ap.add_argument('--solver', choices=['auto', 'shift-invert', 'lobpcg'],
                    default='auto')
    args = ap.parse_args()

    timings = {}
    t_all = time.time()

    # ---- load ----
    seg_img = nib.load(args.synthseg)
    seg = np.asanyarray(seg_img.dataobj).astype(np.int32)
    mask = seg > 0
    tumor_data = tumor_affine = None
    if args.tumor:
        tum_img = nib.load(args.tumor)
        tumor_data = np.asanyarray(tum_img.dataobj).astype(np.int32)
        tumor_affine = tum_img.affine
    log(f"[load] synthseg {seg.shape}, brain voxels {int(mask.sum())}, "
        f"tumor seg: {'yes' if tumor_data is not None else 'no'}")

    # ---- lattice ----
    t0 = time.time()
    nodes_mm, tets, meta = build_lattice(mask, seg_img.affine, args.h)
    tets = fix_orientation(nodes_mm, tets)
    timings['lattice_s'] = time.time() - t0
    n_nodes, n_tets = len(nodes_mm), len(tets)
    log(f"[lattice] h={args.h} mm: n_nodes={n_nodes}, n_tets={n_tets}, "
        f"kept cells={int(meta['keep_cells'].sum())} ({timings['lattice_s']:.1f} s)")

    # ---- materials ----
    t0 = time.time()
    E, nu, tet_labels, tet_tumor, mat_counts = assign_materials(
        nodes_mm, tets, seg, seg_img.affine, tumor_data, tumor_affine)
    timings['materials_s'] = time.time() - t0
    log(f"[materials] {mat_counts} ({timings['materials_s']:.1f} s)")

    # ---- assembly (SI: mm -> m) ----
    t0 = time.time()
    K, mass_dof, vol_total = assemble_K_M(nodes_mm * 1e-3, tets, E, nu)
    timings['assembly_s'] = time.time() - t0
    log(f"[assembly] ndof={K.shape[0]}, nnz(K)={K.nnz}, mesh volume "
        f"{vol_total * 1e6:.0f} cm^3, total mass {mass_dof.sum() / 3:.3f} kg "
        f"({timings['assembly_s']:.1f} s)")

    # ---- foundation ----
    t0 = time.time()
    K, k_found, surf_nodes = add_foundation(K, meta['node_cell_count'], E, tets,
                                            n_nodes, args.k_found_frac)
    timings['foundation_s'] = time.time() - t0
    log(f"[foundation] {len(surf_nodes)} surface nodes "
        f"({100 * len(surf_nodes) / n_nodes:.1f}%), k_found={k_found:.3e} N/m "
        f"({args.k_found_frac * 100:.1f}% of median parenchyma diag) "
        f"({timings['foundation_s']:.1f} s)")

    # ---- solve ----
    t0 = time.time()
    vals, vecs, solver_used = solve_modes(K, mass_dof, args.n_modes, args.solver)
    timings['solve_s'] = time.time() - t0
    log(f"[solve] {solver_used}: {args.n_modes} modes in {timings['solve_s']:.1f} s; "
        f"eigvals [{vals[0]:.4e}, {vals[-1]:.4e}] (rad/s)^2, "
        f"f = [{np.sqrt(vals[0]) / (2 * np.pi):.2f}, "
        f"{np.sqrt(vals[-1]) / (2 * np.pi):.2f}] Hz")

    # ---- QA checks ----
    qa = qa_checks(K, mass_dof, vals, vecs)
    log("[qa] " + json.dumps(qa))
    ok = qa['K_sym_ok'] and qa['eigvals_positive'] and qa['M_orthonormal_ok'] \
        and qa['no_nan'] and qa['eigvals_monotonic']
    log(f"[qa] OVERALL: {'PASS' if ok else 'FAIL'}")

    # ---- save npz ----
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    params = dict(synthseg=os.path.abspath(args.synthseg),
                  tumor=os.path.abspath(args.tumor) if args.tumor else None,
                  h_mm=args.h, n_modes=args.n_modes,
                  k_found_frac=args.k_found_frac, k_found=k_found,
                  rho=RHO, solver=solver_used,
                  materials=dict(csf=MAT_CSF, parenchyma=MAT_PARENCHYMA,
                                 tumor=MAT_TUMOR, cavity=MAT_CAVITY),
                  material_counts=mat_counts,
                  units='nodes: world mm; K,M assembled in SI (m, Pa, kg); '
                        'eigvals (rad/s)^2; modes M-orthonormal (1/sqrt(kg))',
                  qa=qa, timings=timings)
    np.savez_compressed(
        args.out,
        nodes=nodes_mm.astype(np.float64),
        tets=tets.astype(np.int32),
        tet_labels=tet_labels.astype(np.int16),
        tet_tumor=tet_tumor,
        E=E.astype(np.float32), nu=nu.astype(np.float32),
        eigvals=vals.astype(np.float64),
        modes=vecs.reshape(n_nodes, 3, -1).astype(np.float32),
        surface_nodes=surf_nodes.astype(np.int32),
        grid_origin_mm=meta['origin_mm'],
        grid_h_mm=np.float64(args.h),
        grid_n_cells=meta['n_cells'].astype(np.int32),
        grid_keep_cells=meta['keep_cells'],
        node_grid_ijk=meta['node_ijk'].astype(np.int32),
        synthseg_affine=seg_img.affine,
        synthseg_shape=np.array(seg_img.shape, dtype=np.int32),
        params=json.dumps(params))
    timings['save_s'] = time.time() - t0
    log(f"[save] {args.out} ({timings['save_s']:.1f} s)")

    # ---- QA NIfTI exports ----
    if args.qa_dir:
        t0 = time.time()
        os.makedirs(args.qa_dir, exist_ok=True)
        want = [m for m in (0, 4, 14) if m < args.n_modes]  # modes 1, 5, 15
        written = export_mode_niftis(args.qa_dir, want, vecs, n_nodes, meta,
                                     seg_img, E_tet=E, tets=tets)
        timings['qa_export_s'] = time.time() - t0
        for p in written:
            log(f"[qa] wrote {p}")
        with open(os.path.join(args.qa_dir, 'qa_summary.json'), 'w') as f:
            json.dump(dict(qa=qa, timings=timings, params=params,
                           n_nodes=n_nodes, n_tets=n_tets,
                           eigvals=vals.tolist()), f, indent=2)
        log(f"[qa] exports done ({timings['qa_export_s']:.1f} s)")

    timings['total_s'] = time.time() - t_all
    log(f"[timing] {json.dumps({k: round(v, 2) for k, v in timings.items()})}")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
