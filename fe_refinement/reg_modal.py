"""Registro US(fixed) <- T2preop(moving) restringido a base modal FEM (ReMIND).

El campo de deformacion se parametriza como u(x) = sum_i c_i * phi_i(x), donde
phi_i son los 30 modos propios generalizados (K u = lambda M u) precalculados por
biomech_basis.py sobre la malla tetraedrica del caso. Solo se optimizan los 30
coeficientes c_i (Adam full-batch) con:
  - MIND-SSC (mismo termino de datos que reg_core) dentro del FoV del US,
  - prior de energia lambda_eig * sum_i (eigval_i/eigval_0) * c_i^2,
  - anclaje SDF tumor/cavidad opcional (simetrico o asimetrico "us-larger").

Convencion identica a reg_core: fixed = US, moving = T2 preop, mismo marco
mundial; la rejilla de trabajo es la de load_to_grid (isotropica, cubre el US).

Uso (env monai en WSL; CPU con --cpu):
    python reg_modal.py --fixed US.nii.gz --moving T2.nii.gz \
        --basis case_basis.npz --out outdir \
        [--anchor-fixed segUS.nii.gz --anchor-moving segMR.nii.gz \
         --anchor-label 1 --anchor-mode sym|us-larger] \
        [--spacing 1.5 --iters 150 --lambda-eig 1e-4 --residual --cpu]

Salidas: T2_warped, disp_mm (total, mundo mm), jacobian, US_fixed_grid,
coeffs.json (c_i, curvas de perdida por termino, metricas, tiempos).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reg_core import (load_to_grid, resample_to_grid, normalize, mind_ssc,
                      warp, jacobian_det)


# ------------------------------------------------------------------ base modal

def build_phi(basis_path, grid_aff, dims, spacing, falloff_mm=5.0):
    """Interpola los 30 modos desde la retícula regular de nodos a la rejilla
    de trabajo (una sola vez). Devuelve PHI (30,3,D,H,W) en VOXELES de la
    rejilla de trabajo, ya atenuado fuera de la malla (coseno ~falloff_mm),
    y la mascara inside (np bool) de la region cubierta por la malla."""
    z = np.load(str(basis_path), allow_pickle=True)
    modes = np.asarray(z["modes"], dtype=np.float32)        # (n,3,30), mm/coef
    node_ijk = np.asarray(z["node_grid_ijk"], dtype=np.int64)
    origin = np.asarray(z["grid_origin_mm"], dtype=np.float64)
    h = float(z["grid_h_mm"])
    n_cells = np.asarray(z["grid_n_cells"], dtype=np.int64)
    keep = np.asarray(z["grid_keep_cells"], dtype=bool)
    n_modes = modes.shape[2]

    # nodos -> reticula de vertices (n_cells+1); vertices sin nodo = 0
    vd = n_cells + 1
    V = np.zeros((vd[0], vd[1], vd[2], 3, n_modes), dtype=np.float32)
    assert (node_ijk >= 0).all() and (node_ijk < vd).all(), \
        "node_grid_ijk fuera de la reticula de vertices"
    V[node_ijk[:, 0], node_ijk[:, 1], node_ijk[:, 2]] = modes

    # coords mundo de la rejilla de trabajo (grid_aff es diagonal)
    sp = np.diag(grid_aff)[:3]
    lo = grid_aff[:3, 3]
    ax = [lo[a] + sp[a] * np.arange(dims[a], dtype=np.float64) for a in range(3)]
    wx, wy, wz = np.meshgrid(*ax, indexing="ij")
    q = np.stack([(wx - origin[0]) / h, (wy - origin[1]) / h,
                  (wz - origin[2]) / h], axis=-1)          # coords de vertice

    # mascara inside: la celda que contiene el punto existe y esta en keep
    cell = np.floor(q).astype(np.int64)
    ok = ((cell >= 0) & (cell < n_cells)).all(axis=-1)
    inside = np.zeros(tuple(dims), dtype=bool)
    cc = cell[ok]
    inside[ok] = keep[cc[:, 0], cc[:, 1], cc[:, 2]]

    # interpolacion trilineal de los 3*n_modes canales de una vez
    Vt = torch.from_numpy(np.ascontiguousarray(
        V.transpose(3, 4, 0, 1, 2).reshape(3 * n_modes, *vd)))[None]
    norm = 2.0 * torch.from_numpy(q) / (torch.tensor(vd, dtype=torch.float64)
                                        - 1.0) - 1.0
    gridc = norm[..., [2, 1, 0]][None].to(torch.float32)
    phi = F.grid_sample(Vt, gridc, mode="bilinear", padding_mode="zeros",
                        align_corners=True)[0]              # (3*n_modes,D,H,W)
    phi = phi.reshape(3, n_modes, *[int(d) for d in dims]).permute(1, 0, 2, 3, 4)

    # atenuacion coseno fuera de la malla (~falloff_mm desde el borde)
    d_mm = distance_transform_edt(~inside, sampling=[sp[0], sp[1], sp[2]])
    t = np.clip(d_mm / max(falloff_mm, 1e-6), 0.0, 1.0)
    wgt = (0.5 * (1.0 + np.cos(np.pi * t))).astype(np.float32)
    phi = phi * torch.from_numpy(wgt)[None, None]

    phi = phi / float(spacing)          # mm -> voxeles de la rejilla de trabajo
    return phi.contiguous(), inside


# ------------------------------------------------------------------- SDF ancla

def sdf_on_grid(seg_path, label, grid_aff, dims, spacing):
    """SDF (mm, + fuera / - dentro) de `label` remuestreado por afines a la
    rejilla de trabajo (nearest) y EDT ya en la rejilla de trabajo."""
    img = nib.load(str(seg_path))
    lab = (np.asarray(img.dataobj) == label).astype(np.float32)
    if lab.sum() == 0:
        raise ValueError(f"label {label} vacio en {seg_path}")
    arr = resample_to_grid(img, grid_aff, dims, mode="nearest", data=lab)
    obj = arr > 0.5
    if not obj.any():
        raise ValueError(f"label {label} de {seg_path} cae fuera de la rejilla")
    sp = [spacing] * 3
    sdf = (distance_transform_edt(~obj, sampling=sp)
           - distance_transform_edt(obj, sampling=sp))
    return sdf.astype(np.float32)


def anchor_penalty(sdf_moving_warped, sdf_fixed, band, mode, tol_mm=1.5):
    """Penalizacion en banda |sdf_fixed|<band_mm. sym: L1. us-larger: no
    penaliza que la superficie movida quede hasta tol_mm DENTRO de la fija
    (US sobreestima ~8%), penaliza salidas normalmente."""
    diff = sdf_moving_warped - sdf_fixed   # >0: movido dentro de la fija
    if mode == "sym":
        pen = diff.abs()
    else:  # us-larger
        pen = F.relu(-diff) + F.relu(diff - tol_mm)
    return (pen * band).sum() / (band.sum() + 1e-6)


# -------------------------------------------------------------------- registro

def register_modal(f_n, m_n, fov, phi, eigvals, iters=150, lam_eig=1e-4,
                   lr=0.1, anchor=None, anchor_weight=0.1, anchor_mode="sym",
                   anchor_tol=1.5, device="cpu", log=print):
    """Optimiza los coeficientes modales. anchor: None o (sdf_f, sdf_m, band)
    np arrays en la rejilla de trabajo. Devuelve (c, disp_vox, history)."""
    dev = torch.device(device)
    f0 = torch.from_numpy(f_n)[None, None].to(dev)
    m0 = torch.from_numpy(m_n)[None, None].to(dev)
    k0 = torch.from_numpy(fov.astype(np.float32))[None, None].to(dev)
    phi = phi.to(dev)
    n_modes = phi.shape[0]

    eig = np.asarray(eigvals, dtype=np.float64)
    eig_norm = torch.tensor(eig / eig[0], dtype=torch.float32, device=dev)

    mind_f = mind_ssc(f0)
    if anchor is not None:
        sdf_f = torch.from_numpy(anchor[0])[None, None].to(dev)
        sdf_m = torch.from_numpy(anchor[1])[None, None].to(dev)
        band = torch.from_numpy(anchor[2].astype(np.float32))[None, None].to(dev)

    c = torch.zeros(n_modes, device=dev, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    history = []
    every = max(1, iters // 25)
    for it in range(iters):
        opt.zero_grad()
        u = torch.einsum("m,mcdhw->cdhw", c, phi)[None]     # (1,3,D,H,W) vox
        m_w = warp(m0, u)
        mind_m = mind_ssc(m_w)
        sim = (((mind_f - mind_m) ** 2).mean(1, keepdim=True) * k0).sum() \
            / (k0.sum() + 1e-6)
        prior = lam_eig * (eig_norm * c ** 2).sum()
        loss = sim + prior
        anc = None
        if anchor is not None:
            sdf_m_w = warp(sdf_m, u)
            anc = anchor_penalty(sdf_m_w, sdf_f, band, anchor_mode, anchor_tol)
            loss = loss + anchor_weight * anc
        loss.backward()
        opt.step()
        if it % every == 0 or it == iters - 1:
            rec = {"it": it, "sim": float(sim), "prior": float(prior),
                   "total": float(loss)}
            if anc is not None:
                rec["anchor"] = float(anc)
            history.append(rec)
            log(f"  it {it:4d}  sim {rec['sim']:.5f}  prior {rec['prior']:.5f}"
                + (f"  anchor {rec['anchor']:.4f}" if anc is not None else ""))
    c = c.detach()
    u = torch.einsum("m,mcdhw->cdhw", c, phi)[None].detach()
    return c, u, history


def gaussian_smooth3d(x, sigma_vox):
    """Suavizado gaussiano separable de un campo (B,C,D,H,W)."""
    if sigma_vox <= 0:
        return x
    rad = max(1, int(round(3.0 * sigma_vox)))
    t = torch.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
    k1 = torch.exp(-0.5 * (t / sigma_vox) ** 2)
    k1 = (k1 / k1.sum())
    c = x.shape[1]
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = k1.numel()
        k = k1.view(shape).expand(c, 1, *shape[2:])
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = rad
        x = F.conv3d(F.pad(x, pad, mode="replicate"), k, groups=c)
    return x


def refine_residual(f_n, m_n, fov, disp_modal, iters=60, lam=5.0, lr=0.05,
                    max_mm=2.0, spacing=1.0, smooth_mm=0.0, device="cpu",
                    log=print):
    """Residual denso pequeno (<= max_mm via tanh) sobre el resultado modal,
    con difusion fuerte y suavizado gaussiano opcional del campo dentro del
    bucle (smooth_mm > 0 elimina el jitter voxel a voxel que produce textura
    pixelada al remuestrear). Devuelve (disp_total, history)."""
    dev = torch.device(device)
    f0 = torch.from_numpy(f_n)[None, None].to(dev)
    m0 = torch.from_numpy(m_n)[None, None].to(dev)
    k0 = torch.from_numpy(fov.astype(np.float32))[None, None].to(dev)
    mind_f = mind_ssc(f0)
    amp = max_mm / spacing                       # amplitud maxima en voxeles
    sig_vox = smooth_mm / spacing
    v = torch.zeros_like(disp_modal, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr)
    history = []
    every = max(1, iters // 10)
    for it in range(iters):
        opt.zero_grad()
        r = amp * torch.tanh(v)
        r = gaussian_smooth3d(r, sig_vox)
        u = disp_modal + r
        m_w = warp(m0, u)
        mind_m = mind_ssc(m_w)
        sim = (((mind_f - mind_m) ** 2).mean(1, keepdim=True) * k0).sum() \
            / (k0.sum() + 1e-6)
        reg = sum(((torch.gradient(r, dim=ax)[0]) ** 2).mean()
                  for ax in (2, 3, 4))
        loss = sim + lam * reg
        loss.backward()
        opt.step()
        if it % every == 0 or it == iters - 1:
            history.append({"it": it, "sim": float(sim), "reg": float(reg)})
            log(f"  res it {it:4d}  sim {sim.item():.5f}  reg {reg.item():.5f}")
    r = gaussian_smooth3d((amp * torch.tanh(v)), sig_vox).detach()
    return disp_modal + r, history


# ------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--moving", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor-fixed")
    ap.add_argument("--anchor-moving")
    ap.add_argument("--anchor-label", type=int, default=1)
    ap.add_argument("--anchor-mode", choices=["sym", "us-larger"], default="sym")
    ap.add_argument("--anchor-weight", type=float, default=0.1)
    ap.add_argument("--anchor-band", type=float, default=10.0,
                    help="semianchura de la banda |SDF_fixed| (mm)")
    ap.add_argument("--anchor-tol", type=float, default=1.5,
                    help="tolerancia hacia dentro en us-larger (mm)")
    ap.add_argument("--exclude", help="seg cuyas etiquetas se excluyen de la "
                    "similitud (cavidad/residuo: sin correspondencia en preop)")
    ap.add_argument("--exclude-labels", default="1,2")
    ap.add_argument("--exclude-dilate-mm", type=float, default=3.0)
    ap.add_argument("--spacing", type=float, default=1.5)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lambda-eig", type=float, default=1e-4, dest="lam_eig")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--falloff-mm", type=float, default=5.0)
    ap.add_argument("--residual", action="store_true",
                    help="residual denso pequeno tras la convergencia modal")
    ap.add_argument("--residual-iters", type=int, default=60)
    ap.add_argument("--lambda-res", type=float, default=5.0)
    ap.add_argument("--residual-max-mm", type=float, default=2.0)
    ap.add_argument("--residual-smooth-mm", type=float, default=1.5,
                    help="sigma (mm) del suavizado gaussiano del residual "
                    "(default 1.5: elimina la textura pixelada del jitter "
                    "voxelico y mejora TRE; 0 = comportamiento pre-2026-08-10)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if bool(args.anchor_fixed) != bool(args.anchor_moving):
        ap.error("--anchor-fixed y --anchor-moving van juntos")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    t_all = time.time()

    f_arr, m_arr, grid_aff, dims = load_to_grid(args.fixed, args.moving,
                                                args.spacing)
    fov = f_arr > 0
    if args.exclude:
        from scipy.ndimage import binary_dilation
        img = nib.load(args.exclude)
        labs = [int(x) for x in args.exclude_labels.split(",")]
        lab = np.isin(np.asarray(img.dataobj), labs).astype(np.float32)
        ex = resample_to_grid(img, grid_aff, dims, mode="nearest", data=lab) > 0.5
        if ex.any() and args.exclude_dilate_mm > 0:
            ex = binary_dilation(
                ex, iterations=max(1, int(round(args.exclude_dilate_mm
                                                / args.spacing))))
        fov &= ~ex
        print(f"exclusion similitud: {int(ex.sum())} vox")
    f_n = normalize(f_arr, fov)
    m_n = normalize(m_arr, m_arr > 0)

    t0 = time.time()
    phi, inside = build_phi(args.basis, grid_aff, dims, args.spacing,
                            falloff_mm=args.falloff_mm)
    t_phi = time.time() - t0
    print(f"PHI {tuple(phi.shape)} en {t_phi:.1f}s; "
          f"malla cubre {inside.mean()*100:.1f}% de la rejilla, "
          f"{(inside & fov).sum() / max(fov.sum(), 1) * 100:.1f}% del FoV US")

    anchor = None
    if args.anchor_fixed:
        sdf_f = sdf_on_grid(args.anchor_fixed, args.anchor_label, grid_aff,
                            dims, args.spacing)
        sdf_m = sdf_on_grid(args.anchor_moving, args.anchor_label, grid_aff,
                            dims, args.spacing)
        band = np.abs(sdf_f) < args.anchor_band
        anchor = (sdf_f, sdf_m, band)
        print(f"ancla label {args.anchor_label}: banda {int(band.sum())} vox")

    eigvals = np.load(str(args.basis))["eigvals"]
    t0 = time.time()
    c, disp, hist = register_modal(
        f_n, m_n, fov, phi, eigvals, iters=args.iters, lam_eig=args.lam_eig,
        lr=args.lr, anchor=anchor, anchor_weight=args.anchor_weight,
        anchor_mode=args.anchor_mode, anchor_tol=args.anchor_tol, device=dev)
    t_modal = time.time() - t0

    hist_res = None
    t_res = 0.0
    if args.residual:
        t0 = time.time()
        disp, hist_res = refine_residual(
            f_n, m_n, fov, disp, iters=args.residual_iters,
            lam=args.lambda_res, max_mm=args.residual_max_mm,
            spacing=args.spacing, smooth_mm=args.residual_smooth_mm,
            device=dev)
        t_res = time.time() - t0

    m_t = torch.from_numpy(m_n)[None, None].to(disp.device)
    warped = warp(m_t, disp)[0, 0].cpu().numpy()
    jd = jacobian_det(disp)[0].cpu().numpy()
    disp_np = disp[0].cpu().numpy()

    nib.save(nib.Nifti1Image(warped, grid_aff), out / "T2_warped.nii.gz")
    nib.save(nib.Nifti1Image(np.moveaxis(disp_np, 0, -1) * args.spacing,
                             grid_aff), out / "disp_mm.nii.gz")
    nib.save(nib.Nifti1Image(jd, grid_aff), out / "jacobian.nii.gz")
    nib.save(nib.Nifti1Image(f_n, grid_aff), out / "US_fixed_grid.nii.gz")

    jd_fov = jd[fov]
    u_mm = np.linalg.norm(disp_np, axis=0) * args.spacing
    u_fov = u_mm[fov]
    cn = c.cpu().numpy()
    report = {
        "device": dev, "spacing": args.spacing,
        "dims": [int(d) for d in dims], "iters": args.iters,
        "lambda_eig": args.lam_eig, "lr": args.lr,
        "anchor": None if anchor is None else {
            "mode": args.anchor_mode, "weight": args.anchor_weight,
            "label": args.anchor_label, "band_mm": args.anchor_band,
            "tol_mm": args.anchor_tol},
        "residual": bool(args.residual),
        "coeffs": [float(v) for v in cn],
        "coeffs_abs_desc": [[int(i), float(abs(cn[i]))]
                            for i in np.argsort(-np.abs(cn))],
        "eigvals_norm": [float(v) for v in eigvals / eigvals[0]],
        "final_sim": (hist_res[-1]["sim"] if hist_res else hist[-1]["sim"]),
        "final_sim_modal": hist[-1]["sim"],
        "final_anchor": hist[-1].get("anchor"),
        "disp_mm_fov": {"median": float(np.median(u_fov)),
                        "p95": float(np.percentile(u_fov, 95)),
                        "max": float(u_fov.max())},
        "folding_pct": float((jd_fov <= 0).mean() * 100.0),
        "sd_log_jac": float(np.std(np.log(np.clip(jd_fov, 1e-6, None)))),
        "history": hist, "history_residual": hist_res,
        "wall_s": time.time() - t_all,
        "t_phi_s": t_phi, "t_modal_s": t_modal, "t_residual_s": t_res,
    }
    (out / "coeffs.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("final_sim", "disp_mm_fov", "folding_pct",
                       "sd_log_jac", "wall_s")}, indent=2))


if __name__ == "__main__":
    main()
