"""v2 development (independent of the manuscript): building blocks to integrate
MCPO-style components into the pose-first biomechanical pipeline while keeping
the FEM modal basis as the transformation model.  The shipped code is only
imported (reg_core.py, reg_modal.py); nothing there is modified.

  block_correspondences   MCPO / ConvexAdam-style discrete block correlation on the
                          0.5 mm iUS box: MIND-SSC features block-averaged, SSD over
                          all integer block displacements, per-block argmin with
                          parabolic sub-block refinement, optional coupled-convex
                          smoothing (Siebert et al.), a coarse-to-fine pyramid, and a
                          confidence weight per block (SSD margin).  Returns a dense
                          field (mm) on the box plus the block correspondences.
  fit_modal_corr          modal coefficients from block correspondences by robust
                          (Huber IRLS, optional trimming) ridge least squares on the
                          FEM basis: correspondences are the data, the FEM the prior
                          (Clatz et al. 2005 with a modal basis).
  register_modal_from     the shipped MIND-gradient modal optimisation with an
                          explicit MIND scale and an optional warm start.
  refine_residual_grad    the shipped bounded gradient residual with an explicit
                          MIND scale.
  discrete_residual       bounded dense residual (<= max_mm, Gaussian-smoothed) from
                          block correspondences between the iUS and the MR warped by
                          the pose + modal field.
Conventions follow reg_modal.py: fields are displacements of fixed-grid points in
VOXELS of the working grid; the moving is the MR relabelled by the pose, mv(y) = MR(A y).
"""
import itertools
import json
import os
import tempfile
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn.functional as F

from reg_core import mind_ssc, warp, resample_to_grid


# ----------------------------------------------------------------- utilities
def box05(usi, sp=0.5, pad=0.0):
    """0.5 mm isotropic box around the iUS extent (world-axis aligned)."""
    sh = np.array(usi.shape[:3], float)
    c = np.array([[i * sh[0], j * sh[1], k * sh[2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    w = c @ usi.affine[:3, :3].T + usi.affine[:3, 3]
    lo, hi = w.min(0) - pad, w.max(0) + pad
    dims = np.ceil((hi - lo) / sp).astype(int)
    aff = np.diag([sp, sp, sp, 1.0])
    aff[:3, 3] = lo
    return aff, dims


def sample_vol(vol, vox_pts, chunk=4_000_000):
    """Trilinear samples of vol (1,C,D,H,W) at voxel coordinates vox_pts (N,3) ->
    (N,C); zeros outside (align_corners=True convention of reg_core)."""
    D, H, W = vol.shape[2:]
    shp = torch.tensor([D, H, W], device=vol.device, dtype=vol.dtype) - 1.0
    outs = []
    for s in range(0, len(vox_pts), chunk):
        p = vox_pts[s:s + chunk].to(vol.dtype)
        g = (2.0 * p / shp - 1.0)[..., [2, 1, 0]].view(1, -1, 1, 1, 3)
        o = F.grid_sample(vol, g, mode="bilinear", padding_mode="zeros", align_corners=True)
        outs.append(o[0, :, :, 0, 0].T)
    return torch.cat(outs, 0)


def world_to_vox(pts_world, aff):
    inv = torch.from_numpy(np.linalg.inv(aff)).to(pts_world.device, pts_world.dtype)
    return pts_world @ inv[:3, :3].T + inv[:3, 3]


def grid_world_points(aff, dims, dev):
    """World coordinates (N,3) of every voxel of the grid (aff, dims), ijk order."""
    ii, jj, kk = torch.meshgrid(*[torch.arange(int(d), device=dev, dtype=torch.float32) for d in dims], indexing="ij")
    v = torch.stack([ii, jj, kk], -1).view(-1, 3)
    A = torch.from_numpy(aff).to(dev, torch.float32)
    return v @ A[:3, :3].T + A[:3, 3]


def image_tensor(img, dev):
    return torch.from_numpy(np.asarray(img.dataobj, np.float32))[None, None].to(dev)


def normalize_t(x, mask):
    ref = x[mask] if mask.any() else x[x != 0]
    lo, hi = torch.quantile(ref[:: max(1, ref.numel() // 2_000_000)], torch.tensor([0.01, 0.99], device=x.device))
    return ((x - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)


# ------------------------------------------------------ block correspondences
def _correlate_level(Ff, Fm, vf, vm, hw, coupled, dev):
    """Block-level discrete correlation.  Ff, Fm: (1,12,h,w,d) block-averaged MIND;
    vf, vm: (h,w,d) validity.  Returns ub (h,w,d,3) displacement in BLOCKS (refined),
    valid (h,w,d) and margin (h,w,d) = (second best - best) / best SSD."""
    h, w, d = Ff.shape[2:]
    Fm_p = F.pad(Fm, (hw,) * 6)
    vm_p = F.pad(vm[None, None].float(), (hw,) * 6)[0, 0] > 0.5
    shifts = list(itertools.product(range(-hw, hw + 1), repeat=3))
    ssd = torch.empty((len(shifts), h, w, d), device=dev, dtype=torch.float32)
    for n, (a, b, c) in enumerate(shifts):
        sl = (slice(hw + a, hw + a + h), slice(hw + b, hw + b + w), slice(hw + c, hw + c + d))
        diff = (Ff[0] - Fm_p[0][:, sl[0], sl[1], sl[2]]).pow(2).sum(0)
        diff[~vm_p[sl]] = float("inf")
        ssd[n] = diff
    mesh = torch.tensor(shifts, device=dev, dtype=torch.float32)
    amin = ssd.argmin(0)
    top2 = torch.topk(ssd, 2, dim=0, largest=False).values
    margin = ((top2[1] - top2[0]) / top2[0].clamp(min=1e-6))
    margin[~torch.isfinite(margin)] = 0.0
    if coupled:
        disp = mesh[amin].permute(3, 0, 1, 2)[None]
        soft = F.avg_pool3d(disp, 3, stride=1, padding=1)
        acc = torch.zeros_like(disp)
        for alpha in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            pen = (mesh.view(-1, 3, 1, 1, 1) - soft[0][None]).pow(2).sum(1)
            amin_c = (ssd + alpha * pen).argmin(0)
            soft = F.avg_pool3d(mesh[amin_c].permute(3, 0, 1, 2)[None], 3, stride=1, padding=1)
            acc += soft
        ub = (acc / 6.0)[0].permute(1, 2, 3, 0)
        saturated = torch.zeros_like(vf)
    else:
        ub = mesh[amin]
        # a block whose argmin sits on the search boundary is unreliable ONLY when the
        # boundary is far (hw >= 2); at the finest levels (hw = 1) a +-1 match is a valid
        # match (bug found 2026-09-15: with hw = 1 this excluded ~99 % of the blocks)
        saturated = (ub.abs() >= hw).any(-1) if hw >= 2 else torch.zeros_like(vf)
        n1 = 2 * hw + 1
        flat = amin.view(1, -1)
        ssd_flat = ssd.view(len(shifts), -1)
        delta = torch.zeros((3,) + tuple(amin.shape), device=dev)
        for k, step in enumerate((n1 * n1, n1, 1)):
            lo = (flat - step).clamp(0, len(shifts) - 1)
            hi = (flat + step).clamp(0, len(shifts) - 1)
            s0 = ssd_flat.gather(0, flat)[0]
            sm = ssd_flat.gather(0, lo)[0]
            sp_ = ssd_flat.gather(0, hi)[0]
            den = sm - 2 * s0 + sp_
            good = (~saturated).view(-1) & torch.isfinite(sm) & torch.isfinite(sp_) & (den > 1e-6)
            dk = torch.where(good, 0.5 * (sm - sp_) / den.clamp(min=1e-6), torch.zeros_like(s0))
            delta[k] = dk.clamp(-0.5, 0.5).view(amin.shape)
        ub = ub + delta.permute(1, 2, 3, 0)
    valid = vf & ~saturated & torch.isfinite(top2[0])
    del ssd
    return ub, valid, margin


def _upsample_blocks(x_blocks, valid, g, dims):
    """Validity-normalised trilinear upsampling of a block field (h,w,d,C) to the
    voxel grid dims (block I covers fine voxels [gI, gI+g))."""
    h, w, d, C = x_blocks.shape
    v = valid.float()[None, None]
    num = F.interpolate((x_blocks.permute(3, 0, 1, 2)[None] * v), size=(h * g, w * g, d * g), mode="trilinear",
                        align_corners=False)
    den = F.interpolate(v, size=(h * g, w * g, d * g), mode="trilinear", align_corners=False)
    up = num / den.clamp(min=1e-3)
    up = up * (den > 1e-3)
    pad = [0, int(dims[2]) - d * g, 0, int(dims[1]) - w * g, 0, int(dims[0]) - h * g]
    up = F.pad(up, pad, mode="replicate")
    return up                                                                     # (1,C,D,H,W)


def block_correspondences(f_n, m_n, aff, levels, mind=(3, 3), coupled=False, weight="uniform", dev="cuda",
                          max_pts=300000, symmetric=False):
    """symmetric=True (MCPO's inverse-consistency ingredient): the backward field (moving as
    fixed) is computed too and the two are averaged, u = (u_fwd(x) - u_bwd(x)) / 2, weights =
    the minimum of the two."""
    if symmetric:
        u_f, w_f, _ = _block_correspondences(f_n, m_n, aff, levels, mind, coupled, weight, dev, max_pts)
        u_b, w_b, _ = _block_correspondences(m_n, f_n, aff, levels, mind, coupled, weight, dev, max_pts)
        u_mm = 0.5 * (u_f - u_b)
        w_vox = torch.minimum(w_f, w_b)
        g = levels[-1][0]
        with torch.no_grad():
            valid = w_vox[0, 0] > 0
            idx = torch.nonzero(F.avg_pool3d(valid.float()[None, None], g, stride=g)[0, 0] > 0.5).float()
            if len(idx) > max_pts:
                sel = torch.randperm(len(idx), generator=torch.Generator(device="cpu").manual_seed(0))[:max_pts]
                idx = idx[sel.to(idx.device)]
            centre_vox = idx * g + (g - 1) / 2.0
            A = torch.from_numpy(aff).to(dev, torch.float32)
            centres_mm = centre_vox @ A[:3, :3].T + A[:3, 3]
            disp_mm = sample_vol(u_mm, centre_vox)
            wsel = sample_vol(w_vox, centre_vox)[:, 0]
        return u_mm, w_vox, dict(centres_mm=centres_mm, disp_mm=disp_mm, weight=wsel)
    return _block_correspondences(f_n, m_n, aff, levels, mind, coupled, weight, dev, max_pts)


def _block_correspondences(f_n, m_n, aff, levels, mind=(3, 3), coupled=False, weight="uniform", dev="cuda",
                           max_pts=300000):
    """f_n, m_n: normalised fixed / moving tensors (1,1,D,H,W) on the same grid (aff, isotropic
    spacing sp).  Coarse-to-fine: at each level the moving is warped by the field accumulated so
    far, block-averaged MIND-SSC features are correlated over +-hw blocks and the block field is
    added.  Returns u_mm (1,3,D,H,W) dense field in mm, w_vox (1,1,D,H,W) confidence in [0,1]
    (0 outside valid blocks), and corr = dict(centres_mm (N,3), disp_mm (N,3), weight (N,))
    sampled at the block centres of the finest level (subsampled to <= max_pts)."""
    sp = float(aff[0, 0])
    dims = f_n.shape[2:]
    r, d_ = mind
    with torch.no_grad():
        mind_f = mind_ssc(f_n, radius=r, dilation=d_)
        u_mm = torch.zeros((1, 3) + tuple(dims), device=dev)
        valid_vox = torch.ones((1, 1) + tuple(dims), device=dev)     # blocks unreliable at a coarser level stay out
        for (g, hw) in levels:
            m_cur = m_n if float(u_mm.abs().max()) == 0 else warp(m_n, u_mm / sp)
            Ff = F.avg_pool3d(mind_f, g, stride=g)
            Fm = F.avg_pool3d(mind_ssc(m_cur, radius=r, dilation=d_), g, stride=g)
            vf = F.avg_pool3d((f_n > 0).float(), g, stride=g)[0, 0] > 0.9
            vm = F.avg_pool3d((m_cur > 0).float(), g, stride=g)[0, 0] > 0.9
            ub, valid, margin = _correlate_level(Ff, Fm, vf, vm, hw, coupled, dev)
            valid = valid & (F.avg_pool3d(valid_vox, g, stride=g)[0, 0] > 0.5)
            inc = _upsample_blocks(ub * (g * sp), valid, g, dims)
            u_mm = u_mm + inc
            valid_vox = _upsample_blocks(valid.float()[..., None], torch.ones_like(valid), g, dims) > 0.5
            valid_vox = valid_vox.float()
            last = (ub, valid, margin, g)
        ub, valid, margin, g = last
        if weight == "margin":
            tau = float(margin[valid].median()) if valid.any() else 1.0
            wb = margin / (margin + max(tau, 1e-6))
        else:
            wb = torch.ones_like(margin)
        wb = wb * valid.float()
        w_vox = _upsample_blocks(wb[..., None], valid, g, dims)
        idx = torch.nonzero(valid).float()
        if len(idx) > max_pts:
            sel = torch.randperm(len(idx), generator=torch.Generator(device="cpu").manual_seed(0))[:max_pts]
            idx = idx[sel.to(idx.device)]
        centre_vox = idx * g + (g - 1) / 2.0
        A = torch.from_numpy(aff).to(dev, torch.float32)
        centres_mm = centre_vox @ A[:3, :3].T + A[:3, 3]
        disp_mm = sample_vol(u_mm, centre_vox)
        wsel = wb[idx[:, 0].long(), idx[:, 1].long(), idx[:, 2].long()]
    return u_mm, w_vox, dict(centres_mm=centres_mm, disp_mm=disp_mm, weight=wsel)


# ------------------------------------------------------------- modal fits
def fit_modal_corr(phi, eigvals, grid_aff, corr, lam_rel=0.1, huber_mm=1.0, trim=0.0, iters=8):
    """Robust ridge least squares of the modal coefficients on block correspondences.
    phi: (n_modes,3,D,H,W) in VOXELS of grid_aff (spacing sp); corr: centres_mm, disp_mm,
    weight.  Minimises sum_i w_i rho_huber(|Phi(x_i) c - d_i|) + lam * sum_m eig_norm_m c_m^2,
    lam = lam_rel * trace(A^T W A) / n_modes.  Returns c (n_modes,), info."""
    dev = phi.device
    sp = float(grid_aff[0, 0])
    n_modes = phi.shape[0]
    v = world_to_vox(corr["centres_mm"], grid_aff)
    P = sample_vol(phi.reshape(1, n_modes * 3, *phi.shape[2:]), v).view(-1, n_modes, 3) * sp   # (N, modes, 3) mm/coef
    Amat = P.permute(0, 2, 1).reshape(-1, n_modes).double()                                    # (3N, modes)
    d = corr["disp_mm"].reshape(-1).double()
    w0 = corr["weight"].double()
    eig = np.asarray(eigvals, float)
    Pr = torch.diag(torch.from_numpy(eig / eig[0]).to(dev))
    w = w0.clone()
    c = torch.zeros(n_modes, device=dev, dtype=torch.float64)
    for it in range(iters):
        W3 = w.repeat_interleave(3)
        AtW = Amat.T * W3
        G = AtW @ Amat
        lam = lam_rel * float(torch.trace(G)) / n_modes
        c = torch.linalg.solve(G + lam * Pr, AtW @ d)
        res = (Amat @ c - d).view(-1, 3).norm(dim=1)                                          # (N,) mm
        hub = torch.clamp(huber_mm / res.clamp(min=1e-6), max=1.0)
        w = w0 * hub
        if trim > 0:
            k = int(trim * len(res))
            if k > 0:
                w[torch.topk(res, k).indices] = 0.0
    inl = w > 0
    info = dict(n=int(len(res)), n_inl=int(inl.sum()), rms_inl_mm=float(torch.sqrt((res[inl] ** 2).mean())) if inl.any() else float("nan"),
                med_res_mm=float(res.median()), coef_norm=float(c.norm()))
    # posterior covariance of the coefficients (Gaussian view of the robust ridge: the
    # weighted residual variance per axis times the inverse normal matrix) -> S5 uncertainty
    if inl.any():
        W3 = w.repeat_interleave(3)
        G = (Amat.T * W3) @ Amat
        r3 = (Amat @ c - d) ** 2
        sigma2 = float((W3 * r3).sum() / W3.sum())
        cov = sigma2 * torch.linalg.inv(G + lam * Pr)
        info["sigma2_mm2"] = sigma2
        info["cov"] = cov.cpu().numpy().tolist()
    return c.float(), info


def landmark_sigma(phi, grid_aff, cov, pts_mm):
    """Predicted 1-sigma uncertainty (mm, isotropic summary sqrt(trace/3)) of the modal
    displacement at points pts_mm, from the coefficient covariance cov (n_modes x n_modes)."""
    dev = phi.device
    sp = float(grid_aff[0, 0])
    n_modes = phi.shape[0]
    v = world_to_vox(torch.as_tensor(np.asarray(pts_mm, np.float32), device=dev), grid_aff)
    P = sample_vol(phi.reshape(1, n_modes * 3, *phi.shape[2:]), v).view(-1, n_modes, 3) * sp   # (N, modes, 3)
    C = torch.as_tensor(np.asarray(cov, np.float32), device=dev)
    var = torch.einsum("nmk,ml,nlk->nk", P, C, P)                                              # (N, 3)
    return torch.sqrt(var.clamp(min=0).sum(1) / 3.0).cpu().numpy()


def register_modal_from(f_n, m_n, fov, phi, eigvals, c0=None, iters=200, lam_eig=3e-3, lr=0.1, mind=(2, 2), dev="cuda"):
    """The shipped MIND-gradient modal optimisation (reg_modal.register_modal) with an
    explicit MIND scale and an optional warm start c0.  Returns (c, u_vox (1,3,D,H,W))."""
    r, d_ = mind
    f0 = torch.from_numpy(f_n)[None, None].to(dev)
    m0 = torch.from_numpy(m_n)[None, None].to(dev)
    k0 = torch.from_numpy(fov.astype(np.float32))[None, None].to(dev)
    phi = phi.to(dev)
    eig = np.asarray(eigvals, dtype=np.float64)
    eig_norm = torch.tensor(eig / eig[0], dtype=torch.float32, device=dev)
    mind_f = mind_ssc(f0, radius=r, dilation=d_)
    c = torch.zeros(phi.shape[0], device=dev) if c0 is None else c0.detach().clone().to(dev)
    c.requires_grad_(True)
    opt = torch.optim.Adam([c], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        u = torch.einsum("m,mcdhw->cdhw", c, phi)[None]
        m_w = warp(m0, u)
        mind_m = mind_ssc(m_w, radius=r, dilation=d_)
        sim = (((mind_f - mind_m) ** 2).mean(1, keepdim=True) * k0).sum() / (k0.sum() + 1e-6)
        loss = sim + lam_eig * (eig_norm * c ** 2).sum()
        loss.backward()
        opt.step()
    c = c.detach()
    return c, torch.einsum("m,mcdhw->cdhw", c, phi)[None].detach()


def register_modal_offset(f_n, m_n, fov, phi, eigvals, u0, iters=100, lam_eig=3e-3, lr=0.1, mind=(2, 2), dev="cuda"):
    """Gradient fit of the coefficients of a second (local / substructure) modal basis on
    top of a fixed field u0: u = u0 + sum_m c_m phi_m, same MIND loss and eigenvalue prior as
    the global fit.  Returns (c, u_total)."""
    r, d_ = mind
    f0 = torch.from_numpy(f_n)[None, None].to(dev)
    m0 = torch.from_numpy(m_n)[None, None].to(dev)
    k0 = torch.from_numpy(fov.astype(np.float32))[None, None].to(dev)
    phi = phi.to(dev)
    u0 = u0.to(dev).detach()
    eig = np.asarray(eigvals, dtype=np.float64)
    eig_norm = torch.tensor(eig / eig[0], dtype=torch.float32, device=dev)
    mind_f = mind_ssc(f0, radius=r, dilation=d_)
    c = torch.zeros(phi.shape[0], device=dev, requires_grad=True)
    opt = torch.optim.Adam([c], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        u = u0 + torch.einsum("m,mcdhw->cdhw", c, phi)[None]
        mind_m = mind_ssc(warp(m0, u), radius=r, dilation=d_)
        sim = (((mind_f - mind_m) ** 2).mean(1, keepdim=True) * k0).sum() / (k0.sum() + 1e-6)
        loss = sim + lam_eig * (eig_norm * c ** 2).sum()
        loss.backward()
        opt.step()
    c = c.detach()
    return c, (u0 + torch.einsum("m,mcdhw->cdhw", c, phi)[None]).detach()


def refine_residual_grad(f_n, m_n, fov, disp_modal, iters=60, lam=5.0, lr=0.05, max_mm=2.0, spacing=1.0,
                         smooth_mm=1.5, mind=(2, 2), dev="cuda"):
    """The shipped bounded gradient residual (reg_modal.refine_residual) with an explicit MIND scale."""
    import reg_modal
    r, d_ = mind
    f0 = torch.from_numpy(f_n)[None, None].to(dev)
    m0 = torch.from_numpy(m_n)[None, None].to(dev)
    k0 = torch.from_numpy(fov.astype(np.float32))[None, None].to(dev)
    mind_f = mind_ssc(f0, radius=r, dilation=d_)
    amp, sig = max_mm / spacing, smooth_mm / spacing
    v = torch.zeros_like(disp_modal, requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        res = reg_modal.gaussian_smooth3d(amp * torch.tanh(v), sig)
        u = disp_modal + res
        mind_m = mind_ssc(warp(m0, u), radius=r, dilation=d_)
        sim = (((mind_f - mind_m) ** 2).mean(1, keepdim=True) * k0).sum() / (k0.sum() + 1e-6)
        reg = sum(((torch.gradient(res, dim=ax)[0]) ** 2).mean() for ax in (2, 3, 4))
        (sim + lam * reg).backward()
        opt.step()
    res = reg_modal.gaussian_smooth3d(amp * torch.tanh(v), sig).detach()
    return disp_modal + res


def warp_mr_on_box(mri_img, A, u_vox, grid_aff, aff05, dims05, dev):
    """MR under the pose and the modal field, sampled on the 0.5 mm box: value at box point x
    = MR(A (x + u(x))), u interpolated from the working grid (voxels -> mm)."""
    import reg_modal  # noqa: F401  (kept for symmetry; not used)
    sp = float(grid_aff[0, 0])
    x = grid_world_points(aff05, dims05, dev)
    u_mm = sample_vol(u_vox * sp, world_to_vox(x, grid_aff))
    y = x + u_mm
    At = torch.from_numpy(A).to(dev, torch.float32)
    yw = y @ At[:3, :3].T + At[:3, 3]
    mr = image_tensor(mri_img, dev)
    vals = sample_vol(mr, world_to_vox(yw, mri_img.affine))
    return vals.view(1, 1, *[int(d) for d in dims05])


def discrete_residual(f05, mw05, aff05, grid_aff, dims, fov, levels, mind=(3, 3), coupled=False, weight="margin",
                      max_mm=2.0, smooth_mm=1.5, dev="cuda"):
    """Bounded dense residual on the working grid from block correspondences between the iUS
    box (f05) and the MR warped by pose + modal field (mw05).  Returns r_vox (1,3,D,H,W)."""
    import reg_modal
    u05, w05, corr = block_correspondences(f05, mw05, aff05, levels, mind=mind, coupled=coupled, weight=weight, dev=dev)
    x = grid_world_points(grid_aff, dims, dev)
    v05 = world_to_vox(x, aff05)
    num = sample_vol(u05 * w05, v05)
    den = sample_vol(w05, v05)
    r = (num / den.clamp(min=1e-3)) * (den > 1e-3)
    r = r.T.reshape(1, 3, *[int(d) for d in dims])
    n = r.norm(dim=1, keepdim=True)
    r = r * (max_mm / n.clamp(min=max_mm))                     # clip |r| <= max_mm
    sp = float(grid_aff[0, 0])
    r = reg_modal.gaussian_smooth3d(r, smooth_mm / sp)
    return r / sp, dict(n_corr=int(len(corr["weight"])), res_rms_mm=float(torch.sqrt((r[0].norm(dim=0)[torch.from_numpy(fov).to(dev)] ** 2).mean())))


# ------------------------------------------- FEM-native residuals (v2, S4)
def residual_correspondences(f05, mw05, aff05, levels, mind=(3, 3), coupled=False, weight="margin", dev="cuda",
                             max_pts=300000, symmetric=False):
    """Block correspondences between the iUS box and the MR warped by pose + modal field:
    the data of the FEM-native residuals (centres_mm, disp_mm, weight)."""
    _, _, corr = block_correspondences(f05, mw05, aff05, levels, mind=mind, coupled=coupled, weight=weight, dev=dev,
                                       max_pts=max_pts, symmetric=symmetric)
    return corr


def enrich_residual(basis80_path, grid_aff, dims, corr, lam_rel=0.1, huber_mm=1.0, trim=0.0, n_low=30, dev="cuda"):
    """Modal enrichment: the residual is explained by the higher FEM modes (n_low+1 .. 80)
    fitted to the residual correspondences (robust ridge), not by a free-form field.
    Returns r_vox (1,3,D,H,W) on the working grid and info."""
    import reg_modal
    phi80, _ = reg_modal.build_phi(basis80_path, grid_aff, dims, float(grid_aff[0, 0]))
    eig80 = np.load(str(basis80_path))["eigvals"]
    phi_hi = phi80[n_low:].to(dev)
    c_hi, info = fit_modal_corr(phi_hi, eig80[n_low:], grid_aff, corr, lam_rel, huber_mm, trim)
    info.pop("cov", None)
    r = torch.einsum("m,mcdhw->cdhw", c_hi, phi_hi)[None]
    info["n_modes_hi"] = int(phi_hi.shape[0])
    info["r_max_mm"] = float(r[0].norm(dim=0).max()) * float(grid_aff[0, 0])
    del phi80, phi_hi
    return r, info


def lattice_H(z, pts_mm):
    """Sparse trilinear interpolation H (3N x 3 n_nodes) from nodal displacements on the
    basis lattice to points; rows of points outside the kept cells are empty."""
    origin = np.asarray(z["grid_origin_mm"], float)
    h = float(z["grid_h_mm"])
    n_cells = np.asarray(z["grid_n_cells"], int)
    keep = np.asarray(z["grid_keep_cells"], bool)
    node_ijk = np.asarray(z["node_grid_ijk"], int)
    vd = n_cells + 1
    v2n = -np.ones(tuple(vd), np.int64)
    v2n[node_ijk[:, 0], node_ijk[:, 1], node_ijk[:, 2]] = np.arange(len(node_ijk))
    q = (pts_mm - origin) / h
    cell = np.floor(q).astype(int)
    f = q - cell
    ok = ((cell >= 0) & (cell < n_cells)).all(1)
    cc = np.clip(cell, 0, n_cells - 1)
    ok &= keep[cc[:, 0], cc[:, 1], cc[:, 2]]
    N = len(pts_mm)
    rows, cols, vals = [], [], []
    wsum = np.zeros(N)
    for corner in itertools.product((0, 1), repeat=3):
        vidx = np.clip(cell + np.array(corner), 0, vd - 1)
        w = np.prod(np.where(np.array(corner)[None] == 1, f, 1 - f), axis=1)
        node = v2n[vidx[:, 0], vidx[:, 1], vidx[:, 2]]
        m = ok & (node >= 0) & (w > 0)
        for k in range(3):
            rows.append(3 * np.nonzero(m)[0] + k)
            cols.append(3 * node[m] + k)
            vals.append(w[m])
        wsum[m] += w[m]
    H = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * N, 3 * len(node_ijk)))
    good = ok & (wsum > 0.5)
    # renormalise rows whose lattice corners are partly missing
    scale = np.where(wsum > 0, 1.0 / np.maximum(wsum, 1e-9), 0.0)
    H = sp.diags(np.repeat(scale, 3)) @ H
    return H.tocsr(), good


def fem_residual(z, corr, lam_rel=0.1, huber_mm=1.0, iters=6, trim=0.0):
    """Local FEM response (Clatz-style): nodal displacements u minimising
    sum_i w_i rho_huber(|H_i u - d_i|) + lam * u^T K_n u, with K_n the case's FEM stiffness
    (parenchyma / CSF / tumour materials, foundation springs on the surface nodes) normalised
    by its median diagonal and lam = lam_rel * trace(H^T W H) / ndof.  Returns u_nodes (mm)."""
    import biomech_basis as BB
    nodes_mm = np.asarray(z["nodes"], float)
    tets = np.asarray(z["tets"], int)
    K, _, _ = BB.assemble_K_M(nodes_mm * 1e-3, tets, np.asarray(z["E"], float), np.asarray(z["nu"], float))
    p = json.loads(str(z["params"]))
    surf = np.asarray(z["surface_nodes"], int)
    dfound = np.zeros(K.shape[0])
    dfound[(3 * surf[:, None] + np.arange(3)[None]).ravel()] = float(p["k_found"])
    K = (K + sp.diags(dfound)).tocsr()
    Kn = (K / float(np.median(K.diagonal()))).tocsr()
    pts = corr["centres_mm"].cpu().numpy().astype(float)
    d = corr["disp_mm"].cpu().numpy().astype(float)
    w0 = corr["weight"].cpu().numpy().astype(float)
    H, good = lattice_H(z, pts)
    H, d, w0 = H[np.repeat(good, 3)], d[good], w0[good]
    ndof = K.shape[0]
    w = w0.copy()
    u = np.zeros(ndof)
    for it in range(iters):
        W = sp.diags(np.repeat(w, 3))
        HtW = H.T @ W
        G = (HtW @ H).tocsr()
        gd = G.diagonal()
        covered = gd > 0
        # lam relative to the data strength of the DoF that carry correspondences (most
        # nodes carry none: the FEM has to interpolate / extrapolate from the covered region)
        lam = lam_rel * float(gd[covered].mean()) if covered.any() else 1.0
        A = (G + lam * Kn).tocsc()
        b = HtW @ d.reshape(-1)
        Minv = sp.diags(1.0 / np.maximum(A.diagonal(), 1e-12))
        try:
            u, info_cg = spla.cg(A, b, x0=u, M=Minv, rtol=1e-6, maxiter=2000)
        except TypeError:                                   # scipy < 1.12
            u, info_cg = spla.cg(A, b, x0=u, M=Minv, tol=1e-6, maxiter=2000)
        res = np.linalg.norm((H @ u).reshape(-1, 3) - d, axis=1)
        w = w0 * np.minimum(1.0, huber_mm / np.maximum(res, 1e-6))
        if trim > 0:
            k = int(trim * len(res))
            if k > 0:
                w[np.argsort(res)[-k:]] = 0.0
    inl = w > 0
    info = dict(n=int(len(res)), n_inl=int(inl.sum()), rms_inl_mm=float(np.sqrt((res[inl] ** 2).mean())) if inl.any() else float("nan"),
                med_res_mm=float(np.median(res)), u_max_mm=float(np.linalg.norm(u.reshape(-1, 3), axis=1).max()),
                cg_info=int(info_cg))
    return u.reshape(-1, 3), info


def lattice_field(z, u_nodes_mm, grid_aff, dims, falloff_mm=5.0):
    """Nodal displacements (n_nodes,3) mm on the basis lattice -> field on the working grid
    (1,3,D,H,W) in VOXELS, through reg_modal.build_phi's own interpolation (temporary
    one-mode basis), so that the falloff outside the mesh is identical to the modal field."""
    import reg_modal
    fd, path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    np.savez(path, modes=u_nodes_mm[:, :, None].astype(np.float32), node_grid_ijk=z["node_grid_ijk"],
             grid_origin_mm=z["grid_origin_mm"], grid_h_mm=z["grid_h_mm"], grid_n_cells=z["grid_n_cells"],
             grid_keep_cells=z["grid_keep_cells"])
    try:
        phi, _ = reg_modal.build_phi(path, grid_aff, dims, float(grid_aff[0, 0]), falloff_mm)
    finally:
        os.unlink(path)
    return phi[0][None]


# ------------------------------------------------- extrapolation test (S6)
def deep_holdout(fov, aff, frac=0.3):
    """Held-out region of the iUS field of view for the extrapolation test: the `frac` of the
    FoV extent farthest from the probe along the FoV's principal axis (the probe end is the
    end with the smaller cross-section).  Returns (mask on this grid, params) with params
    usable by holdout_flags() on any set of world points."""
    ijk = np.argwhere(fov)
    if len(ijk) > 200000:
        ijk = ijk[np.random.default_rng(0).choice(len(ijk), 200000, replace=False)]
    pts = ijk @ aff[:3, :3].T + aff[:3, 3]
    ctr = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - ctr, full_matrices=False)
    ax = vt[0]
    t = (pts - ctr) @ ax
    lo, hi = np.percentile(t, [1, 99])
    # cross-section (number of voxels) in the first and last 15 % of the extent
    n_lo = int(((t < lo + 0.15 * (hi - lo))).sum())
    n_hi = int(((t > hi - 0.15 * (hi - lo))).sum())
    if n_lo < n_hi:           # narrow end (probe) at lo -> hold out the hi end
        thr, side = hi - frac * (hi - lo), +1
    else:
        thr, side = lo + frac * (hi - lo), -1
    params = dict(axis=ax.tolist(), centre=ctr.tolist(), thr=float(thr), side=side, frac=frac)
    return holdout_flags_grid(fov, aff, params), params


def holdout_flags(pts_world, params):
    t = (np.asarray(pts_world, float) - np.asarray(params["centre"])) @ np.asarray(params["axis"])
    return (t > params["thr"]) if params["side"] > 0 else (t < params["thr"])


def holdout_flags_grid(fov, aff, params):
    ijk = np.argwhere(fov)
    pts = ijk @ aff[:3, :3].T + aff[:3, 3]
    m = np.zeros_like(fov)
    m[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = holdout_flags(pts, params)
    return m


def regularity(u_vox, fov):
    """Folding (% of FoV voxels with det J <= 0) and SDLogJ over the FoV."""
    from reg_core import jacobian_det
    jd = jacobian_det(u_vox)[0].cpu().numpy()
    det = jd[fov]
    return float((det <= 0).mean() * 100), float(np.std(np.log(np.clip(det, 1e-6, None))))
