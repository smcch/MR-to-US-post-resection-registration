"""Nucleo de registro deformable US(fixed) <- T2preop(moving) para ReMIND.

Convencion (ReMIND2Reg / BrainUS-AI): fixed = US, moving = T2 preoperatorio.
El campo phi se define en la rejilla de trabajo (espacio US remuestreado) y
lleva el T2 al espacio US.

Uso tipico (env monai en WSL, GPU; tambien corre en CPU para smoke tests):
    python reg_core.py --fixed US.nii.gz --moving T2.nii.gz --out outdir \
        --spacing 1.0 --levels 3 --iters 120 --lambda-diff 1.0 [--cpu]

Salidas: warped T2, campo de desplazamiento denso (mm, espacio de trabajo),
jacobiano, log JSON con metricas (MIND loss, SDLogJ, %folding, tiempos).
"""
import argparse
import json
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- IO

def resample_to_grid(img, grid_aff, dims, mode="bilinear", data=None):
    """Remuestrea img (nibabel) a la rejilla de trabajo por grid_sample.
    mode: "bilinear" (imagenes) o "nearest" (etiquetas). `data` opcional
    sustituye al array del propio img (misma rejilla/affine)."""
    if data is None:
        data = np.asarray(img.dataobj, dtype=np.float32)
    t = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))[None, None]
    # coords de la rejilla de trabajo -> voxeles de img
    ii, jj, kk = torch.meshgrid(
        torch.arange(dims[0], dtype=torch.float64),
        torch.arange(dims[1], dtype=torch.float64),
        torch.arange(dims[2], dtype=torch.float64), indexing="ij")
    ones = torch.ones_like(ii)
    vox = torch.stack([ii, jj, kk, ones], -1) @ torch.from_numpy(
        (np.linalg.inv(img.affine) @ grid_aff).T)
    vox = vox[..., :3]
    shp = torch.tensor(img.shape[:3], dtype=torch.float64)
    norm = 2.0 * vox / (shp - 1.0) - 1.0            # [-1,1] por eje
    gridc = norm[..., [2, 1, 0]][None].to(torch.float32)  # grid_sample: x=dim2
    out = F.grid_sample(t, gridc, mode=mode, padding_mode="zeros",
                        align_corners=True)
    return out[0, 0].numpy()


def load_to_grid(fixed_path, moving_path, spacing):
    """Carga fixed y moving; remuestrea ambos a una rejilla isotropica `spacing`
    que cubre el FoV del fixed (US). Ambos comparten marco mundial en ReMIND."""
    fx = nib.load(str(fixed_path))
    mv = nib.load(str(moving_path))

    # rejilla de trabajo: bounding box del fixed en mundo, isotropica
    sh = np.array(fx.shape[:3], dtype=float)
    corners = np.array([[i * sh[0], j * sh[1], k * sh[2], 1.0]
                        for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    world = corners @ fx.affine.T
    lo, hi = world[:, :3].min(0), world[:, :3].max(0)
    dims = np.maximum(np.ceil((hi - lo) / spacing).astype(int), 8)
    grid_aff = np.diag([spacing, spacing, spacing, 1.0])
    grid_aff[:3, 3] = lo

    f_arr = resample_to_grid(fx, grid_aff, dims)
    m_arr = resample_to_grid(mv, grid_aff, dims)
    return f_arr, m_arr, grid_aff, dims


def normalize(arr, mask=None, p_lo=1.0, p_hi=99.0):
    ref = arr[mask] if mask is not None and mask.any() else arr[arr != 0]
    if ref.size == 0:
        return arr
    lo, hi = np.percentile(ref, [p_lo, p_hi])
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------------ MIND-SSC

def pdist_squared(x):
    xx = (x ** 2).sum(dim=1).unsqueeze(2)
    yy = xx.permute(0, 2, 1)
    dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
    return dist.clamp(min=0.0)


def mind_ssc(img, radius=2, dilation=2):
    """MIND-SSC 3D (Heinrich; implementacion de referencia ConvexAdam).
    img: (B,1,D,H,W) en [0,1]. Devuelve (B,12,D,H,W)."""
    device, dtype = img.device, img.dtype
    six = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1],
                        [1, 1, 2], [2, 1, 1], [1, 2, 1]],
                       dtype=torch.float32, device=device)
    dist = pdist_squared(six.t().unsqueeze(0)).squeeze(0)
    x, y = torch.meshgrid(torch.arange(6, device=device),
                          torch.arange(6, device=device), indexing="ij")
    mask = (x > y).view(-1) & (dist == 2).view(-1)
    idx_shift1 = six.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :].long()
    idx_shift2 = six.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :].long()

    mshift1 = torch.zeros(12, 1, 3, 3, 3, device=device, dtype=dtype)
    mshift1.view(-1)[torch.arange(12, device=device) * 27
                     + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    mshift2 = torch.zeros(12, 1, 3, 3, 3, device=device, dtype=dtype)
    mshift2.view(-1)[torch.arange(12, device=device) * 27
                     + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad = nn_replication_pad(dilation)

    ssd = smooth((F.conv3d(rpad(img), mshift1, dilation=dilation)
                  - F.conv3d(rpad(img), mshift2, dilation=dilation)) ** 2, radius)

    mind = ssd - ssd.min(dim=1, keepdim=True)[0]
    mind_var = mind.mean(dim=1, keepdim=True)
    mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    mind = torch.exp(-mind / (mind_var + 1e-8))
    order = torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], device=device)
    return mind[:, order]


def nn_replication_pad(d):
    return torch.nn.ReplicationPad3d(d)


def smooth(img, radius):
    """Box smoothing separable de radio `radius`."""
    k = 2 * radius + 1
    w = torch.ones(1, 1, k, 1, 1, device=img.device, dtype=img.dtype) / k
    c = img.shape[1]
    img = F.conv3d(F.pad(img, (0, 0, 0, 0, radius, radius), mode="replicate"),
                   w.expand(c, 1, k, 1, 1), groups=c)
    img = F.conv3d(F.pad(img, (0, 0, radius, radius, 0, 0), mode="replicate"),
                   w.expand(c, 1, k, 1, 1).transpose(2, 3), groups=c)
    img = F.conv3d(F.pad(img, (radius, radius, 0, 0, 0, 0), mode="replicate"),
                   w.expand(c, 1, k, 1, 1).transpose(2, 4), groups=c)
    return img


# ---------------------------------------------------------------------- transformar

def warp(moving, disp_vox):
    """moving: (B,1,D,H,W); disp_vox: (B,3,D,H,W) desplazamiento en VOXELES
    de la rejilla de trabajo. Devuelve moving deformado."""
    B, _, D, H, W = moving.shape
    base = torch.stack(torch.meshgrid(
        torch.arange(D, device=moving.device, dtype=moving.dtype),
        torch.arange(H, device=moving.device, dtype=moving.dtype),
        torch.arange(W, device=moving.device, dtype=moving.dtype),
        indexing="ij"))[None]
    coords = base + disp_vox
    shp = torch.tensor([D, H, W], device=moving.device, dtype=moving.dtype)
    norm = 2.0 * coords / (shp.view(1, 3, 1, 1, 1) - 1.0) - 1.0
    gridc = norm.permute(0, 2, 3, 4, 1)[..., [2, 1, 0]]
    return F.grid_sample(moving, gridc, mode="bilinear",
                         padding_mode="border", align_corners=True)


def jacobian_det(disp_vox):
    """Determinante jacobiano de phi = id + disp (diferencias centrales)."""
    grads = []
    for ax in (2, 3, 4):
        g = torch.gradient(disp_vox, dim=ax)[0]
        grads.append(g)
    J = torch.stack(grads, dim=-1)  # (B,3,D,H,W,3): dcomp/daxis
    eye = torch.eye(3, device=disp_vox.device, dtype=disp_vox.dtype)
    Jm = J.permute(0, 2, 3, 4, 1, 5) + eye
    return torch.linalg.det(Jm)


# ------------------------------------------------------------------------- registro

def register(fixed, moving, fov_mask, levels=3, iters=120, lam=1.0,
             lr=0.5, device="cpu", log=print):
    """Campo denso multiresolucion. fixed/moving: np (D,H,W) en [0,1].
    fov_mask: np bool, region valida del US. Devuelve disp final (3,D,H,W) vox."""
    dev = torch.device(device)
    f0 = torch.from_numpy(fixed)[None, None].to(dev)
    m0 = torch.from_numpy(moving)[None, None].to(dev)
    k0 = torch.from_numpy(fov_mask.astype(np.float32))[None, None].to(dev)

    disp = None
    for lev in range(levels, 0, -1):
        scale = 2 ** (lev - 1)
        size = [max(int(round(s / scale)), 8) for s in fixed.shape]
        f = F.interpolate(f0, size=size, mode="trilinear", align_corners=True)
        m = F.interpolate(m0, size=size, mode="trilinear", align_corners=True)
        k = F.interpolate(k0, size=size, mode="trilinear", align_corners=True)
        mind_f = mind_ssc(f)
        if disp is None:
            disp = torch.zeros(1, 3, *size, device=dev)
        else:
            prev = torch.tensor(prev_size, dtype=torch.float32, device=dev)
            disp = F.interpolate(disp, size=size, mode="trilinear",
                                 align_corners=True)
            disp = disp * (torch.tensor(size, dtype=torch.float32, device=dev)
                           / prev).view(1, 3, 1, 1, 1)
        disp.requires_grad_(True)
        opt = torch.optim.Adam([disp], lr=lr / scale)
        t0 = time.time()
        for it in range(iters):
            opt.zero_grad()
            m_w = warp(m, disp)
            mind_m = mind_ssc(m_w)
            sim = (((mind_f - mind_m) ** 2).mean(1, keepdim=True) * k).sum() \
                / (k.sum() + 1e-6)
            reg = sum(((torch.gradient(disp, dim=ax)[0]) ** 2).mean()
                      for ax in (2, 3, 4))
            loss = sim + lam * reg
            loss.backward()
            opt.step()
            if it % max(iters // 4, 1) == 0 or it == iters - 1:
                log(f"  nivel {lev} it {it:4d}  sim {sim.item():.5f} "
                    f"reg {reg.item():.5f}")
        disp = disp.detach()
        prev_size = size
        log(f"  nivel {lev} listo en {time.time()-t0:.1f}s")
    return disp


# ------------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", required=True)
    ap.add_argument("--moving", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spacing", type=float, default=1.0)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--lambda-diff", type=float, default=1.0, dest="lam")
    ap.add_argument("--exclude", help="seg cuyas etiquetas se excluyen de la "
                    "similitud (cavidad/residuo: sin correspondencia en preop)")
    ap.add_argument("--exclude-labels", default="1,2")
    ap.add_argument("--exclude-dilate-mm", type=float, default=3.0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

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

    disp = register(f_n, m_n, fov, levels=args.levels, iters=args.iters,
                    lam=args.lam, device=dev)

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
    metrics = {
        "device": dev, "spacing": args.spacing, "dims": [int(d) for d in dims],
        "levels": args.levels, "iters": args.iters, "lambda": args.lam,
        "folding_pct": float((jd_fov <= 0).mean() * 100.0),
        "sd_log_jac": float(np.std(np.log(np.clip(jd_fov, 1e-6, None)))),
        "wall_s": time.time() - t_all,
    }
    (out / "log.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
