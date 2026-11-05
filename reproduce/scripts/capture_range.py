"""capture_range.py -- diagnose WHY an image-driven rigid fails on the two
RESECT "capture range" cases (Case12, Case25) against two healthy controls
(Case18, Case21).

This is a DIAGNOSTIC / SEPARATE ARM.  It touches nothing in the frozen method:
it only *evaluates* the frozen method's own masked MIND-SSC data term over a
grid of rigid poses applied to the moving FLAIR.

Cost reproduced exactly from reg_modal.py / reg_core.py:
    working grid  = load_to_grid(US_after, FLAIR, spacing)   (isotropic, US bbox)
    fov           = (US > 0)  &  ~dilate(cavity==1, 3 mm)
    f_n           = normalize(US, fov);  m_n = normalize(FLAIR, FLAIR>0)
    cost(A)       = sum_x fov(x) * mean_c (MIND_f(x) - MIND_{m∘A}(x))^2 / sum fov

The only deviation is that the moving is resampled *directly from the native
FLAIR* at world coords A·x (instead of warping an already-cropped copy), so
that large perturbations do not sample replicated border garbage.  At A = I the
two agree to ~1e-6 (gate `--check`).

Pose parameterisation (all relative to the FoV-mask centroid c, world mm):
    A(r, t) x  =  R(r) (x - c) + c + t          r = rotation vector, degrees
`A` maps US world -> MR world, the same direction as the oracle Kabsch fit and
as reg_aladin's aff.txt.  TRE(A) = mean_i |A x_us,i - x_mr,i|.

Subcommands: oracle | bench | profiles | grids | global | basin | confounds
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
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reg_core import load_to_grid, resample_to_grid, normalize, mind_ssc  # noqa
from resect_tags import load_tag_pairs  # noqa

WD = Path("/mnt/c/ReMIND_work")
DATA = WD / "data/resect"
OUT = WD / "results/capture_range"
TAGS = Path("/mnt/g/ECOGRAFIA/ReMIND_downloads/RESECT/NIFTI")


# --------------------------------------------------------------- rigid helpers

def fit_rigid(X, Y):
    """Kabsch: R, t minimising |R X + t - Y| (no scaling).  Identical to
    scripts/rigid_decomposition.py:fit_rigid (gated in `oracle`)."""
    xc, yc = X.mean(0), Y.mean(0)
    H = (X - xc).T @ (Y - yc)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, yc - R @ xc


def rodrigues(rv_deg):
    """(...,3) rotation vectors in degrees -> (...,3,3) rotation matrices."""
    rv = np.asarray(rv_deg, dtype=np.float64) * np.pi / 180.0
    th = np.linalg.norm(rv, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, rv / np.maximum(th, 1e-12), 0.0)
    K = np.zeros(rv.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    I = np.broadcast_to(np.eye(3), K.shape)
    s = np.sin(th)[..., None]
    c = (1.0 - np.cos(th))[..., None]
    return I + s * K + c * (K @ K)


def rotvec_deg(R):
    """rotation matrix -> rotation vector in degrees."""
    ang = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if ang < 1e-9:
        return np.zeros(3)
    ax = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    ax = ax / (2.0 * np.sin(ang))
    return ax * np.degrees(ang)


def pose_to_affine(r_deg, t_mm, c):
    """A x = R(r)(x - c) + c + t."""
    R = rodrigues(np.asarray(r_deg, float))
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = c + np.asarray(t_mm, float) - R @ c
    return A


def affine_to_pose(A, c):
    R = A[:3, :3]
    return rotvec_deg(R), (A[:3, :3] @ c + A[:3, 3]) - c


def compose(A2, A1):
    return A2 @ A1


# ------------------------------------------------------------- batched MIND-SSC

def mind_ssc_batch(img, radius=2, dilation=2):
    """MIND-SSC identical to reg_core.mind_ssc but with the variance clamp
    computed PER SAMPLE, so a batch of B poses gives exactly the same value as
    B separate B=1 calls (verified in `bench --check`)."""
    device, dtype = img.device, img.dtype
    six = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1],
                        [1, 1, 2], [2, 1, 1], [1, 2, 1]],
                       dtype=torch.float32, device=device)
    from reg_core import pdist_squared, nn_replication_pad, smooth
    dist = pdist_squared(six.t().unsqueeze(0)).squeeze(0)
    x, y = torch.meshgrid(torch.arange(6, device=device),
                          torch.arange(6, device=device), indexing="ij")
    mask = (x > y).view(-1) & (dist == 2).view(-1)
    idx1 = six.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :].long()
    idx2 = six.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :].long()
    ms1 = torch.zeros(12, 1, 3, 3, 3, device=device, dtype=dtype)
    ms1.view(-1)[torch.arange(12, device=device) * 27
                 + idx1[:, 0] * 9 + idx1[:, 1] * 3 + idx1[:, 2]] = 1
    ms2 = torch.zeros(12, 1, 3, 3, 3, device=device, dtype=dtype)
    ms2.view(-1)[torch.arange(12, device=device) * 27
                 + idx2[:, 0] * 9 + idx2[:, 1] * 3 + idx2[:, 2]] = 1
    rpad = nn_replication_pad(dilation)
    ssd = smooth((F.conv3d(rpad(img), ms1, dilation=dilation)
                  - F.conv3d(rpad(img), ms2, dilation=dilation)) ** 2, radius)
    mind = ssd - ssd.min(dim=1, keepdim=True)[0]
    mv = mind.mean(dim=1, keepdim=True)
    mmean = mv.mean(dim=(1, 2, 3, 4), keepdim=True)          # per sample
    mv = torch.maximum(torch.minimum(mv, mmean * 1000.0), mmean * 0.001)
    mind = torch.exp(-mind / (mv + 1e-8))
    order = torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], device=device)
    return mind[:, order]


# ------------------------------------------------------------------------ case

class CaseCost:
    """Masked MIND-SSC cost of the frozen pipeline as a function of rigid pose."""

    def __init__(self, case, spacing=2.0, device="cuda", exclude_cavity=True,
                 exclude_dilate_mm=3.0):
        self.case, self.spacing = case, spacing
        d = DATA / case
        self.fx_path = d / f"{case}-US-after.nii.gz"
        self.mv_path = d / f"{case}-FLAIR.nii.gz"
        self.cav_path = d / f"{case}-US-after-cavity.nii.gz"
        dev = torch.device(device)
        self.dev = dev

        f_arr, m_arr, grid_aff, dims = load_to_grid(self.fx_path, self.mv_path,
                                                    spacing)
        self.grid_aff, self.dims = grid_aff, tuple(int(x) for x in dims)
        fov = f_arr > 0
        self.fov_raw = fov.copy()
        self.n_fov_raw = int(fov.sum())
        if exclude_cavity and self.cav_path.exists():
            cav = nib.load(str(self.cav_path))
            lab = (np.asarray(cav.dataobj) == 1).astype(np.float32)
            ex = resample_to_grid(cav, grid_aff, dims, mode="nearest",
                                  data=lab) > 0.5
            if ex.any() and exclude_dilate_mm > 0:
                ex = binary_dilation(ex, iterations=max(
                    1, int(round(exclude_dilate_mm / spacing))))
            fov = fov & ~ex
            self.n_cav = int(ex.sum())
        else:
            self.n_cav = 0
        self.fov = fov
        self.n_fov = int(fov.sum())

        # fixed, normalised exactly as the pipeline
        f_n = normalize(f_arr, fov)
        self.f_arr = f_arr
        # moving normalisation constants frozen at identity (pipeline behaviour)
        ref = m_arr[m_arr > 0]
        self.m_lo, self.m_hi = np.percentile(ref, [1.0, 99.0])
        self.m_arr_id = m_arr

        # GPU state
        self.f_t = torch.from_numpy(f_n)[None, None].to(dev)
        self.mind_f = mind_ssc_batch(self.f_t)
        self.k = torch.from_numpy(fov.astype(np.float32))[None, None].to(dev)
        self.ksum = float(self.k.sum())

        mv = nib.load(str(self.mv_path))
        self.mv_shape = mv.shape[:3]
        self.mv_t = torch.from_numpy(
            np.ascontiguousarray(np.asarray(mv.dataobj, dtype=np.float32))
        )[None, None].to(dev)
        self.mv_inv = np.linalg.inv(mv.affine)

        # world coords of the working grid (D,H,W,3)
        ii, jj, kk = np.meshgrid(*[np.arange(n) for n in self.dims],
                                 indexing="ij")
        ijk1 = np.stack([ii, jj, kk, np.ones_like(ii)], -1).astype(np.float64)
        self.X = (ijk1 @ grid_aff.T)[..., :3]
        self.Xt = torch.from_numpy(self.X.astype(np.float32)).to(dev)
        self.center = self.X[fov].mean(0)          # rotation centre = FoV centroid
        self.fov_extent = self.X[fov].max(0) - self.X[fov].min(0)

        # landmarks
        tag = TAGS / case / "Landmarks" / f"{case}-MRI-afterUS.tag"
        if not tag.exists():
            tag = d / f"{case}-MRI-afterUS.tag"
        self.mri_pts, self.us_pts = load_tag_pairs(str(tag))
        R, t = fit_rigid(self.us_pts, self.mri_pts)
        self.A_oracle = np.eye(4)
        self.A_oracle[:3, :3], self.A_oracle[:3, 3] = R, t
        self.r_oracle, self.t_oracle = affine_to_pose(self.A_oracle, self.center)

    # -- metrics ------------------------------------------------------------
    def tre(self, A):
        p = self.us_pts @ A[:3, :3].T + A[:3, 3]
        return float(np.linalg.norm(p - self.mri_pts, axis=1).mean())

    def tre_pose(self, r, t):
        return self.tre(pose_to_affine(r, t, self.center))

    # -- cost ---------------------------------------------------------------
    def _sample(self, Rs, ts):
        """Rs (B,3,3), ts (B,3) numpy -> moving resampled+normalised (B,1,D,H,W)."""
        B = Rs.shape[0]
        Rs_t = torch.from_numpy(np.ascontiguousarray(Rs, dtype=np.float32)).to(self.dev)
        b = np.einsum("bij,j->bi", Rs, -self.center) + self.center + ts   # A x = R x + b
        # world -> moving voxel:  v = Minv[:3,:3] (R x + b) + Minv[:3,3]
        M = self.mv_inv[:3, :3]
        MR = np.einsum("ij,bjk->bik", M, Rs)
        Mb = np.einsum("ij,bj->bi", M, b) + self.mv_inv[:3, 3]
        MR_t = torch.from_numpy(np.ascontiguousarray(MR, dtype=np.float32)).to(self.dev)
        Mb_t = torch.from_numpy(np.ascontiguousarray(Mb, dtype=np.float32)).to(self.dev)
        vox = torch.einsum("bij,dhwj->bdhwi", MR_t, self.Xt) \
            + Mb_t[:, None, None, None, :]
        shp = torch.tensor(self.mv_shape, dtype=torch.float32, device=self.dev)
        norm = 2.0 * vox / (shp - 1.0) - 1.0
        gridc = norm[..., [2, 1, 0]]
        out = F.grid_sample(self.mv_t.expand(B, -1, -1, -1, -1), gridc,
                            mode="bilinear", padding_mode="zeros",
                            align_corners=True)
        out = torch.clamp((out - self.m_lo) / max(self.m_hi - self.m_lo, 1e-6),
                          0.0, 1.0)
        del vox, norm, gridc
        return out

    def cost_poses(self, poses, batch=16):
        """poses: (N,6) [rx,ry,rz deg, tx,ty,tz mm] -> (N,) cost."""
        poses = np.asarray(poses, float).reshape(-1, 6)
        out = np.empty(len(poses), dtype=np.float64)
        for s in range(0, len(poses), batch):
            p = poses[s:s + batch]
            Rs = rodrigues(p[:, :3])
            m = self._sample(Rs, p[:, 3:])
            mm = mind_ssc_batch(m)
            c = (((self.mind_f - mm) ** 2).mean(1, keepdim=True) * self.k)
            c = c.sum(dim=(1, 2, 3, 4)) / (self.ksum + 1e-6)
            out[s:s + batch] = c.detach().cpu().numpy()
            del m, mm, c
        return out

    def cost_affine(self, A, batch=1):
        r, t = affine_to_pose(A, self.center)
        return float(self.cost_poses(np.concatenate([r, t])[None], batch)[0])

    # -- differentiable version (for local descent) --------------------------
    def cost_torch(self, r_t, t_t, mind_f=None, k=None, X=None, ksum=None):
        """r_t (3,) deg, t_t (3,) mm as torch leaf tensors -> scalar cost."""
        mind_f = self.mind_f if mind_f is None else mind_f
        k = self.k if k is None else k
        X = self.Xt if X is None else X
        ksum = self.ksum if ksum is None else ksum
        nrm = torch.sqrt((r_t ** 2).sum() + 1e-12)      # smooth at r = 0
        th = nrm * np.pi / 180.0
        kv = r_t / nrm
        K = torch.zeros(3, 3, device=self.dev, dtype=torch.float32)
        K = torch.stack([
            torch.stack([torch.zeros((), device=self.dev), -kv[2], kv[1]]),
            torch.stack([kv[2], torch.zeros((), device=self.dev), -kv[0]]),
            torch.stack([-kv[1], kv[0], torch.zeros((), device=self.dev)])])
        I = torch.eye(3, device=self.dev)
        R = I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
        c = torch.tensor(self.center, dtype=torch.float32, device=self.dev)
        b = -(R @ c) + c + t_t
        M = torch.tensor(self.mv_inv[:3, :3], dtype=torch.float32, device=self.dev)
        m3 = torch.tensor(self.mv_inv[:3, 3], dtype=torch.float32, device=self.dev)
        MR, Mb = M @ R, M @ b + m3
        vox = torch.einsum("ij,dhwj->dhwi", MR, X) + Mb
        shp = torch.tensor(self.mv_shape, dtype=torch.float32, device=self.dev)
        norm = 2.0 * vox / (shp - 1.0) - 1.0
        gridc = norm[..., [2, 1, 0]][None]
        out = F.grid_sample(self.mv_t, gridc, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        out = torch.clamp((out - self.m_lo) / max(self.m_hi - self.m_lo, 1e-6),
                          0.0, 1.0)
        mm = mind_ssc_batch(out)
        return (((mind_f - mm) ** 2).mean(1, keepdim=True) * k).sum() / (ksum + 1e-6)

    def pyramid(self, factor):
        """Downsampled (mind_f, k, X, ksum) for a coarse level of the descent."""
        if factor == 1:
            return self.mind_f, self.k, self.Xt, self.ksum
        size = [max(int(round(d / factor)), 8) for d in self.dims]
        f = F.interpolate(self.f_t, size=size, mode="trilinear", align_corners=True)
        k = F.interpolate(self.k, size=size, mode="trilinear", align_corners=True)
        mind_f = mind_ssc_batch(f)
        Xc = F.interpolate(self.Xt.permute(3, 0, 1, 2)[None], size=size,
                           mode="trilinear", align_corners=True)[0].permute(1, 2, 3, 0)
        return mind_f, k, Xc.contiguous(), float(k.sum())

    def descend(self, r0, t0, levels=(4, 2, 1), iters=(200, 150, 120),
                lr_t=1.0, lr_r=0.4, decay=0.1, cache=None):
        """Multi-resolution Adam local optimisation of the 6 rigid parameters,
        started at (r0, t0).  lr scales as sqrt(level) and decays x`decay`
        linearly within each level.  **The best-cost iterate of each level is
        kept** and carried to the next, so a diverging trajectory can never
        return a pose worse than its own start (verified: descend() never
        raises the cost).  Returns (r, t, cost)."""
        cur_r = np.asarray(r0, np.float32).copy()
        cur_t = np.asarray(t0, np.float32).copy()
        cache = {} if cache is None else cache
        for lev, nit in zip(levels, iters):
            if lev not in cache:
                cache[lev] = self.pyramid(lev)
            mf, k, X, ks = cache[lev]
            r = torch.tensor(cur_r, device=self.dev, requires_grad=True)
            t = torch.tensor(cur_t, device=self.dev, requires_grad=True)
            s = float(lev) ** 0.5
            opt = torch.optim.Adam([{"params": [t], "lr": lr_t * s},
                                    {"params": [r], "lr": lr_r * s}])
            sch = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda i, n=nit: 1.0 - (1.0 - decay) * i / max(n - 1, 1))
            best = (float(self.cost_torch(r, t, mf, k, X, ks)),
                    cur_r.copy(), cur_t.copy())
            for _ in range(nit):
                opt.zero_grad()
                loss = self.cost_torch(r, t, mf, k, X, ks)
                loss.backward()
                opt.step()
                sch.step()
                lv = float(loss)
                if lv < best[0]:
                    best = (lv, r.detach().cpu().numpy().copy(),
                            t.detach().cpu().numpy().copy())
            cur_r, cur_t = best[1], best[2]
        rr, tt = cur_r.astype(float), cur_t.astype(float)
        return rr, tt, float(self.cost_poses(np.concatenate([rr, tt])[None])[0])


# -------------------------------------------------------------------- distances

def pose_dist(r1, t1, r2, t2, radius):
    """Composite distance: translational (mm) + rotational converted to mm at
    `radius` (the RMS landmark/FoV lever arm).  Returns (d_t, d_r_deg, d_mm)."""
    R1, R2 = rodrigues(r1), rodrigues(r2)
    dR = R2 @ R1.T
    ang = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    dt = float(np.linalg.norm(np.asarray(t2) - np.asarray(t1)))
    return dt, float(ang), float(np.hypot(dt, np.radians(ang) * radius))
