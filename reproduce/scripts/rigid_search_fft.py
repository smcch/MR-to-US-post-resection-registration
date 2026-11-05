"""rigid_search_fft.py -- FFT-accelerated global rigid search.

Drop-in fast replacement for the STAGE-1 (candidate generation) half of
`rigid_search.py`.  It produces the same four files per case
(`search.json`, `starts.csv`, `aff.txt`, `aff_gated.txt`) with the same schema,
so `rs_summary.py` and every other downstream consumer needs no change.

`rigid_search.py` and `capture_range.py` are NOT touched or imported-and-mutated
by this file; they are imported read-only so that the cost, the masking, the
pose convention, the candidate-selection rule and the whole stage-2 refinement
are literally the same code.

--------------------------------------------------------------------- why FFT

The frozen data term is

    cost(R, t) = 1/(ksum+eps) * sum_x k(x) * 1/12 sum_c (A_c(x) - B_c(x;R,t))^2

with A = MIND-SSC of the fixed US on the working grid, k = the fixed FoV mask,
and B(.;R,t) = MIND-SSC of the moving MR resampled at world  R(x-c)+c+t.

Expanding the square,

    12 * ksum * cost(R,t) = SA + T2(t) - 2 T3(t)
    SA    = sum_c sum_x k(x) A_c(x)^2                       (independent of t)
    T2(t) = sum_x k(x) * [sum_c B_c(x+u)^2]                 (u = R^T t)
    T3(t) = sum_c sum_x [k A_c](x) * B_c(x+u)

T2 and T3 are cross-correlations, so for a FIXED rotation every integer-voxel
translation is obtained in ONE FFT pass instead of one cost evaluation each.

Shift bookkeeping.  The moving sampled at grid point x for pose (R,t) is
mv(R(x-c)+c+t).  Writing H_R(y) = mv(R(y-c)+c) for the t=0 pose, we need
H_R(x+u) with R u = t, i.e.  u = R^T t.  So the shift that the FFT sweeps lives
in the *sampling grid* frame and the translations actually evaluated are

    t = spacing * R n,      n in Z^3,  |n|_inf <= p .

That is a rotated cubic lattice of translations, not the axis-aligned lattice of
the brute force.  It covers the same ball with the same density (|t| = spacing*|n|),
which is all a candidate generator needs -- and stage 2 refines continuously
from each candidate anyway, so the lattice orientation cannot bias the answer.

--------------------------------------------------------- why NOT Padfield MNCC

Padfield (IEEE TIP 2012) is required when the *denominator* of the similarity
depends on the translation, i.e. when both images carry a mask and the overlap
support moves with t.  That is not the situation here, and adopting it would
silently change the cost and break comparability with the frozen arm:

  * the mask k = (US>0) & ~dilate(cavity) lives ONLY in the fixed frame and does
    not move with t;
  * the normaliser in the frozen data term is the CONSTANT `ksum` (see
    `capture_range.CaseCost.cost_poses`: `... .sum() / (self.ksum + 1e-6)`),
    not the per-translation overlap count;
  * the moving is resampled from the native MR with `padding_mode="zeros"`, so
    a translation that pushes the FoV off the MR contributes real zeros to the
    sum -- exactly as the brute force does. Renormalising by an overlap count
    would *reward* poses that fall off the image, which is the opposite of what
    the frozen cost does.

So the correct masked formulation here is the explicit one: correlate the mask
k against the moving energy (the T2 term above) to get the per-translation
masked energy, and keep the fixed constant denominator.  The mask enters both
correlations exactly, no approximation.  (The moving-side "mask" is the implicit
zero padding, which the brute force also has, so it needs no separate handling.)

-------------------------------------------------- MIND under rotation (item 2)

MIND-SSC channels are directional: the 12 channels are differences between
specific pairs of the 6-neighbourhood offsets, so rotating the image does NOT
map the descriptor field to a resampled copy of itself.  Two options:

  (a) SHORTCUT  -- compute MIND once in the moving frame and resample the
      12-channel field through each rotation (what the brief proposed).
  (b) PER-ROTATION -- resample the moving IMAGE through each rotation and
      recompute MIND on it, once per rotation.

The FFT collapses the search to ~125 rotations, so (b) costs 125 MIND
evaluations instead of 166,375 -- a rounding error in the total runtime -- while
being exact with respect to the brute force's interpolate-then-MIND order.
(b) is therefore the DEFAULT here.  (a) is implemented behind
`--mind-mode resample` purely so that its error can be measured; `--check`
reports it.  Note that for the |r| <= 20 deg rotations searched, the nearest
signed axis permutation is the identity, so there is no channel permutation to
apply -- the shortcut's error is the genuine directional distortion of the
descriptor, and it is reported, not hidden.

Residual approximation of the coarse FFT stage (all measured by `--check`):
  1. MIND's variance clamp uses a per-sample global mean.  The brute force takes
     it over the working grid at that pose; here it is taken over the central
     working-grid crop of the extended volume (i.e. the t=0 crop) and reused for
     every shift.  `--mmean-mode ext` uses the whole extended volume instead.
  2. MIND border effects: the brute force replication-pads the working grid,
     while the extended volume supplies true neighbours.  Differences are
     confined to a 4-voxel shell (radius 2 + dilation 2) at the working-grid
     border.
Neither can limit the final answer: stage 2 re-optimises with the EXACT cost
(`PoseCost.descend` / `cost_poses`, recomputing MIND per pose) at 1 mm.

-------------------------------------------- the coarse stage MUST NOT decide

This is a hard design constraint, not a caution.  The capture-range diagnosis
established that the reliability of this cost depends critically on the working
resolution: at 2 mm the landscape is misleading, while at 1 mm the ordering
inverts and the true pose carries the lowest cost.  Measured here on Case12,
at 2 mm:

    exhaustive 166,375-pose grid  ->  global minimum at TRE 16.57 mm
    this FFT sweep, 1,953,125 poses -> global minimum at TRE  5.54 mm
    stage 2, exact cost at 1 mm     ->              answer at TRE  1.81 mm

The denser FFT lattice finds a strictly lower coarse cost than the brute-force
grid (0.11947 vs 0.12088) and a better coarse TRE, but 5.54 mm is still not an
answer.  So stage 1 here is a CANDIDATE GENERATOR and nothing else: it exists to
put the true basin somewhere in the top-K short list, and it is scored only by
whether it does that.  Everything that is reported comes from stage 2, at 1 mm,
with the exact recomputed-MIND cost.  Making stage 1 cheap is therefore free
accuracy: we can afford a ~12x denser translation lattice than the brute force
purely to improve RECALL of the true basin, and let the 1 mm stage arbitrate.

Usage
    python rigid_search_fft.py --selftest
    python rigid_search_fft.py --check Case12 --coarse-spacing 2.0
    python rigid_search_fft.py --dataset resect --case Case12 Case25 --tag _fft
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reg_core import pdist_squared, nn_replication_pad, smooth            # noqa
from capture_range import (mind_ssc_batch, rodrigues, rotvec_deg,         # noqa
                           pose_to_affine, affine_to_pose, pose_dist)
from rigid_search import (WD, PoseCost, case_paths, case_landmarks, CFG,  # noqa
                          RESECT_TUNE, RESECT_TEST, REMIND_PIDS)

OUT = WD / "results/capture_range/armA_fft"
SEED = 20260811


# ------------------------------------------------------- MIND on a bigger box

def mind_ssc_ext(img, radius=2, dilation=2, stat=None):
    """VERBATIM copy of `capture_range.mind_ssc_batch` with ONE change: the
    per-sample mean used for the variance clamp may be taken over a sub-box
    `stat` (a tuple of three slices) instead of the whole volume.

    With `stat=None` it is bit-identical to `mind_ssc_batch` (asserted in
    `--selftest`).  With `stat` = the central working-grid crop it reproduces
    the clamp statistic the brute force would have used at t = 0.
    """
    device, dtype = img.device, img.dtype
    six = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1],
                        [1, 1, 2], [2, 1, 1], [1, 2, 1]],
                       dtype=torch.float32, device=device)
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
    if stat is None:
        mmean = mv.mean(dim=(1, 2, 3, 4), keepdim=True)
    else:
        mmean = mv[:, :, stat[0], stat[1], stat[2]].mean(
            dim=(1, 2, 3, 4), keepdim=True)
    mv = torch.maximum(torch.minimum(mv, mmean * 1000.0), mmean * 0.001)
    mind = torch.exp(-mind / (mv + 1e-8))
    order = torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], device=device)
    return mind[:, order]


# ------------------------------------------------------------ FFT correlation

def corr_fft(small, big_f, shape):
    """circular cross-correlation  C[m] = sum_j small[j] * big[j+m].

    `small` is already zero-padded to `shape`; `big_f` is rfftn(big).  Returns
    the full `shape` array; entry [m mod N] is the correlation at shift m.
    Valid without wraparound for every shift the caller padded for.
    """
    sf = torch.fft.rfftn(small, dim=(-3, -2, -1)).conj()
    return torch.fft.irfftn(sf * big_f, s=shape, dim=(-3, -2, -1))


class FFTCoarse:
    """Stage-1 candidate generator.  Wraps a `PoseCost` at the coarse spacing
    and adds the extended-grid / Fourier machinery."""

    def __init__(self, cc, pad_vox, mind_mode="recompute", mmean_mode="crop"):
        self.cc = cc
        self.p = int(pad_vox)
        self.mind_mode = mind_mode
        self.mmean_mode = mmean_mode
        p, dev = self.p, cc.dev
        D, H, W = cc.dims
        self.edims = (D + 2 * p, H + 2 * p, W + 2 * p)
        self.crop = (slice(p, p + D), slice(p, p + H), slice(p, p + W))

        # world coords of the extended grid: origin shifted by -p voxels
        sp = cc.spacing
        idx = [np.arange(n, dtype=np.float64) - p for n in self.edims]
        ii, jj, kk = np.meshgrid(*idx, indexing="ij")
        lo = cc.grid_aff[:3, 3]
        Xe = np.stack([lo[0] + sp * ii, lo[1] + sp * jj, lo[2] + sp * kk], -1)
        self.Xe = torch.from_numpy(Xe.astype(np.float32)).to(dev)

        # ---- fixed side, computed once -------------------------------------
        A = cc.mind_f[0]                                   # (12,D,H,W) exact
        k = cc.k[0, 0]                                     # (D,H,W)
        self.SA = float((k[None] * A ** 2).sum())
        Pk = torch.zeros(self.edims, device=dev)
        Pk[self.crop] = k
        PkA = torch.zeros((12,) + self.edims, device=dev)
        PkA[(slice(None),) + self.crop] = k[None] * A
        self.Fk = torch.fft.rfftn(Pk, dim=(-3, -2, -1)).conj()
        self.FkA = torch.fft.rfftn(PkA, dim=(-3, -2, -1)).conj()
        del Pk, PkA

        # moving MIND in its own frame, for the --mind-mode resample shortcut
        self._B0 = None

    # -- moving resampled on the extended grid at rotation R, t = 0 ----------
    def _sample_ext(self, R):
        cc = self.cc
        R = np.asarray(R, dtype=np.float64)
        b = -R @ cc.center + cc.center                     # t = 0
        M = cc.mv_inv[:3, :3]
        MR = torch.from_numpy(np.ascontiguousarray(
            M @ R, dtype=np.float32)).to(cc.dev)
        Mb = torch.from_numpy(np.ascontiguousarray(
            M @ b + cc.mv_inv[:3, 3], dtype=np.float32)).to(cc.dev)
        vox = torch.einsum("ij,dhwj->dhwi", MR, self.Xe) + Mb
        shp = torch.tensor(cc.mv_shape, dtype=torch.float32, device=cc.dev)
        gridc = (2.0 * vox / (shp - 1.0) - 1.0)[..., [2, 1, 0]][None]
        out = F.grid_sample(cc.mv_t, gridc, mode="bilinear",
                            padding_mode="zeros", align_corners=True)
        out = torch.clamp((out - cc.m_lo) / max(cc.m_hi - cc.m_lo, 1e-6),
                          0.0, 1.0)
        del vox, gridc
        return out

    def _mind_ext(self, R):
        """(12, ED,EH,EW) MIND field of the moving at rotation R, t = 0."""
        stat = None if self.mmean_mode == "ext" else self.crop
        if self.mind_mode == "recompute":
            return mind_ssc_ext(self._sample_ext(R), stat=stat)[0]
        # ---- shortcut: rotate the descriptor field, do not re-derive it -----
        if self._B0 is None:
            self._B0 = mind_ssc_ext(self._sample_ext(np.eye(3)), stat=stat)
        cc = self.cc
        R = np.asarray(R, dtype=np.float64)
        # sample B0 (defined on the extended grid) at extended-grid coords of
        # the rotated points: y = R(x-c)+c, expressed in extended voxel index
        Rt = torch.from_numpy(np.ascontiguousarray(R, np.float32)).to(cc.dev)
        c = torch.tensor(cc.center, dtype=torch.float32, device=cc.dev)
        y = torch.einsum("ij,dhwj->dhwi", Rt, self.Xe - c) + c
        lo = torch.tensor(cc.grid_aff[:3, 3], dtype=torch.float32,
                          device=cc.dev)
        vox = (y - lo) / cc.spacing + self.p
        shp = torch.tensor(self.edims, dtype=torch.float32, device=cc.dev)
        gridc = (2.0 * vox / (shp - 1.0) - 1.0)[..., [2, 1, 0]][None]
        return F.grid_sample(self._B0, gridc, mode="bilinear",
                             padding_mode="zeros", align_corners=True)[0]

    # -- all integer shifts for one rotation, in one FFT pass ----------------
    def costs_for_rotation(self, R):
        """-> (S,S,S) cost array, S = 2p+1, index [a,b,c] is shift
        n = (a-p, b-p, c-p) i.e. translation t = spacing * R @ n."""
        p, S = self.p, 2 * self.p + 1
        B = self._mind_ext(R)                              # (12,ED,EH,EW)
        Bf = torch.fft.rfftn(B, dim=(-3, -2, -1))
        T3 = torch.fft.irfftn((self.FkA * Bf).sum(0), s=self.edims,
                              dim=(-3, -2, -1))
        Sq = (B ** 2).sum(0)
        T2 = torch.fft.irfftn(self.Fk * torch.fft.rfftn(Sq, dim=(-3, -2, -1)),
                              s=self.edims, dim=(-3, -2, -1))
        num = (self.SA + T2 - 2.0 * T3) / 12.0
        cost = num / (self.cc.ksum + 1e-6)
        sh = torch.arange(-p, p + 1, device=cost.device)
        out = cost[sh[:, None, None] % self.edims[0],
                   sh[None, :, None] % self.edims[1],
                   sh[None, None, :] % self.edims[2]]
        del B, Bf, T3, T2, Sq, num, cost
        return out.reshape(S, S, S)

    def run(self, rots, log=print):
        """rots (NR,3) rotation vectors in deg -> (cost (NR,S,S,S), T (NR,S^3,3))
        with T the world translations actually evaluated."""
        p, S = self.p, 2 * self.p + 1
        n = np.stack(np.meshgrid(*[np.arange(-p, p + 1)] * 3, indexing="ij"),
                     -1).reshape(-1, 3).astype(np.float64)
        C = np.empty((len(rots), S, S, S), dtype=np.float32)
        T = np.empty((len(rots), S ** 3, 3), dtype=np.float32)
        Rs = rodrigues(rots)
        for i in range(len(rots)):
            C[i] = self.costs_for_rotation(Rs[i]).float().cpu().numpy()
            T[i] = (self.cc.spacing * (n @ Rs[i].T)).astype(np.float32)
            if log and (i % 25 == 0 or i == len(rots) - 1):
                log(f"    rotation {i+1}/{len(rots)}")
        return C, T


# ------------------------------------------------------ candidate short-list

def select_candidates(cost, T, rots, lever, topk, sep):
    """Greedy top-K with a minimum composite pose separation.

    Identical rule to `rigid_search.search`: walk the candidates in increasing
    cost and accept one if it is > `sep` mm (composite `pose_dist`) from every
    already-accepted candidate.  Implemented as "accept best alive, then kill
    everything within `sep` of it", which yields the same sequence but is
    vectorised, so it stays cheap on the ~2e6 poses the FFT stage produces.
    Returns (poses (K,6), costs (K,)).
    """
    NR, S = len(rots), cost.shape[1]
    flat = cost.reshape(NR, -1)
    Rm = rodrigues(rots)
    tr = np.einsum("aij,bij->ab", Rm, Rm)                     # trace(Ra Rb^T)
    ANG = np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
    ri = np.repeat(np.arange(NR), S ** 3)
    c = flat.reshape(-1).astype(np.float64)
    t = T.reshape(-1, 3).astype(np.float64)
    alive = np.ones(len(c), dtype=bool)
    sel_p, sel_c = [], []
    while len(sel_p) < topk and alive.any():
        idx = np.flatnonzero(alive)
        j = idx[np.argmin(c[idx])]
        sel_p.append(np.concatenate([rots[ri[j]], t[j]]))
        sel_c.append(c[j])
        d_t = np.linalg.norm(t - t[j], axis=1)
        d_r = np.radians(ANG[ri[j]][ri]) * lever
        alive &= np.hypot(d_t, d_r) > sep
    return np.array(sel_p), np.array(sel_c)


# --------------------------------------------------------------------- search

def rot_grid(r_max, r_step):
    """Same rotation grid, same lexicographic order, as
    `rigid_search.coarse_poses`."""
    rv = np.arange(-r_max, r_max + 1e-9, r_step)
    return np.stack(np.meshgrid(rv, rv, rv, indexing="ij"), -1).reshape(-1, 3)


def search(name, dataset, cfg, device="cuda", log=print, out_root=None):
    t_all = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    fx, mv, ex, exl = case_paths(dataset, name)
    us_pts, mri_pts = case_landmarks(dataset, name)
    out_root = OUT if out_root is None else out_root

    # ---- stage 1: FFT over translations, loop over rotations ---------------
    t0 = time.time()
    cc = PoseCost(name, fx, mv, ex, exl, spacing=cfg["coarse_spacing"],
                  device=device, us_pts=us_pts, mri_pts=mri_pts)
    lever = cc.lever
    p = int(np.floor(cfg["t_max"] / cfg["coarse_spacing"]))
    rots = rot_grid(cfg["r_max"], cfg["r_step"])
    fc = FFTCoarse(cc, p, mind_mode=cfg.get("mind_mode", "recompute"),
                   mmean_mode=cfg.get("mmean_mode", "crop"))
    C, T = fc.run(rots, log=None)
    n_eval = int(C.size)
    t_coarse = time.time() - t0
    log(f"  coarse-FFT: {n_eval} poses ({len(rots)} rot x {(2*p+1)**3} trans) "
        f"@{cfg['coarse_spacing']} mm (grid {cc.dims}+2*{p}, fov {cc.n_fov}) "
        f"in {t_coarse:.1f}s ({n_eval/max(t_coarse,1e-9):.0f}/s); "
        f"lever {lever:.1f} mm; min {C.min():.6f}")

    P, cost_c = select_candidates(C, T, rots, lever, cfg["topk"], cfg["sep"])
    grid_npz = dict(cost=C, trans=T, rots=rots.astype(np.float32),
                    spacing=np.float32(cfg["coarse_spacing"]), pad=np.int32(p),
                    cand_pose=P.astype(np.float32),
                    cand_cost=cost_c.astype(np.float32))
    del fc, cc, C, T
    torch.cuda.empty_cache()

    # ---- stage 2: EXACT cost, identical to rigid_search.py -----------------
    t0 = time.time()
    ff = PoseCost(name, fx, mv, ex, exl, spacing=cfg["fine_spacing"],
                  device=device, us_pts=us_pts, mri_pts=mri_pts)
    zero = np.zeros(6)
    c_id = float(ff.cost_poses(zero[None], batch=1)[0])
    tre_id = ff.tre(np.eye(4))
    cache, rows = {}, []
    # (tag, pose, cost0) -- cost0 carried explicitly so that it stays correct
    # if the start list is filtered by the tiered mode below.
    starts = [("identity", zero, c_id)] + [(f"coarse{i}", P[i],
                                            float(cost_c[i]))
                                           for i in range(len(P))]
    t_pre = 0.0
    if cfg.get("refine_mode", "full") == "tiered":
        # cheap 4->2 mm pass on every start, then the full 4->2->1 schedule on
        # the best `tier_k` + identity.  OFF by default; validated separately.
        t0p = time.time()
        pre = [ff.descend(s[1][:3], s[1][3:], levels=cfg["levels"][:2],
                          iters=cfg["iters"][:2], cache=cache)[2]
               for s in starts]
        keep = {0} | set(int(i) for i in
                         np.argsort(pre)[:cfg.get("tier_k", 6)])
        starts = [starts[i] for i in sorted(keep)]
        t_pre = time.time() - t0p
        log(f"  tiered: {len(pre)} -> {len(starts)} full refinements "
            f"({t_pre:.0f}s prescreen)")

    for si, (tag, p0, c0) in enumerate(starts):
        r1, t1, c1 = ff.descend(p0[:3], p0[3:], levels=cfg["levels"],
                                iters=cfg["iters"], cache=cache)
        A1 = pose_to_affine(r1, t1, ff.center)
        rows.append(dict(start=tag, cost0=c0,
                         rx=r1[0], ry=r1[1], rz=r1[2],
                         tx=t1[0], ty=t1[1], tz=t1[2], cost=c1,
                         tre=ff.tre(A1),
                         rot_deg=float(np.linalg.norm(rotvec_deg(A1[:3, :3]))),
                         trans_mm=float(np.linalg.norm(t1)),
                         bg=ff.bg_fraction(r1, t1)))
        log(f"  start {si:2d} {tag:9s} -> cost {c1:.6f} TRE {rows[-1]['tre']:6.2f} "
            f"rot {rows[-1]['rot_deg']:5.1f} deg |t| {rows[-1]['trans_mm']:5.1f} "
            f"bg {rows[-1]['bg']:.4f}")
    t_fine = time.time() - t0

    # ---- selection / gate: identical to rigid_search.py --------------------
    D = pd.DataFrame(rows)
    valid = D[D.bg < cfg["max_bg"]]
    pool = valid if len(valid) else D
    ib = int(pool.cost.values.argmin())
    best = pool.iloc[ib]
    br = np.array([best.rx, best.ry, best.rz])
    bt = np.array([best.tx, best.ty, best.tz])
    A = pose_to_affine(br, bt, ff.center)

    dists = np.array([pose_dist(br, bt, np.array([v.rx, v.ry, v.rz]),
                                np.array([v.tx, v.ty, v.tz]), lever)[2]
                      for v in pool.itertuples()])
    n_agree = int((dists < 3.0).sum())
    out_cluster = pool.cost.values[dists >= 3.0]
    margin = (float((out_cluster.min() - best.cost) / best.cost * 100.0)
              if len(out_cluster) else float("nan"))
    id_row = D.iloc[0]
    gain_vs_id = float((id_row.cost - best.cost) / id_row.cost * 100.0)
    fired = bool(best.rot_deg <= cfg["rot_cap"]
                 and best.trans_mm <= cfg["trans_cap"])
    A_gated = A if fired else np.eye(4)

    res = {
        "case": name, "dataset": dataset, "cfg": cfg,
        "n_coarse": n_eval, "n_starts": int(len(starts)),
        "dims_fine": list(ff.dims), "n_fov": ff.n_fov, "n_excluded": ff.n_cav,
        "lever_mm": lever,
        "cost_identity": c_id, "tre_identity": tre_id,
        "cost_identity_refined": float(id_row.cost),
        "tre_identity_refined": float(id_row.tre),
        "search_cost": float(best.cost), "search_tre": float(best.tre),
        "rot_deg": float(best.rot_deg), "trans_mm": float(best.trans_mm),
        "bg": float(best.bg), "n_agree_3mm": n_agree,
        "margin_pct": margin, "cost_gain_vs_identity_pct": gain_vs_id,
        "gate_fired": fired, "gate_rot_cap": cfg["rot_cap"],
        "gate_trans_cap": cfg["trans_cap"],
        "tre_gated": float(best.tre) if fired else tre_id,
        "A": A.tolist(), "A_gated": A_gated.tolist(),
        "t_coarse_s": t_coarse, "t_fine_s": t_fine,
        "wall_s": time.time() - t_all,
        "engine": "fft", "seed": SEED, "t_prescreen_s": t_pre,
        "winning_start": str(best.start),
        "n_rot": int(len(rots)), "n_trans_per_rot": int((2 * p + 1) ** 3),
        "coarse_min_cost": float(grid_npz["cost"].min()),
    }
    if us_pts is not None:
        res["tre_oracle"] = ff.tre(ff.A_oracle)
        res["oracle_rot_deg"] = float(np.linalg.norm(rotvec_deg(
            ff.A_oracle[:3, :3])))
        res["d_oracle_mm"] = pose_dist(br, bt, ff.r_oracle, ff.t_oracle,
                                       lever)[2]
    d = out_root / dataset / name
    d.mkdir(parents=True, exist_ok=True)
    D.to_csv(d / "starts.csv", index=False)
    (d / "search.json").write_text(json.dumps(res, indent=2))
    np.savetxt(d / "aff.txt", A, fmt="%.10f")
    np.savetxt(d / "aff_gated.txt", A_gated, fmt="%.10f")
    np.savez_compressed(d / "coarse_fft.npz", **grid_npz)
    return res


# ------------------------------------------------------------------- selftest

def selftest(device="cuda"):
    """Machine-checkable identities, no data needed."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device="cpu").manual_seed(SEED)

    # 1. mind_ssc_ext(stat=None) == mind_ssc_batch, bitwise
    v = torch.rand(2, 1, 20, 22, 24, generator=g).to(dev)
    a, b = mind_ssc_batch(v), mind_ssc_ext(v, stat=None)
    assert torch.equal(a, b), "mind_ssc_ext != mind_ssc_batch"
    print(f"  [ok] mind_ssc_ext(stat=None) bit-identical to mind_ssc_batch")

    # 2. corr_fft == direct correlation, for every shift it claims to cover
    D, p = (7, 8, 9), 2
    E = tuple(d + 2 * p for d in D)
    P = torch.rand(*D, generator=g).to(dev)
    Q = torch.rand(*E, generator=g).to(dev)
    Pp = torch.zeros(E, device=dev)
    Pp[p:p + D[0], p:p + D[1], p:p + D[2]] = P
    C = corr_fft(Pp, torch.fft.rfftn(Q, dim=(-3, -2, -1)), E)
    err = 0.0
    for a in range(-p, p + 1):
        for b_ in range(-p, p + 1):
            for c_ in range(-p, p + 1):
                ref = (P * Q[p + a:p + a + D[0], p + b_:p + b_ + D[1],
                             p + c_:p + c_ + D[2]]).sum()
                err = max(err, abs(float(C[a % E[0], b_ % E[1], c_ % E[2]] - ref)))
    assert err < 1e-3, f"corr_fft wrong, max abs err {err}"
    print(f"  [ok] corr_fft == direct correlation over all {(2*p+1)**3} shifts "
          f"(max abs err {err:.2e})")
    print("SELFTEST OK")


# ---------------------------------------------------------------------- check

def check(case, dataset, cfg, n_probe=192, device="cuda"):
    """Quantify the coarse-stage approximation against the EXACT cost, and the
    descriptor-resampling shortcut against per-rotation recomputation."""
    rng = np.random.default_rng(SEED)
    fx, mv, ex, exl = case_paths(dataset, case)
    us, mri = case_landmarks(dataset, case)
    cc = PoseCost(case, fx, mv, ex, exl, spacing=cfg["coarse_spacing"],
                  device=device, us_pts=us, mri_pts=mri)
    p = int(np.floor(cfg["t_max"] / cfg["coarse_spacing"]))
    rots = rot_grid(cfg["r_max"], cfg["r_step"])
    S = 2 * p + 1
    rep = {"case": case, "spacing": cfg["coarse_spacing"], "pad_vox": p,
           "dims": list(cc.dims), "n_fov": cc.n_fov, "n_rot": len(rots),
           "n_trans_per_rot": S ** 3}

    # random (rotation, integer shift) probes
    ri = rng.integers(0, len(rots), n_probe)
    nn = rng.integers(-p, p + 1, (n_probe, 3))
    Rm = rodrigues(rots)

    for mode in ("recompute", "resample"):
        fc = FFTCoarse(cc, p, mind_mode=mode,
                       mmean_mode=cfg.get("mmean_mode", "crop"))
        t0 = time.time()
        cache = {}
        pred = np.empty(n_probe)
        for a in range(n_probe):
            i = int(ri[a])
            if i not in cache:
                cache[i] = fc.costs_for_rotation(Rm[i]).float().cpu().numpy()
            pred[a] = cache[i][tuple(nn[a] + p)]
        # EXACT cost at exactly those poses:  t = spacing * R n
        P = np.empty((n_probe, 6))
        for a in range(n_probe):
            P[a, :3] = rots[ri[a]]
            P[a, 3:] = cfg["coarse_spacing"] * (Rm[int(ri[a])] @ nn[a])
        exact = cc.cost_poses(P, batch=24)
        rel = np.abs(pred - exact) / np.abs(exact)
        # does the approximation preserve the RANKING (what stage 1 is for)?
        rho = float(np.corrcoef(pred, exact)[0, 1])
        k = max(1, n_probe // 10)
        top_p = set(np.argsort(pred)[:k])
        top_e = set(np.argsort(exact)[:k])
        rep[mode] = {
            "n_probe": n_probe,
            "rel_err_median": float(np.median(rel)),
            "rel_err_p95": float(np.percentile(rel, 95)),
            "rel_err_max": float(rel.max()),
            "abs_err_max": float(np.abs(pred - exact).max()),
            "cost_range": [float(exact.min()), float(exact.max())],
            "pearson_r": rho,
            "top10pct_overlap": len(top_p & top_e) / k,
            "s_per_rotation": (time.time() - t0) / max(len(cache), 1),
        }
        print(f"  [{mode:9s}] rel err  med {rep[mode]['rel_err_median']:.3e} "
              f"p95 {rep[mode]['rel_err_p95']:.3e} "
              f"max {rep[mode]['rel_err_max']:.3e}   r={rho:.6f}  "
              f"top10%%-overlap {rep[mode]['top10pct_overlap']:.2f}")
        del fc
        torch.cuda.empty_cache()

    # full FFT sweep: where does the coarse stage put its global minimum, and
    # what is the EXACT cost there?
    fc = FFTCoarse(cc, p, mind_mode="recompute",
                   mmean_mode=cfg.get("mmean_mode", "crop"))
    t0 = time.time()
    C, T = fc.run(rots, log=None)
    rep["sweep_s"] = time.time() - t0
    rep["sweep_n"] = int(C.size)
    j = int(np.argmin(C))
    ir, it = np.unravel_index(j, C.shape[:1] + (S ** 3,))[0], j % (S ** 3)
    bp = np.concatenate([rots[ir], T[ir, it]])
    rep["fft_argmin_pose"] = bp.tolist()
    rep["fft_argmin_cost"] = float(C.reshape(-1)[j])
    rep["fft_argmin_cost_exact"] = float(cc.cost_poses(bp[None], batch=1)[0])
    rep["fft_argmin_tre"] = cc.tre(pose_to_affine(bp[:3], bp[3:], cc.center))

    # independent ground truth: the exhaustive EXACT 6-D grid, if present
    gp = WD / f"results/capture_range/{case}/grid6d.npz"
    if gp.exists():
        g = np.load(str(gp))
        jb = int(np.argmin(g["cost"]))
        gp_pose = g["pose"][jb].astype(float)
        rep["grid6d"] = {
            "n": int(len(g["cost"])), "min_cost": float(g["cost"][jb]),
            "min_pose": gp_pose.tolist(), "min_tre": float(g["tre"][jb]),
            "fft_min_beats_grid6d_min": bool(rep["fft_argmin_cost_exact"]
                                             <= float(g["cost"][jb])),
            "d_mm_fft_vs_grid6d": pose_dist(bp[:3], bp[3:], gp_pose[:3],
                                            gp_pose[3:], cc.lever)[2],
        }
        # R = I slice: our lattice is axis-aligned there, so translations that
        # are multiples of both the FFT spacing and the 5 mm grid step coincide
        # exactly -> a like-for-like check against exhaustive EXACT costs.
        iz = int(np.argmin(np.abs(rots).sum(1)))
        tv = g["trans_vals"]
        hits = [(v, int(round(v / cfg["coarse_spacing"])))
                for v in tv if abs(v / cfg["coarse_spacing"]
                                   - round(v / cfg["coarse_spacing"])) < 1e-9
                and abs(round(v / cfg["coarse_spacing"])) <= p]
        if hits and np.allclose(rots[iz], 0):
            pr, ex_ = [], []
            gpose, gcost = g["pose"], g["cost"]
            key = {tuple(np.round(q, 6)): i for i, q in enumerate(gpose)}
            for vx, nx in hits:
                for vy, ny in hits:
                    for vz, nz in hits:
                        i = key.get((0.0, 0.0, 0.0, round(vx, 6),
                                     round(vy, 6), round(vz, 6)))
                        if i is not None:
                            pr.append(C[iz, nx + p, ny + p, nz + p])
                            ex_.append(gcost[i])
            if pr:
                pr, ex_ = np.array(pr, float), np.array(ex_, float)
                r2 = np.abs(pr - ex_) / np.abs(ex_)
                rep["grid6d"]["identity_rot_slice"] = {
                    "n": len(pr), "rel_err_median": float(np.median(r2)),
                    "rel_err_max": float(r2.max()),
                    "pearson_r": float(np.corrcoef(pr, ex_)[0, 1])}
    print(json.dumps(rep, indent=2, default=float))
    return rep


# ---------------------------------------------------------------------- bench

def bench(case, dataset, cfg, n_bf=None, device="cuda"):
    """Controlled A/B of stage 1 ONLY: the brute-force pose sweep and the FFT
    sweep, back to back, in ONE process on ONE machine state, on the same case
    at the same coarse spacing.  Wall-clock comparisons taken from separate runs
    are contaminated by whatever else this box is doing; this is not.
    `rigid_search.coarse_poses` is imported, not reimplemented.
    """
    from rigid_search import coarse_poses
    fx, mv, ex, exl = case_paths(dataset, case)
    us, mri = case_landmarks(dataset, case)
    cc = PoseCost(case, fx, mv, ex, exl, spacing=cfg["coarse_spacing"],
                  device=device, us_pts=us, mri_pts=mri)
    P = coarse_poses(cfg["t_max"], cfg["t_step"], cfg["r_max"], cfg["r_step"])
    n_all = len(P)
    n_bf = n_all if n_bf is None else min(int(n_bf), n_all)
    cc.cost_poses(P[:cfg["batch"]], batch=cfg["batch"])          # warm up
    torch.cuda.synchronize()
    t0 = time.time()
    cost_bf = cc.cost_poses(P[:n_bf], batch=cfg["batch"])
    torch.cuda.synchronize()
    t_bf = time.time() - t0
    bf_rate = n_bf / t_bf
    bf_full = n_all / bf_rate

    p = int(np.floor(cfg["t_max"] / cfg["coarse_spacing"]))
    rots = rot_grid(cfg["r_max"], cfg["r_step"])
    fc = FFTCoarse(cc, p, mind_mode=cfg.get("mind_mode", "recompute"),
                   mmean_mode=cfg.get("mmean_mode", "crop"))
    fc.costs_for_rotation(np.eye(3))                             # warm up
    torch.cuda.synchronize()
    t0 = time.time()
    C, T = fc.run(rots, log=None)
    torch.cuda.synchronize()
    t_fft = time.time() - t0
    t0 = time.time()
    Pc, _ = select_candidates(C, T, rots, cc.lever, cfg["topk"], cfg["sep"])
    t_sel = time.time() - t0

    rep = {
        "case": case, "spacing": cfg["coarse_spacing"], "dims": list(cc.dims),
        "n_fov": cc.n_fov,
        "bruteforce": {"n_poses_grid": int(n_all), "n_poses_timed": int(n_bf),
                       "s_timed": t_bf, "poses_per_s": bf_rate,
                       "s_full_grid_measured_or_extrapolated": bf_full,
                       "extrapolated": n_bf < n_all,
                       "min_cost": float(cost_bf.min())},
        "fft": {"n_poses": int(C.size), "n_rot": len(rots),
                "n_trans_per_rot": int((2 * p + 1) ** 3),
                "s_sweep": t_fft, "s_select": t_sel,
                "poses_per_s": C.size / t_fft,
                "min_cost": float(C.min())},
    }
    rep["speedup_same_work_equivalent"] = bf_full / t_fft
    rep["throughput_ratio"] = (C.size / t_fft) / bf_rate
    print(json.dumps(rep, indent=2, default=float))
    return rep


# ------------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["resect", "remind"], default="resect")
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--split", choices=["TUNE", "TEST", "ALL"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", metavar="CASE")
    ap.add_argument("--bench", metavar="CASE")
    ap.add_argument("--bench-n", type=int, default=None,
                    help="time only the first N brute-force poses and "
                         "extrapolate (default: the whole grid)")
    ap.add_argument("--n-probe", type=int, default=192)
    ap.add_argument("--mind-mode", choices=["recompute", "resample"],
                    default="recompute")
    ap.add_argument("--mmean-mode", choices=["crop", "ext"], default="crop")
    ap.add_argument("--refine-mode", choices=["full", "tiered"], default="full")
    ap.add_argument("--tier-k", type=int, default=6)
    for k, v in CFG.items():
        if k in ("levels", "iters"):
            continue
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return

    cfg = {k: getattr(a, k) for k in CFG if k not in ("levels", "iters")}
    cfg["levels"], cfg["iters"] = CFG["levels"], CFG["iters"]
    cfg.update(mind_mode=a.mind_mode, mmean_mode=a.mmean_mode,
               refine_mode=a.refine_mode, tier_k=a.tier_k)

    if a.check:
        check(a.check, a.dataset, cfg, n_probe=a.n_probe)
        return
    if a.bench:
        bench(a.bench, a.dataset, cfg, n_bf=a.bench_n)
        return

    out_root = OUT if not a.tag else OUT.parent / f"armA_fft{a.tag}"
    if a.dataset == "resect":
        nums = {"TUNE": RESECT_TUNE, "TEST": RESECT_TEST,
                "ALL": sorted(RESECT_TUNE + RESECT_TEST)}.get(a.split, [])
        cases = a.case or [f"Case{c}" for c in nums]
    else:
        cases = a.case or REMIND_PIDS
    for name in cases:
        j = out_root / a.dataset / name / "search.json"
        if j.exists() and not a.force:
            print(f"SKIP {name} (exists)")
            continue
        print(f"=== {name} ({a.dataset}) ===", flush=True)
        r = search(name, a.dataset, cfg, out_root=out_root)
        print(json.dumps({k: r[k] for k in
                          ("search_cost", "search_tre", "tre_identity",
                           "rot_deg", "trans_mm", "gate_fired",
                           "t_coarse_s", "t_fine_s", "wall_s")},
                         indent=2), flush=True)


if __name__ == "__main__":
    main()
