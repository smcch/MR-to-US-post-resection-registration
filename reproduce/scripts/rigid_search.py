"""rigid_search.py -- MIND-driven global rigid search.

**ARM A of the capture-range extension.  A SEPARATE, DECLARED ARM: it does not
modify, re-run or re-tune the frozen method, and no number produced here
replaces a published one.**

The frozen pipeline (`reg_modal.py`) optimises 30 modal coefficients from a
zero initialisation.  Its basis is built on an elastic foundation, so it has no
rigid-body modes and a bulk pose error is outside its search space by
construction.  `results/rigid_decomposition/RIGID_DECOMPOSITION.md` measured
that cost; `results/capture_range/DIAGNOSIS.md` showed the information needed to
undo the pose error IS present in the frozen method's own masked MIND-SSC data
term, but only when that term is evaluated at the pipeline's native 1 mm working
grid.  This script is the deployable consequence:

    stage 1   exhaustive coarse pose grid at `--coarse-spacing` mm   (candidates)
    stage 2   top-K coarse minima, forced >`--sep` mm apart in a composite pose
              metric, + identity, each refined by multi-resolution Adam on the
              **1 mm** working grid (4 mm -> 2 mm -> 1 mm)
    select    lowest FINAL (1 mm) cost among poses that are valid, i.e. sample
              < `--max-bg` of the FoV from image background
    gate      abstain (return identity) if the winner's rotation exceeds
              `--rot-cap`, a plausibility prior on neuronavigation pose error

Everything is deterministic: a fixed grid, a fixed candidate ordering, a fixed
optimiser schedule, no random restarts, no landmark anywhere inside the search.
Landmarks, when given, are used ONLY to score the result afterwards.

The cost is the frozen data term, reproduced to <=3e-6 relative
(`capture_range.py` gate; re-checked here with `--check`):

    working grid = load_to_grid(fixed, moving, spacing)      (isotropic, US bbox)
    fov          = (US > 0) & ~dilate(exclude labels, 3 mm)
    cost(A)      = sum_x fov(x) mean_c (MIND_f(x) - MIND_{m.A}(x))^2 / sum_x fov(x)

Pose convention (identical to capture_range.py, to the Kabsch oracle and to
reg_aladin's aff.txt):  A maps **US world -> MR world**,
`A(r,t) x = R(r)(x - c) + c + t`, c = FoV centroid, r in degrees.

Usage
    python rigid_search.py --dataset resect --case Case12
    python rigid_search.py --dataset remind --case ReMIND-045
    python rigid_search.py --check Case18
"""
import argparse
import json
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reg_core import load_to_grid, resample_to_grid, normalize          # noqa
from capture_range import (CaseCost, mind_ssc_batch, rodrigues, rotvec_deg,   # noqa
                           pose_to_affine, affine_to_pose, pose_dist, fit_rigid)
from resect_tags import load_tag_pairs                                   # noqa

WD = Path("/mnt/c/ReMIND_work")
OUT = WD / "results/capture_range/armA"
TAGS = Path("/mnt/g/ECOGRAFIA/ReMIND_downloads/RESECT/NIFTI")

RESECT_TUNE = [1, 3, 7, 11, 15, 17, 19, 21, 23, 27]
RESECT_TEST = [2, 4, 6, 8, 12, 14, 16, 18, 24, 25]
REMIND_PIDS = ["ReMIND-003", "ReMIND-023", "ReMIND-031", "ReMIND-032",
               "ReMIND-034", "ReMIND-045", "ReMIND-056", "ReMIND-077",
               "ReMIND-079", "ReMIND-087", "ReMIND-091", "ReMIND-096",
               "ReMIND-107", "ReMIND-109"]
MRC = ["mr_x_RAS_mm", "mr_y_RAS_mm", "mr_z_RAS_mm"]
USC = ["us_x_RAS_mm", "us_y_RAS_mm", "us_z_RAS_mm"]


# --------------------------------------------------------------- dataset paths

def case_paths(dataset, case):
    """(fixed, moving, exclude, exclude_labels) for a case of either dataset,
    exactly the files the frozen arm uses."""
    if dataset == "resect":
        d = WD / "data/resect" / case
        ex = d / f"{case}-US-after-cavity.nii.gz"
        return (d / f"{case}-US-after.nii.gz", d / f"{case}-FLAIR.nii.gz",
                ex if ex.exists() else None, [1])
    if dataset == "remind":
        ex = WD / f"data/segmentations/US/{case}-post-us-segmentation.nii.gz"
        return (WD / f"data/us/{case}_USpreimri.nii.gz",
                WD / f"data/synthseg_in/preop/{case}_T2.nii.gz",
                ex if ex.exists() else None, [1, 2])
    raise ValueError(dataset)


def case_landmarks(dataset, case):
    """(us_pts, mri_pts) world mm, or (None, None).  SCORING ONLY."""
    if dataset == "resect":
        tag = TAGS / case / "Landmarks" / f"{case}-MRI-afterUS.tag"
        if not tag.exists():
            return None, None
        mri, us = load_tag_pairs(str(tag))
        return us, mri
    p = WD / "release/landmarks/ReMIND-BS" / case / f"landmarks_{case}.csv"
    if not p.exists():
        return None, None
    t = pd.read_csv(p, comment="#")
    return t[USC].to_numpy(float), t[MRC].to_numpy(float)


# ------------------------------------------------------------------- pose cost

class PoseCost(CaseCost):
    """Same cost/optimiser machinery as capture_range.CaseCost but constructed
    from explicit paths, so it serves RESECT and ReMIND alike.  All methods
    (_sample, cost_poses, cost_torch, pyramid, descend) are inherited unchanged
    -- only the setup differs."""

    def __init__(self, name, fixed, moving, exclude=None, exclude_labels=(1,),
                 exclude_dilate_mm=3.0, spacing=1.0, device="cuda",
                 us_pts=None, mri_pts=None):
        self.case, self.spacing = name, spacing
        self.fx_path, self.mv_path = Path(fixed), Path(moving)
        dev = torch.device(device)
        self.dev = dev

        f_arr, m_arr, grid_aff, dims = load_to_grid(self.fx_path, self.mv_path,
                                                    spacing)
        self.grid_aff, self.dims = grid_aff, tuple(int(x) for x in dims)
        fov = f_arr > 0
        self.fov_raw, self.n_fov_raw = fov.copy(), int(fov.sum())
        self.n_cav = 0
        if exclude is not None and Path(exclude).exists():
            img = nib.load(str(exclude))
            lab = np.isin(np.asarray(img.dataobj),
                          list(exclude_labels)).astype(np.float32)
            ex = resample_to_grid(img, grid_aff, dims, mode="nearest",
                                  data=lab) > 0.5
            if ex.any() and exclude_dilate_mm > 0:
                ex = binary_dilation(ex, iterations=max(
                    1, int(round(exclude_dilate_mm / spacing))))
            fov = fov & ~ex
            self.n_cav = int(ex.sum())
        self.fov, self.n_fov = fov, int(fov.sum())

        f_n = normalize(f_arr, fov)
        self.f_arr = f_arr
        ref = m_arr[m_arr > 0]
        self.m_lo, self.m_hi = np.percentile(ref, [1.0, 99.0])
        self.m_arr_id = m_arr

        self.f_t = torch.from_numpy(f_n)[None, None].to(dev)
        self.mind_f = mind_ssc_batch(self.f_t)
        self.k = torch.from_numpy(fov.astype(np.float32))[None, None].to(dev)
        self.ksum = float(self.k.sum())

        mv = nib.load(str(self.mv_path))
        self.mv_shape = mv.shape[:3]
        self.mv_t = torch.from_numpy(np.ascontiguousarray(
            np.asarray(mv.dataobj, dtype=np.float32)))[None, None].to(dev)
        self.mv_inv = np.linalg.inv(mv.affine)

        ii, jj, kk = np.meshgrid(*[np.arange(n) for n in self.dims],
                                 indexing="ij")
        ijk1 = np.stack([ii, jj, kk, np.ones_like(ii)], -1).astype(np.float64)
        self.X = (ijk1 @ grid_aff.T)[..., :3]
        self.Xt = torch.from_numpy(self.X.astype(np.float32)).to(dev)
        self.center = self.X[fov].mean(0)
        self.fov_extent = self.X[fov].max(0) - self.X[fov].min(0)
        # LANDMARK-FREE lever arm: RMS radius of the FoV about its centroid.
        self.lever = float(np.sqrt(
            ((self.X[fov] - self.center) ** 2).sum(1).mean()))

        # landmarks: scoring only, never used by the search
        self.us_pts, self.mri_pts = us_pts, mri_pts
        if us_pts is not None:
            R, t = fit_rigid(us_pts, mri_pts)
            self.A_oracle = np.eye(4)
            self.A_oracle[:3, :3], self.A_oracle[:3, 3] = R, t
            self.r_oracle, self.t_oracle = affine_to_pose(self.A_oracle,
                                                          self.center)
        else:
            self.A_oracle = np.eye(4)
            self.r_oracle, self.t_oracle = np.zeros(3), np.zeros(3)

    def tre(self, A):
        if self.us_pts is None:
            return float("nan")
        p = self.us_pts @ A[:3, :3].T + A[:3, 3]
        return float(np.linalg.norm(p - self.mri_pts, axis=1).mean())

    def bg_fraction(self, r, t):
        """Fraction of the FoV that samples image background (moving <= 0.005).
        MIND is degenerate on a constant image, so such poses are invalid."""
        m = self._sample(rodrigues(np.asarray(r, float))[None],
                         np.asarray(t, float)[None])
        kk = torch.from_numpy(self.fov).to(self.dev)
        v = float((m[0, 0][kk] <= 0.005).float().mean())
        del m
        return v


# ---------------------------------------------------------------------- search

def coarse_poses(t_max, t_step, r_max, r_step):
    """Deterministic exhaustive 6-D pose grid, ordered lexicographically."""
    tv = np.arange(-t_max, t_max + 1e-9, t_step)
    rv = np.arange(-r_max, r_max + 1e-9, r_step)
    R = np.stack(np.meshgrid(rv, rv, rv, indexing="ij"), -1).reshape(-1, 3)
    T = np.stack(np.meshgrid(tv, tv, tv, indexing="ij"), -1).reshape(-1, 3)
    P = np.empty((len(R) * len(T), 6))
    P[:, :3] = np.repeat(R, len(T), axis=0)
    P[:, 3:] = np.tile(T, (len(R), 1))
    return P


def search(name, dataset, cfg, device="cuda", log=print):
    """Run arm A on one case.  Returns the result dict."""
    t_all = time.time()
    fx, mv, ex, exl = case_paths(dataset, name)
    us_pts, mri_pts = case_landmarks(dataset, name)

    # ---- stage 1: exhaustive coarse grid (candidate generation) -------------
    t0 = time.time()
    cc = PoseCost(name, fx, mv, ex, exl, spacing=cfg["coarse_spacing"],
                  device=device, us_pts=us_pts, mri_pts=mri_pts)
    P = coarse_poses(cfg["t_max"], cfg["t_step"], cfg["r_max"], cfg["r_step"])
    cost_c = cc.cost_poses(P, batch=cfg["batch"])
    lever = cc.lever
    t_coarse = time.time() - t0
    log(f"  coarse: {len(P)} poses @{cfg['coarse_spacing']} mm "
        f"(grid {cc.dims}, fov {cc.n_fov}) in {t_coarse:.0f}s "
        f"({len(P)/max(t_coarse,1e-9):.0f}/s); lever {lever:.1f} mm")

    order = np.argsort(cost_c)
    sel = []
    for j in order:
        q = P[j]
        if all(pose_dist(q[:3], q[3:], P[w][:3], P[w][3:], lever)[2] > cfg["sep"]
               for w in sel):
            sel.append(int(j))
        if len(sel) >= cfg["topk"]:
            break
    del cc
    torch.cuda.empty_cache()

    # ---- stage 2: refinement at the pipeline's native 1 mm working grid -----
    t0 = time.time()
    ff = PoseCost(name, fx, mv, ex, exl, spacing=cfg["fine_spacing"],
                  device=device, us_pts=us_pts, mri_pts=mri_pts)
    zero = np.zeros(6)
    c_id = float(ff.cost_poses(zero[None], batch=1)[0])
    tre_id = ff.tre(np.eye(4))
    cache, rows = {}, []
    starts = [("identity", zero)] + [(f"coarse{i}", P[j])
                                     for i, j in enumerate(sel)]
    for si, (tag, p0) in enumerate(starts):
        r1, t1, c1 = ff.descend(p0[:3], p0[3:], levels=cfg["levels"],
                                iters=cfg["iters"], cache=cache)
        A1 = pose_to_affine(r1, t1, ff.center)
        rows.append(dict(start=tag, cost0=float(cost_c[sel[si - 1]]) if si else c_id,
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

    # ---- selection ---------------------------------------------------------
    D = pd.DataFrame(rows)
    valid = D[D.bg < cfg["max_bg"]]
    pool = valid if len(valid) else D
    ib = int(pool.cost.values.argmin())
    best = pool.iloc[ib]
    br = np.array([best.rx, best.ry, best.rz])
    bt = np.array([best.tx, best.ty, best.tz])
    A = pose_to_affine(br, bt, ff.center)

    # agreement + margin (landmark-free diagnostics, candidate gates)
    dists = np.array([pose_dist(br, bt,
                                np.array([v.rx, v.ry, v.rz]),
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
        "n_coarse": int(len(P)), "n_starts": int(len(starts)),
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
    }
    if us_pts is not None:
        res["tre_oracle"] = ff.tre(ff.A_oracle)
        res["oracle_rot_deg"] = float(np.linalg.norm(rotvec_deg(
            ff.A_oracle[:3, :3])))
        res["d_oracle_mm"] = pose_dist(br, bt, ff.r_oracle, ff.t_oracle,
                                       lever)[2]
    d = OUT / dataset / name
    d.mkdir(parents=True, exist_ok=True)
    D.to_csv(d / "starts.csv", index=False)
    (d / "search.json").write_text(json.dumps(res, indent=2))
    np.savetxt(d / "aff.txt", A, fmt="%.10f")
    np.savetxt(d / "aff_gated.txt", A_gated, fmt="%.10f")
    return res


# ------------------------------------------------------------------ CFG (spec)

CFG = dict(coarse_spacing=3.0, fine_spacing=1.0,
           t_max=24.0, t_step=6.0, r_max=24.0, r_step=12.0,
           topk=24, sep=8.0, batch=48,
           levels=(4, 2, 1), iters=(200, 150, 120),
           max_bg=0.02, rot_cap=20.0, trans_cap=30.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["resect", "remind"], default="resect")
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--split", choices=["TUNE", "TEST", "ALL"])
    ap.add_argument("--check", metavar="CASE",
                    help="reproduction gate against capture_range.CaseCost")
    for k, v in CFG.items():
        if k in ("levels", "iters"):
            continue
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--tag", default="", help="suffix for the output dir")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.check:
        gate(a.check)
        return

    cfg = {k: getattr(a, k) for k in CFG if k not in ("levels", "iters")}
    cfg["levels"], cfg["iters"] = CFG["levels"], CFG["iters"]
    if a.dataset == "resect":
        nums = {"TUNE": RESECT_TUNE, "TEST": RESECT_TEST,
                "ALL": sorted(RESECT_TUNE + RESECT_TEST)}.get(a.split, [])
        cases = a.case or [f"Case{c}" for c in nums]
    else:
        cases = a.case or REMIND_PIDS
    global OUT
    if a.tag:
        OUT = OUT.parent / f"armA{a.tag}"
    for name in cases:
        j = OUT / a.dataset / name / "search.json"
        if j.exists() and not a.force:
            print(f"SKIP {name} (exists)")
            continue
        print(f"=== {name} ({a.dataset}) ===", flush=True)
        r = search(name, a.dataset, cfg)
        print(json.dumps({k: r[k] for k in
                          ("search_cost", "search_tre", "tre_identity",
                           "rot_deg", "trans_mm", "gate_fired", "wall_s")},
                         indent=2), flush=True)


def gate(case):
    """Correctness gate: PoseCost must reproduce capture_range.CaseCost (which
    itself reproduces the frozen reg_modal.py data term to <=2.6e-6) exactly."""
    for sp in (2.0, 1.0):
        ref = CaseCost(case, spacing=sp)
        fx, mv, ex, exl = case_paths("resect", case)
        us, mri = case_landmarks("resect", case)
        new = PoseCost(case, fx, mv, ex, exl, spacing=sp, us_pts=us,
                       mri_pts=mri)
        assert ref.dims == new.dims and ref.n_fov == new.n_fov, "grid mismatch"
        P = np.concatenate([np.zeros((1, 6)),
                            np.concatenate([ref.r_oracle, ref.t_oracle])[None],
                            np.array([[3.0, -2.0, 1.0, 5.0, -4.0, 2.0]])])
        ca, cb = ref.cost_poses(P, batch=3), new.cost_poses(P, batch=3)
        rel = np.abs(ca - cb) / np.abs(ca)
        print(f"{case} @{sp} mm  dims {ref.dims} fov {ref.n_fov}  "
              f"cost {ca}  max rel diff {rel.max():.3e}  "
              f"lever_fov {new.lever:.2f} mm  "
              f"lever_landmark {np.sqrt(((ref.us_pts-ref.us_pts.mean(0))**2).sum(1).mean()):.2f} mm")
        assert rel.max() < 1e-9, "PoseCost != CaseCost"
        del ref, new
        torch.cuda.empty_cache()
    print("GATE OK")


if __name__ == "__main__":
    main()
