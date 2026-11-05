"""MCPO lesson, outside the manuscript: can a DISCRETE correlation step at 0.5 mm
(MCPO / ConvexAdam style: MIND-SSC features, dense SSD over a small set of
integer block displacements, robust rigid fit of the block correspondences)
replace the gradient-descent 0.5 mm polish of the shipped pipeline, which is
its memory bottleneck (5.8-16.8 GB, Table S6) and the reason the external
whole-head sweeps could not be polished?

Protocol (no landmark enters the computation):
  start  = the converged 1 mm search pose  results/rigid_gate/search_tight/<cohort>/<cohort>/<case>/aff.txt
  grid   = the 0.5 mm iUS box (fixed = iUS; moving = MR under the pose, MR(A0 x))
  step   = MIND-SSC (radius 3, dilation 3 at 0.5 mm, as MCPO) -> block average
           (g voxels) -> SSD for every integer block displacement in [-hw, hw]^3
           -> per-block argmin (optionally MCPO's coupled-convex smoothing)
           -> trimmed least-squares rigid fit of block centre -> displaced centre
           -> pose A1 = A0 . T
  guard  = the shipped polish's guard: RMS move over the FoV <= 2 mm, else A0 kept
Scored afterwards against the landmarks next to the search pose and the shipped
polished pose (results/rigid_gate/search_polish); wall time and peak GPU memory
of the step are recorded.  --selfcheck registers an MR to a copy of itself moved
by a known rigid transform (validates the convention and the fit).
Output: results/mcpo_lessons/polish/<setting>/{PER_CASE.csv,SUMMARY.txt}
Usage: python mcpo_lessons_polish.py [--cohorts bite_g4 resect_after resect_before]
                                     [--g 3 --hw 3 --mind 3 3] [--coupled] [--all]
       python mcpo_lessons_polish.py --selfcheck [resect_after Case21]
"""
import argparse
import csv
import itertools
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rg_cohorts                                                    # noqa
import method_arms as M                                              # noqa
import single_config as SC                                           # noqa
from reg_core import resample_to_grid, mind_ssc, normalize           # noqa
from deeds_ssc_cohorts import padded_box                             # noqa
from l2r_baseline_cohorts import horn                                # noqa

R = M.R
OUT = R / "mcpo_lessons/polish"
ORDER = ["bite_g4", "resect_after", "resect_before"]
LABEL = {"resect_after": "RESECT post-resection", "bite_g4": "BITE post-resection",
         "resect_before": "RESECT pre-resection"}


def rotvec_deg(Rm):
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))


def trimmed_rigid(p, q, keep=0.6, iters=4):
    """Rigid (R, t) with q ~ R p + t, least-trimmed-squares style: refit on the
    `keep` fraction of smallest residuals.  Returns R, t, inlier mask."""
    idx = np.arange(len(p))
    for _ in range(iters):
        Rm, t = horn(p[idx], q[idx])
        r = np.linalg.norm(p @ Rm.T + t - q, axis=1)
        k = max(int(keep * len(p)), 6)
        idx = np.argsort(r)[:k]
    Rm, t = horn(p[idx], q[idx])
    inl = np.zeros(len(p), bool)
    inl[idx] = True
    return Rm, t, inl


def discrete_step(f_n, m_n, aff, g, hw, mind_r, mind_d, coupled=False, dev="cuda"):
    """The discrete correlation step.  f_n, m_n: normalised fixed / moving on the
    same 0.5 mm grid (aff).  Returns (R, t) in the fixed world frame with
    MR-under-pose(R x + t) ~ US(x), the number of blocks used, and a dict of
    diagnostics."""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    f_t = torch.from_numpy(f_n)[None, None].to(dev)
    m_t = torch.from_numpy(m_n)[None, None].to(dev)
    with torch.no_grad():
        Ff = F.avg_pool3d(mind_ssc(f_t, radius=mind_r, dilation=mind_d), g, stride=g)
        Fm = F.avg_pool3d(mind_ssc(m_t, radius=mind_r, dilation=mind_d), g, stride=g)
        vf = F.avg_pool3d((f_t > 0).float(), g, stride=g)[0, 0] > 0.9      # block inside the iUS FoV
        vm = F.avg_pool3d((m_t > 0).float(), g, stride=g)[0, 0] > 0.9      # block with MR content
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
        mesh = torch.tensor(shifts, device=dev, dtype=torch.float32)          # (N, 3) in blocks
        amin = ssd.argmin(0)                                                  # (h, w, d)
        if coupled:
            # MCPO / ConvexAdam coupled convex smoothing (Siebert et al.): alternate
            # the SSD argmin with a quadratic coupling to the locally averaged field
            disp = mesh[amin].permute(3, 0, 1, 2)[None]                       # (1, 3, h, w, d)
            soft = F.avg_pool3d(disp, 3, stride=1, padding=1)
            acc = torch.zeros_like(disp)
            for alpha in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
                pen = (mesh.view(-1, 3, 1, 1, 1) - soft[0][None]).pow(2).sum(1)  # (N, h, w, d)
                amin_c = (ssd + alpha * pen).argmin(0)
                soft = F.avg_pool3d(mesh[amin_c].permute(3, 0, 1, 2)[None], 3, stride=1, padding=1)
                acc += soft
            disp = (acc / 6.0)[0]                                             # (3, h, w, d) blocks
            ub = disp.permute(1, 2, 3, 0)
            saturated = torch.zeros_like(vf)
        else:
            ub = mesh[amin]                                                   # (h, w, d, 3) blocks
            saturated = (ub.abs() >= hw).any(-1)
            # sub-block refinement: a parabola through the SSD at the argmin and
            # its two neighbours along each axis (standard sub-pixel step); the
            # block displacements are otherwise quantised to g voxels
            n1 = 2 * hw + 1
            flat = amin.view(1, -1)
            ssd_flat = ssd.view(len(shifts), -1)
            delta = torch.zeros((3,) + tuple(amin.shape), device=dev)
            for k, step in enumerate((n1 * n1, n1, 1)):
                ok = (~saturated).view(-1)
                lo = (flat - step).clamp(0, len(shifts) - 1)
                hi = (flat + step).clamp(0, len(shifts) - 1)
                s0 = ssd_flat.gather(0, flat)[0]
                sm = ssd_flat.gather(0, lo)[0]
                sp = ssd_flat.gather(0, hi)[0]
                den = sm - 2 * s0 + sp
                good = ok & torch.isfinite(sm) & torch.isfinite(sp) & (den > 1e-6)
                dk = torch.where(good, 0.5 * (sm - sp) / den.clamp(min=1e-6), torch.zeros_like(s0))
                delta[k] = dk.clamp(-0.5, 0.5).view(amin.shape)
            ub = ub + delta.permute(1, 2, 3, 0)
        # second-best margin as a distinctiveness diagnostic
        top2 = torch.topk(ssd, 2, dim=0, largest=False).values
        valid = vf & ~saturated & torch.isfinite(top2[0])
        torch.cuda.synchronize()
        t_step = time.time() - t0
        peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
        idx = torch.nonzero(valid).cpu().numpy().astype(float)
        u_blocks = ub[valid].cpu().numpy()
    centre_vox = idx * g + (g - 1) / 2.0                                       # fine-voxel block centres
    p = centre_vox @ aff[:3, :3].T + aff[:3, 3]
    q = p + (u_blocks * g) @ aff[:3, :3].T
    Rm, t, inl = trimmed_rigid(p, q)
    diag = dict(n_blocks=int(len(p)), n_inliers=int(inl.sum()), t_step_s=t_step, peak_mb=peak_mb,
                rot_deg=rotvec_deg(Rm), trans_mm=float(np.linalg.norm(t)),
                resid_inl_mm=float(np.linalg.norm(p[inl] @ Rm.T + t - q[inl], axis=1).mean()))
    del ssd, Ff, Fm, Fm_p, f_t, m_t
    torch.cuda.empty_cache()
    return Rm, t, diag


def as_affine(Rm, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = Rm, t
    return T


def fov_move(A0, A1, f_arr, aff, n=4000):
    ijk = np.argwhere(f_arr > 0)
    if len(ijk) > n:
        ijk = ijk[np.random.default_rng(0).choice(len(ijk), n, replace=False)]
    pts = ijk @ aff[:3, :3].T + aff[:3, 3]
    h = np.c_[pts, np.ones(len(pts))]
    return float(np.sqrt((((h @ A1.T) - (h @ A0.T))[:, :3] ** 2).sum(1).mean()))


def tre(A, us, mri):
    return float(np.linalg.norm(us @ A[:3, :3].T + A[:3, 3] - mri, axis=1).mean())


def selfcheck(a):
    coh, c = (a.cohorts[0], a.case) if a.case else ("resect_after", "Case21")
    spec = rg_cohorts.install(coh)
    tagp, loader = SC.TAG[coh](c)
    mri, _ = loader(str(tagp))
    mri = np.asarray(mri, float)
    usp, mrp = spec["paths"](c)[:2]
    usi, mri_img = nib.load(str(usp)), nib.load(str(mrp))
    aff, dims = padded_box(usi)
    cen = aff[:3, :3] @ (np.array(dims, float) / 2) + aff[:3, 3]
    th = np.deg2rad(2.0)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    G = as_affine(Rz, cen - Rz @ cen + np.array([1.5, -2.0, 1.0]))
    # fixed(x) = MR(G x); moving(x) = MR(x)  ->  the step must find T ~ G
    fx = resample_to_grid(mri_img, G @ aff, dims).astype(np.float32)
    mv = resample_to_grid(mri_img, aff, dims).astype(np.float32)
    f_n, m_n = normalize(fx, fx > 0), normalize(mv, mv > 0)
    Rm, t, dg = discrete_step(f_n, m_n, aff, a.g, a.hw, a.mind[0], a.mind[1], a.coupled)
    T = as_affine(Rm, t)
    # landmarks: the MR points appear in the fixed at q = G^-1 mri; T q must return mri
    Ginv = np.linalg.inv(G)
    q = mri @ Ginv[:3, :3].T + Ginv[:3, 3]
    err = np.linalg.norm(q @ T[:3, :3].T + T[:3, 3] - mri, axis=1)
    e0 = np.linalg.norm(q - mri, axis=1)
    print(f"selfcheck {coh} {c}: known move {e0.mean():.2f} mm ({rotvec_deg(G[:3, :3]):.2f} deg); "
          f"recovered {dg['rot_deg']:.2f} deg; residual at landmarks mean {err.mean():.3f} max {err.max():.3f} mm; "
          f"blocks {dg['n_blocks']} inliers {dg['n_inliers']}; step {dg['t_step_s']:.1f} s, peak {dg['peak_mb']:.0f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="*", default=ORDER)
    ap.add_argument("--g", type=int, default=3)
    ap.add_argument("--hw", type=int, default=3)
    ap.add_argument("--mind", type=int, nargs=2, default=[3, 3])
    ap.add_argument("--coupled", action="store_true")
    ap.add_argument("--iters", type=int, default=1, help="repeat the step from the updated pose")
    ap.add_argument("--all", action="store_true", help="also the abstained cases")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--case")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck(a)
        return
    setting = (f"g{a.g}_hw{a.hw}_mind{a.mind[0]}{a.mind[1]}" + ("_coupled" if a.coupled else "")
               + (f"_it{a.iters}" if a.iters > 1 else ""))
    out = OUT / setting
    out.mkdir(parents=True, exist_ok=True)
    S = M.scores()
    rows = []
    csvp = out / "PER_CASE.csv"
    done = {(r["cohort"], r["case"]): r for r in csv.DictReader(open(csvp))} if csvp.exists() else {}
    for coh in a.cohorts:
        spec = rg_cohorts.install(coh)
        for c in sorted(k for (k2, k) in S if k2 == coh):
            if a.case and c != a.case:
                continue
            if S[(coh, c)]["abstained"] and not a.all:
                continue
            if (coh, c) in done:
                rows.append({k: (float(v) if k not in ("cohort", "case") else v) for k, v in done[(coh, c)].items()})
                continue
            tagp, loader = SC.TAG[coh](c)
            mri, us = loader(str(tagp))
            mri, us = np.asarray(mri, float), np.asarray(us, float)
            A0 = np.loadtxt(R / f"rigid_gate/search_tight/{coh}/{coh}/{c}/aff.txt")   # the 1 mm search pose (pre-polish)
            A_ship = SC.pose_of(coh, c)                                       # the shipped (guarded 0.5 mm) polished pose
            usp, mrp = spec["paths"](c)[:2]
            usi, mri_img = nib.load(str(usp)), nib.load(str(mrp))
            t0 = time.time()
            aff, dims = padded_box(usi)
            fx = resample_to_grid(usi, aff, dims).astype(np.float32)
            mv = resample_to_grid(mri_img, A0 @ aff, dims).astype(np.float32)
            f_n, m_n = normalize(fx, fx > 0), normalize(mv, mv > 0)
            t_prep = time.time() - t0
            Rm, t, dg = discrete_step(f_n, m_n, aff, a.g, a.hw, a.mind[0], a.mind[1], a.coupled)
            A1 = A0 @ as_affine(Rm, t)
            for _ in range(1, a.iters):            # optional: repeat from the updated pose (re-resample the MR)
                mv = resample_to_grid(mri_img, A1 @ aff, dims).astype(np.float32)
                m_n = normalize(mv, mv > 0)
                Rm, t, dg2 = discrete_step(f_n, m_n, aff, a.g, a.hw, a.mind[0], a.mind[1], a.coupled)
                A1 = A1 @ as_affine(Rm, t)
                dg["t_step_s"] += dg2["t_step_s"]
                dg["peak_mb"] = max(dg["peak_mb"], dg2["peak_mb"])
                dg["resid_inl_mm"] = dg2["resid_inl_mm"]
            move = fov_move(A0, A1, fx, aff)
            keep = move <= 2.0
            A_out = A1 if keep else A0
            row = dict(cohort=coh, case=c, initial=float(np.linalg.norm(us - mri, axis=1).mean()),
                       tre_search=tre(A0, us, mri), tre_shipped_polish=tre(A_ship, us, mri),
                       tre_discrete_raw=tre(A1, us, mri), tre_discrete=tre(A_out, us, mri),
                       move_mm=move, accepted=int(keep), shipped_move_mm=fov_move(A0, A_ship, fx, aff),
                       t_prep_s=t_prep, **dg, dims="x".join(str(int(x)) for x in dims))
            rows.append(row)
            with open(csvp, "a" if csvp.exists() else "w", newline="") as f:
                wcsv = csv.DictWriter(f, fieldnames=list(row))
                if f.tell() == 0:
                    wcsv.writeheader()
                wcsv.writerow(row)
            print(f"[disc] {coh} {c}: search {row['tre_search']:.2f} | shipped polish {row['tre_shipped_polish']:.2f} "
                  f"| discrete {row['tre_discrete_raw']:.2f} (move {move:.2f}{'' if keep else ' REJECTED'}) "
                  f"| blocks {dg['n_blocks']} | {dg['t_step_s']:.1f} s, {dg['peak_mb']:.0f} MB", flush=True)
    Ls = [f"Discrete 0.5 mm polish ({setting}) vs the shipped gradient polish, accepted cases; mean TRE (mm)"]
    for coh in a.cohorts:
        rr = [r for r in rows if r["cohort"] == coh]
        if not rr:
            continue
        g = lambda k: np.array([float(r[k]) for r in rr])                        # noqa: E731
        s0, sp, sd, sr = g("tre_search"), g("tre_shipped_polish"), g("tre_discrete"), g("tre_discrete_raw")
        p = stats.wilcoxon(sd, sp, zero_method="zsplit").pvalue if len(rr) > 1 else float("nan")
        Ls.append(f"=== {LABEL[coh]} n={len(rr)} ===")
        Ls.append(f"  search 1 mm {s0.mean():.3f} | shipped polish {sp.mean():.3f} | discrete {sd.mean():.3f} "
                  f"(raw {sr.mean():.3f}, rejected {int((g('accepted') == 0).sum())}) | discrete vs shipped p={p:.4f}, "
                  f"discrete better {int((sd < sp).sum())}/{len(rr)}; vs search: discrete better {int((sd < s0).sum())}, "
                  f"shipped better {int((sp < s0).sum())}")
        Ls.append(f"  step {g('t_step_s').mean():.1f} s (+ {g('t_prep_s').mean():.1f} s resampling), peak GPU "
                  f"{g('peak_mb').mean():.0f} MB (max {g('peak_mb').max():.0f}); move {g('move_mm').mean():.2f} mm "
                  f"(shipped {g('shipped_move_mm').mean():.2f}); blocks {g('n_blocks').mean():.0f}, inlier residual "
                  f"{g('resid_inl_mm').mean():.2f} mm")
    (out / "SUMMARY.txt").write_text("\n".join(Ls))
    print("\n".join(Ls))
    print("DISCRETE POLISH DONE")


if __name__ == "__main__":
    main()
