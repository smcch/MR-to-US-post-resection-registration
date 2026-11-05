"""Round 4, phase B (results/v2/contrib/PROTOCOL.md): factorial POSE x DEFORMABLE on the 55 pairs.

Poses (US world -> MR world, pred = A x):
  nav    identity (navigation alignment)
  nr     the official ReMIND2Reg NiftyReg baseline affine (results/l2r_niftyreg_baseline/<cohort>/<case>/aff.txt)
  ours   our pose (search + guarded polish, single_config.pose_of), UNGATED (abstained cases keep their rejected pose)
Deformables (the same stage applied from every pose):
  none      the pose alone
  f3d       NiftyReg reg_f3d MIND-SSC (sx 5, be 0.005) on the 1 mm working grid, -aff A (as the official f3d arm)
  convexadam  convex_adam_pt defaults on the 1 mm working grid, MR resampled under A
  mcpo2 / mcpo6   the official MCPO script on the 0.5 mm iUS box, MR resampled under A, --disp_hw 2 / default (6)
  paper     the shipped modal stage (30 FEM modes + bounded residual)
  v2        the frozen v2 stage (30 FEM modes + FEM response to symmetric correspondences)
Cells already computed in earlier experiments are REUSED (sources listed in CELLS.csv); missing cells are computed
here.  Each new cell stores TRE and the regularity of its field over the iUS FoV (folding %, SDLogJ).
Usage: python contrib_factorial.py run [--cells f3d:nav f3d:ours ...] | collect | report
Output: results/v2/contrib/B/<pose>/<deformable>/<cohort>/<case>/meta.json, B/CELLS.csv, B/REPORT.md
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rg_cohorts                                                    # noqa
import method_arms as M                                              # noqa
import single_config as SC                                           # noqa
import v2_deform as V                                                # noqa
import l2r_f3d_cohorts as LF                                         # noqa
import contrib_ablation as CA                                        # noqa
from reg_core import resample_to_grid                                # noqa

ROOT = M.R / "v2/contrib/B"
POSES = ["nr", "ours"]          # simplified by the author 2026-09-24 (navigation pose and ConvexAdam dropped)
DEFS = ["none", "f3d_mind", "f3d_lncc", "f3d_mind_lncc", "mcpo2", "mcpo6", "paper", "v2"]
F3D_FLAGS = {"f3d_mind": ["--mindssc", "1", "-sx", "5", "-be", "0.005"],
             "f3d_lncc": ["--lncc", "5", "-sx", "5", "-be", "0.005"],
             # both similarity terms in one reg_f3d run (NiftyReg sums the measures, weight 1 each)
             "f3d_mind_lncc": ["--mindssc", "1", "--lncc", "5", "-sx", "5", "-be", "0.005"]}
ORDER = ["resect_after", "bite_g4", "resect_before"]
PY = SC.PY
MCPO = M.W / "scripts/third_party/MCPO_p4_6543.py"


def pose(p, coh, c):
    if p == "nav":
        return np.eye(4)
    if p == "nr":
        return np.loadtxt(M.R / f"l2r_niftyreg_baseline/{coh}/{c}/aff.txt")
    return SC.pose_of(coh, c)


def landmarks(coh, c):
    tagp, loader = SC.TAG[coh](c)
    mri, us = loader(str(tagp))
    return np.asarray(mri, float), np.asarray(us, float)


def tre(pred, mri):
    return float(np.linalg.norm(pred - mri, axis=1).mean())


def reg_of(u_vox_hw3, fov):
    """folding %, SDLogJ of x + u (u in voxels, (H,W,D,3)) over the FoV."""
    u = torch.from_numpy(np.moveaxis(np.asarray(u_vox_hw3, np.float32), -1, 0))[None]
    return V.regularity(u, fov)


def save(p, d, coh, c, meta):
    out = ROOT / p / d / coh / c
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta))


def done(p, d, coh, c):
    return (ROOT / p / d / coh / c / "meta.json").exists()


# ---------------------------------------------------------------- cells
def cell_f3d(p, coh, c, kind="f3d_mind"):
    spec = rg_cohorts.install(coh)
    usp, mrp = spec["paths"](c)[:2]
    wgp = LF.WG[coh](c)
    fd = Path("/tmp/cf3d") / p / kind / coh / c                     # scratch on the local SSD (G: is slow for writes)
    shutil.rmtree(fd, ignore_errors=True)
    fd.mkdir(parents=True, exist_ok=True)
    affp = fd / "aff_init.txt"
    np.savetxt(affp, pose(p, coh, c))
    t0 = time.time()
    LF.F3D = F3D_FLAGS[kind]                    # MIND-SSC (as the official f3d arm) or LNCC (5 mm kernel)
    disp = LF.f3d_case(usp, mrp, wgp, affp, fd)
    mri, us = landmarks(coh, c)
    wg = nib.load(str(wgp))
    u = np.asarray(nib.load(str(disp)).dataobj, np.float32)          # mm on the 1 mm axis-aligned grid (= voxels)
    fold, sdlj = reg_of(u, np.asarray(wg.dataobj) > 0)
    t = tre(us + SC.disp_at(disp, us), mri)
    shutil.rmtree(fd, ignore_errors=True)
    return dict(tre=t, fold=fold, sdlogj=sdlj, wall_s=time.time() - t0)


def cell_convexadam(p, coh, c):
    from convexAdam.convex_adam_MIND import convex_adam_pt
    spec = rg_cohorts.install(coh)
    mrp = spec["paths"](c)[1]
    wg = nib.load(str(LF.WG[coh](c)))
    fx = np.asarray(wg.dataobj, np.float32)
    A = pose(p, coh, c)
    mv = resample_to_grid(nib.load(str(mrp)), A @ wg.affine, fx.shape).astype(np.float32)   # moving(x) = MR(A x)
    t0 = time.time()
    u = np.asarray(convex_adam_pt(fx, mv), np.float32)              # (H,W,D,3) voxels = mm
    mri, us = landmarks(coh, c)
    inv = np.linalg.inv(wg.affine)
    v = us @ inv[:3, :3].T + inv[:3, 3]
    du = np.stack([map_coordinates(u[..., k], v.T, order=1, mode="nearest") for k in range(3)], -1)
    fold, sdlj = reg_of(u, fx > 0)
    return dict(tre=tre((us + du) @ A[:3, :3].T + A[:3, 3], mri), fold=fold, sdlogj=sdlj, wall_s=time.time() - t0)


def cell_mcpo(p, hw, coh, c):
    spec = rg_cohorts.install(coh)
    usp, mrp = spec["paths"](c)[:2]
    usi, mri_img = nib.load(str(usp)), nib.load(str(mrp))
    aff, dims = V.box05(usi)
    A = pose(p, coh, c)
    fx = resample_to_grid(usi, aff, dims).astype(np.float32)
    mv = resample_to_grid(mri_img, A @ aff, dims).astype(np.float32)
    pad = [(0, int(s % 2)) for s in fx.shape]
    tag = f"{p}{hw}{coh.replace('_', '')}{c.replace('_', '')}"
    ind, outd = Path(f"/tmp/cfmcpo/{tag}"), Path(f"/tmp/cfmcpoout/{tag}")
    shutil.rmtree(ind, ignore_errors=True)
    ind.mkdir(parents=True)
    outd.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.pad(fx, pad), aff), str(ind / "ReMIND2Reg_x0_0000.nii.gz"))
    nib.save(nib.Nifti1Image(np.pad(mv, pad), aff), str(ind / "ReMIND2Reg_x0_0001.nii.gz"))
    t0 = time.time()
    extra = [] if hw == 6 else ["--disp_hw", str(hw)]
    with open(M.W / "logs/contrib_B_mcpo.log", "a") as fh:
        subprocess.run([PY, str(MCPO), "--datasets_path", str(ind) + "/", "--result_path", str(outd) + "/"] + extra,
                       stdout=fh, stderr=subprocess.STDOUT)
    dp = outd / "disp_x0_0000_x0_0001.nii.gz"
    if not dp.exists():
        raise RuntimeError("MCPO produced no output")
    ui = nib.load(str(dp))
    u = np.asarray(ui.dataobj, np.float32)
    mri, us = landmarks(coh, c)
    inv = np.linalg.inv(ui.affine)
    v = us @ inv[:3, :3].T + inv[:3, 3]
    du = np.stack([map_coordinates(u[..., k], v.T, order=1, mode="nearest") for k in range(3)], -1)
    pred = (us + du @ ui.affine[:3, :3].T) @ A[:3, :3].T + A[:3, 3]
    fold, sdlj = reg_of(u, np.pad(fx, pad) > 0)
    shutil.rmtree(ind, ignore_errors=True)
    shutil.rmtree(outd, ignore_errors=True)
    return dict(tre=tre(pred, mri), fold=fold, sdlogj=sdlj, wall_s=time.time() - t0)


def cell_v2_ours(coh, c):
    """v2 stage from our (possibly rejected) pose -- only needed for the abstained cases."""
    CA.ROOT = ROOT / "ours" / "v2_fields"
    CA.run_case("v2", CA.ARMS["C0"], coh, c)
    d = CA.ROOT / "v2" / coh / c
    mri, us = landmarks(coh, c)
    A = SC.pose_of(coh, c)
    meta = json.loads((d / "meta.json").read_text())
    return dict(tre=tre((us + SC.disp_at(d / "disp_mm.nii.gz", us)) @ A[:3, :3].T + A[:3, 3], mri),
                fold=meta["fold_pct"], sdlogj=meta["sdlogj"], wall_s=meta["wall_s"])


# -------------------------------------------------------------- reuse
def reused():
    """(pose, def, cohort, case) -> (tre, fold, sdlogj, source) from earlier experiments."""
    S = M.scores()
    out = {}
    for (coh, c), r in S.items():
        out[("nav", "none", coh, c)] = (r["initial"], np.nan, np.nan, "SCORES.initial")
        out[("ours", "none", coh, c)] = (r["pose_only"], np.nan, np.nan, "SCORES.pose_only")
        out[("nav", "paper", coh, c)] = (r["deform_only"], np.nan, np.nan, "SCORES.deform_only")
        out[("ours", "paper", coh, c)] = (r["pose_modal"], np.nan, np.nan, "SCORES.pose_modal")
    for r in csv.DictReader(open(M.R / "l2r_niftyreg_baseline/PER_CASE.csv")):
        out[("nr", "none", r["cohort"], r["case"])] = (float(r["l2r_niftyreg"]), np.nan, np.nan, "l2r PER_CASE")
    for r in csv.DictReader(open(M.R / "l2r_niftyreg_baseline/PER_CASE_F3D.csv")):
        out[("nr", "f3d_mind", r["cohort"], r["case"])] = (float(r["l2r_f3d"]), np.nan, np.nan, "l2r PER_CASE_F3D (same recipe: reg_f3d -aff)")
    for r in csv.DictReader(open(M.R / "v2/nr_affine_cohorts/PER_CASE.csv")):
        out[("nr", "paper", r["cohort"], r["case"])] = (float(r["nr_paper"]), np.nan, np.nan, "nr_affine_cohorts")
        out[("nr", "v2", r["cohort"], r["case"])] = (float(r["nr_v2"]), float(r["v2_fold"]), float(r["v2_sdlogj"]), "nr_affine_cohorts")
    for r in csv.DictReader(open(M.R / "v2/PER_CASE.csv")):
        if r["arm"] == "I4_sym":
            out[("nav", "v2", r["cohort"], r["case"])] = (float(r["tre"]), float(r["fold_pct"]), float(r["sdlogj"]), "v2 I4_sym")
        if r["arm"] == "R3_sym":
            out[("ours", "v2", r["cohort"], r["case"])] = (float(r["tre"]), float(r["fold_pct"]), float(r["sdlogj"]), "v2 R3_sym")
    for r in csv.DictReader(open(M.R / "mcpo_05/PER_CASE.csv")):
        out[("nav", "mcpo6", r["cohort"], r["case"])] = (float(r["mcpo"]), np.nan, np.nan, "mcpo_05")
    for hw, d in ((2, "deform_hw2"), (6, "deform_hw6")):
        for r in csv.DictReader(open(M.R / f"mcpo_lessons/{d}/PER_CASE.csv")):
            out[("ours", f"mcpo{hw}", r["cohort"], r["case"])] = (float(r["mcpo_from_pose"]), np.nan, np.nan, f"mcpo_lessons {d}")
    return out


def todo():
    S = M.scores()
    have = reused()
    cells = []
    for p in POSES:
        for d in DEFS:
            for coh in ORDER:
                for c in sorted(k for (k2, k) in S if k2 == coh):
                    if (p, d, coh, c) not in have and not done(p, d, coh, c):
                        cells.append((p, d, coh, c))
    return cells


def do_run(filt):
    cells = todo()
    if filt:
        cells = [x for x in cells if f"{x[1]}:{x[0]}" in filt]
    print(f"[B] {len(cells)} cells to compute", flush=True)
    for p, d, coh, c in cells:
        try:
            if d in F3D_FLAGS:
                meta = cell_f3d(p, coh, c, d)
            elif d == "convexadam":
                meta = cell_convexadam(p, coh, c)
            elif d in ("mcpo2", "mcpo6"):
                meta = cell_mcpo(p, int(d[-1]), coh, c)
            elif d == "v2" and p == "ours":
                meta = cell_v2_ours(coh, c)
            else:
                print(f"[B] no recipe for {p}/{d}", flush=True)
                continue
            save(p, d, coh, c, meta)
            print(f"[B] {p}/{d} {coh} {c}: TRE {meta['tre']:.2f}, fold {meta['fold']:.3f} %, SDLogJ {meta['sdlogj']:.3f}, "
                  f"{meta['wall_s']:.0f} s", flush=True)
        except Exception as ex:                                      # noqa: BLE001
            print(f"[B] {p}/{d} {coh} {c} FAILED: {ex}", flush=True)
        torch.cuda.empty_cache()
    print("PHASE B RUN DONE", flush=True)


def do_collect():
    S = M.scores()
    have = reused()
    rows = []
    for p in POSES:
        for d in DEFS:
            for coh in ORDER:
                for c in sorted(k for (k2, k) in S if k2 == coh):
                    f = ROOT / p / d / coh / c / "meta.json"
                    if f.exists():
                        m = json.loads(f.read_text())
                        rows.append(dict(pose=p, deform=d, cohort=coh, case=c, tre=m["tre"], fold=m["fold"], sdlogj=m["sdlogj"], source="phase B"))
                    elif (p, d, coh, c) in have:
                        t, fo, sd, src = have[(p, d, coh, c)]
                        rows.append(dict(pose=p, deform=d, cohort=coh, case=c, tre=t, fold=fo, sdlogj=sd, source=src))
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "CELLS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"collected {len(rows)} cells")


def do_report():
    rows = list(csv.DictReader(open(ROOT / "CELLS.csv")))
    T = {(r["pose"], r["deform"], r["cohort"], r["case"]): r for r in rows}
    cases = sorted({(r["cohort"], r["case"]) for r in rows})
    L = ["# Phase B — factorial pose x deformable (55 pairs; mean TRE mm; n per cell in brackets if < 55)", "",
         "| pose \\ deformable | " + " | ".join(DEFS) + " |", "|---|" + "---|" * len(DEFS)]
    for p in POSES:
        cells = []
        for d in DEFS:
            v = [float(T[(p, d) + k]["tre"]) for k in cases if (p, d) + k in T]
            cells.append(f"{np.mean(v):.3f}" + (f" [{len(v)}]" if len(v) < len(cases) else "") if v else "—")
        L.append(f"| {p} | " + " | ".join(cells) + " |")
    L += ["", "## v2 against each rival deformable, within pose (paired over the cases where both exist)", ""]
    for p in POSES:
        L.append(f"**pose {p}**")
        for d in ("none", "f3d_mind", "f3d_lncc", "f3d_mind_lncc", "mcpo2", "mcpo6", "paper"):
            ks = [k for k in cases if (p, "v2") + k in T and (p, d) + k in T]
            if len(ks) < 5:
                continue
            x = np.array([float(T[(p, "v2") + k]["tre"]) for k in ks]); y = np.array([float(T[(p, d) + k]["tre"]) for k in ks])
            pv = stats.wilcoxon(x, y, zero_method="zsplit").pvalue
            L.append(f"- v2 vs {d}: {x.mean():.3f} vs {y.mean():.3f} (n={len(ks)}); v2 better {int((x < y).sum())}/{len(ks)}; "
                     f"worst {np.max(x - y):+.2f}; p = {pv:.4g}")
        base = [k for k in cases if (p, "none") + k in T]
        worse = {d: sum(float(T[(p, d) + k]["tre"]) > float(T[(p, "none") + k]["tre"]) for k in base if (p, d) + k in T) for d in DEFS[1:]}
        L.append("- cases left worse than the pose alone: " + ", ".join(f"{d} {n}" for d, n in worse.items()))
        L.append("")
    L += ["## Regularity of the fields (where available)", "", "| pose | deformable | fold % mean (max) | SDLogJ mean (max) | n |", "|---|---|---|---|---|"]
    for p in POSES:
        for d in DEFS[1:]:
            rr = [T[(p, d) + k] for k in cases if (p, d) + k in T and T[(p, d) + k]["fold"] not in ("", "nan")]
            rr = [r for r in rr if not np.isnan(float(r["fold"]))]
            if rr:
                fo = np.array([float(r["fold"]) for r in rr]); sd = np.array([float(r["sdlogj"]) for r in rr])
                L.append(f"| {p} | {d} | {fo.mean():.3f} ({fo.max():.3f}) | {sd.mean():.3f} ({sd.max():.3f}) | {len(rr)} |")
    (ROOT / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "collect", "report"])
    ap.add_argument("--cells", nargs="*", help="deformable:pose filters, e.g. f3d:nav convexadam:ours")
    a = ap.parse_args()
    {"run": lambda: do_run(a.cells), "collect": do_collect, "report": do_report}[a.mode]()


if __name__ == "__main__":
    main()
