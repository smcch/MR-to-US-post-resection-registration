"""Experiment (independent of the manuscript), run from the C: working copy: MCPO POSE + OUR deformable stages.

Per case:
  1. MCPO (official script, defaults) from the navigation start on the 0.5 mm iUS box (as results/mcpo_05).
  2. MCPO pose = the rigid transform that best fits MCPO's displacement over the iUS field of view
     (trimmed least squares on every 4th FoV voxel, keep 60 %, 4 passes; the same fit as mcpo_lessons_decompose.py);
     A maps US world -> MR world (same convention as our pose and the NiftyReg affine).
  3. MR and SynthSeg relabelled by A, FEM basis rebuilt there (biomech_basis, h 5, 30 modes; ReMIND with the tumour
     segmentation, as rs_prestage 'remind').
  4. Paper stage on top: RESECT/BITE as single_config (30 modes + residual s1.5, lambda 3e-3, no mask);
     ReMIND as rs_prestage 'remind' (lambda 3e-3, exclusion of iUS labels 1,2 dilated 3 mm).
  5. v2 stage on top (v2_method.ACCEPTED through v2_remind_eval.deform).
Scored pred = A (x_us + u(x_us)).  Also reports MCPO's own full result and the MCPO rigid pose alone.
Usage: python mcpo_pose_deform.py run [--only remind|resect] | report
Output: /mnt/c/ReMIND_work/results/mcpo_pose/<cohort>/<case>/{pose.txt,meta.json,basis30.npz,paper/,v2/}, REPORT.md
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
from scipy import stats
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import method_arms as M                                              # noqa
import single_config as SC                                           # noqa
import rg_cohorts                                                    # noqa
import v2_deform as V                                                # noqa
import v2_method as FM                                               # noqa
import v2_remind_eval as VE                                          # noqa
import contrib_factorial_remind as CR                                # noqa
from rs_prestage import relabel                                      # noqa
from reg_core import resample_to_grid                                # noqa
from mcpo_lessons_polish import trimmed_rigid                        # noqa

W = Path("/mnt/c/ReMIND_work")
OUT = W / "results/mcpo_pose"
PY = SC.PY
MCPO = W / "scripts/third_party/MCPO_p4_6543.py"
ORDER = ["resect_after", "bite_g4", "resect_before"]
REMIND = CR.CASES


def inputs(coh, c):
    """fixed US, moving MR, SynthSeg, tumour seg or None, iUS exclusion seg or None, landmarks (mri, us)."""
    if coh == "remind":
        us, mr, _ = CR.paths(c)
        sg = W / f"remind/segs/{c}_T2_synthseg.nii.gz"
        tum = W / f"remind/segs/{c}-pre-mri-segmentation.nii.gz"
        exs = W / f"remind/segs/{c}-post-us-segmentation.nii.gz"
        mri, usl = CR.landmarks(c)
        return us, mr, sg, (tum if tum.exists() else None), (exs if exs.exists() else None), mri, usl
    spec = rg_cohorts.install(coh)
    us, mr = spec["paths"](c)[:2]
    tagp, loader = SC.TAG[coh](c)
    mri, usl = loader(str(tagp))
    return us, mr, spec["seg"](c), None, None, np.asarray(mri, float), np.asarray(usl, float)


def run_cmd(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        return subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT).returncode


def mcpo_pose(coh, c, d):
    """Run MCPO from navigation, return (A rigid fit, MCPO full TRE, fit residual)."""
    us, mr, _, _, _, mri, usl = inputs(coh, c)
    usi, mri_img = nib.load(str(us)), nib.load(str(mr))
    aff, dims = V.box05(usi)
    fx = resample_to_grid(usi, aff, dims).astype(np.float32)
    mv = resample_to_grid(mri_img, aff, dims).astype(np.float32)
    pad = [(0, int(s % 2)) for s in fx.shape]
    ind, outd = Path("/tmp/mpd/in"), Path("/tmp/mpd/out")
    for p in (ind, outd):
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True)
    nib.save(nib.Nifti1Image(np.pad(fx, pad), aff), str(ind / "ReMIND2Reg_x0_0000.nii.gz"))
    nib.save(nib.Nifti1Image(np.pad(mv, pad), aff), str(ind / "ReMIND2Reg_x0_0001.nii.gz"))
    run_cmd([PY, MCPO, "--datasets_path", str(ind) + "/", "--result_path", str(outd) + "/"], d / "mcpo.log")
    ui = nib.load(str(outd / "disp_x0_0000_x0_0001.nii.gz"))
    u = np.asarray(ui.dataobj, np.float32)
    inv = np.linalg.inv(ui.affine)
    v = usl @ inv[:3, :3].T + inv[:3, 3]
    du = np.stack([map_coordinates(u[..., k], v.T, order=1, mode="nearest") for k in range(3)], -1)
    tre_full = float(np.linalg.norm(usl + du @ ui.affine[:3, :3].T - mri, axis=1).mean())
    fov = np.pad(fx, pad) > 0
    ijk = np.argwhere(fov[::4, ::4, ::4]) * 4
    p = ijk @ ui.affine[:3, :3].T + ui.affine[:3, 3]
    q = p + u[ijk[:, 0], ijk[:, 1], ijk[:, 2]] @ ui.affine[:3, :3].T
    Rm, t, inl = trimmed_rigid(p, q, keep=0.6, iters=4)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = Rm, t
    res = float(np.linalg.norm(p[inl] @ Rm.T + t - q[inl], axis=1).mean())
    shutil.rmtree("/tmp/mpd", ignore_errors=True)
    return A, tre_full, res


def run_case(coh, c):
    d = OUT / coh / c
    if (d / "meta.json").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    us, mr, sg, tum, exs, mri, usl = inputs(coh, c)
    A, tre_mcpo, fitres = mcpo_pose(coh, c, d)
    np.savetxt(d / "pose.txt", A)
    ang = float(np.degrees(np.arccos(np.clip((np.trace(A[:3, :3]) - 1) / 2, -1, 1))))
    tmp = Path("/tmp/mpd_rel")
    tmp.mkdir(exist_ok=True)
    mv_r, sg_r = tmp / "moving_rig.nii.gz", tmp / "synthseg_rig.nii.gz"
    relabel(mr, mv_r, A)
    relabel(sg, sg_r, A)
    targ = []
    if tum is not None:
        relabel(tum, tmp / "tumor_rig.nii.gz", A)
        targ = ["--tumor", tmp / "tumor_rig.nii.gz"]
    basis = d / "basis30.npz"
    if run_cmd([PY, SC.S / "biomech_basis.py", "--synthseg", sg_r, *targ, "--out", basis, "--h", "5.0", "--n-modes", "30"],
               d / "run.log") != 0 or not basis.exists():
        raise RuntimeError("basis failed")
    cmd = [PY, SC.S / "reg_modal.py", "--fixed", us, "--moving", mv_r, "--basis", basis, "--out", d / "paper",
           "--spacing", "1.0", "--iters", "200", "--lambda-eig", "3e-3", "--residual", "--residual-smooth-mm", "1.5"]
    if exs is not None:
        cmd += ["--exclude", exs, "--exclude-labels", "1,2", "--exclude-dilate-mm", "3"]
    if run_cmd(cmd, d / "run.log") != 0:
        raise RuntimeError("paper stage failed")
    (d / "paper/T2_warped.nii.gz").unlink(missing_ok=True)
    VE.paths = lambda case, us=us, mr=mr: (us, mr)
    VE.OUT = OUT / coh
    VE.deform(c, FM.ACCEPTED, "v2", A, basis)
    ap = lambda y: y @ A[:3, :3].T + A[:3, 3]                              # noqa: E731
    err = lambda y: float(np.linalg.norm(ap(y) - mri, axis=1).mean())     # noqa: E731
    mv2 = json.loads((d / "v2/meta.json").read_text())
    meta = dict(cohort=coh, case=c, initial=float(np.linalg.norm(usl - mri, axis=1).mean()), mcpo_full=tre_mcpo,
                mcpo_pose=err(usl), mcpo_pose_paper=err(usl + SC.disp_at(d / "paper/disp_mm.nii.gz", usl)),
                mcpo_pose_v2=err(usl + SC.disp_at(d / "v2/disp_mm.nii.gz", usl)), pose_rot_deg=ang, fit_res_mm=fitres,
                v2_fold=mv2["fold_pct"], v2_sdlogj=mv2["sdlogj"], wall_s=time.time() - t0)
    (d / "meta.json").write_text(json.dumps(meta))
    for f in tmp.glob("*.nii.gz"):
        f.unlink()
    print(f"[mpd] {coh} {c}: MCPO {tre_mcpo:.2f} | MCPO pose {meta['mcpo_pose']:.2f} (rot {ang:.1f} deg) | +paper "
          f"{meta['mcpo_pose_paper']:.2f} | +v2 {meta['mcpo_pose_v2']:.2f} | {meta['wall_s']:.0f} s", flush=True)


def do_run(only):
    todo = []
    if only in (None, "remind"):
        todo += [("remind", c) for c in REMIND]
    if only in (None, "resect"):
        S = M.scores()
        todo += [(coh, c) for coh in ORDER for c in sorted(k for (k2, k) in S if k2 == coh)]
    for coh, c in todo:
        try:
            run_case(coh, c)
        except Exception as ex:                                      # noqa: BLE001
            print(f"[mpd] {coh} {c} FAILED: {ex}", flush=True)
    print("MPD RUN DONE", flush=True)


def do_report():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*/*/meta.json"))]
    S = M.scores()
    nr = {(r["cohort"], r["case"]): float(r["l2r_niftyreg"]) for r in csv.DictReader(open(W / "results/l2r_niftyreg_baseline/PER_CASE.csv"))}
    v2rows = list(csv.DictReader(open(W / "results/v2/PER_CASE.csv")))
    v2a = {(r["cohort"], r["case"]): float(r["tre"]) for r in v2rows if r["arm"] == "R3_sym"}
    v2i = {(r["cohort"], r["case"]): float(r["tre"]) for r in v2rows if r["arm"] == "I4_fem"}
    nrv2 = {(r["cohort"], r["case"]): float(r["nr_v2"]) for r in csv.DictReader(open(W / "results/v2/nr_affine_cohorts/PER_CASE.csv"))}
    Tr = CR.reused()
    for r in rows:
        k = (r["cohort"], r["case"])
        if r["cohort"] == "remind":
            c = r["case"]
            r.update(paper_pipeline=Tr[("ours", "paper", c)], v2_pipeline=Tr[("ours", "v2", c)], nr=Tr[("nr", "none", c)],
                     nr_v2=Tr[("nr", "v2", c)])
        else:
            ab = bool(S[k]["abstained"])
            r.update(paper_pipeline=S[k]["final"], v2_pipeline=(v2i if ab else v2a)[k], nr=nr[k], nr_v2=nrv2[k])
    cols = ["initial", "nr", "mcpo_full", "paper_pipeline", "v2_pipeline", "nr_v2", "mcpo_pose", "mcpo_pose_paper", "mcpo_pose_v2"]
    L = ["# MCPO pose (rigid fit of MCPO's field) + our deformable stages", "", "Mean TRE (mm).", "",
         "| cohort | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    groups = [("RESECT post", "resect_after"), ("BITE", "bite_g4"), ("RESECT pre", "resect_before"), ("RESECT+BITE (55)", None),
              ("ReMIND (provisional)", "remind")]
    for lab, coh in groups:
        rr = [r for r in rows if (r["cohort"] == coh if coh else r["cohort"] != "remind")]
        if rr:
            L.append(f"| {lab} | {len(rr)} | " + " | ".join(f"{np.mean([r[k] for r in rr]):.3f}" for k in cols) + " |")
    L.append("")
    for lab, coh in (("RESECT+BITE (55)", None), ("ReMIND", "remind")):
        rr = [r for r in rows if (r["cohort"] == coh if coh else r["cohort"] != "remind")]
        if len(rr) < 5:
            continue
        L.append(f"**{lab}**")
        for a, b in (("mcpo_pose_v2", "mcpo_full"), ("mcpo_pose_v2", "v2_pipeline"), ("mcpo_pose_v2", "nr"), ("mcpo_pose_v2", "nr_v2"),
                     ("mcpo_pose_v2", "mcpo_pose_paper"), ("mcpo_pose_v2", "mcpo_pose")):
            x = np.array([r[a] for r in rr]); y = np.array([r[b] for r in rr])
            p = stats.wilcoxon(x, y, zero_method="zsplit").pvalue
            L.append(f"- {a} vs {b}: {x.mean():.3f} vs {y.mean():.3f}; better {int((x < y).sum())}/{len(rr)}; worst {np.max(x - y):+.2f}; p = {p:.4f}")
        L.append(f"- v2 stage on the MCPO pose: folding max {max(r['v2_fold'] for r in rr):.3f} %, SDLogJ mean {np.mean([r['v2_sdlogj'] for r in rr]):.3f}; "
                 f"MCPO pose rotation {min(r['pose_rot_deg'] for r in rr):.1f}–{max(r['pose_rot_deg'] for r in rr):.1f} deg")
        L.append("")
    (OUT / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--only", choices=["remind", "resect"])
    a = ap.parse_args()
    do_run(a.only) if a.mode == "run" else do_report()


if __name__ == "__main__":
    main()
