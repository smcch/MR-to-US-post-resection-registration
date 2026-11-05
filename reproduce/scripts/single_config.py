"""One configuration for every arm of the paper's three landmark cohorts.

The shipped method is: frozen pose search (TIGHT/K12) -> guarded 0.5 mm polish
-> abstention if the search rotation is >= 9.5 deg -> modal stage.  This
script runs the modal stage with ONE configuration for both roles it plays:
  pose      the accepted polished pose (results/rigid_gate/search_polish/<cohort>/<case>/aff.txt);
            the MR and its SynthSeg are relabelled by A (no resampling) and
            the FEM basis is rebuilt in the posed frame
  identity  the same modal stage from the navigation alignment: the
            deformation-only arm and the abstention fallback
Candidate configurations (both lambda_eig = 3e-3, no cavity/segmentation mask):
  m30r   30 modes + bounded smoothed residual (1.5 mm)
  m20    20 modes, no residual
Resumable.  Output: results/single_config/<cfg>/<cohort>/<case>/{pose,identity}/modal/disp_mm.nii.gz

Usage:
  python single_config.py run --cfg m20 --cohort bite_g4 [--arms pose identity] [--shard 0/2]
  python single_config.py score --cfg m20 [--cohort ...]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rg_cohorts                                                    # noqa
from rs_prestage import relabel                                      # noqa
from resect_tags import load_tag_pairs as rt                         # noqa
from bite_tags import load_tag_pairs as bt                           # noqa

W = Path("/mnt/c/ReMIND_work")
R = W / "results"
S = W / "scripts"
PY = "/home/ncrhurh/anaconda3/envs/monai/bin/python"
OUT = R / "single_config"
DEMO = R / "bratislava_demo"
INJ = R / "landmark_free/poseinject_bratislava"
SAT = 9.5
LAM = "3e-3"
CFG = {"m30r": dict(modes=30, residual=True), "m20": dict(modes=20, residual=False),
       # development ablation on BITE (pose arm only), around the selected m30r
       "m30": dict(modes=30, residual=False), "m20r": dict(modes=20, residual=True),
       "m50r": dict(modes=50, residual=True), "m80r": dict(modes=80, residual=True)}
TAG = {"resect_after": lambda c: (W / f"data/resect/{c}/{c}-MRI-afterUS.tag", rt),
       "resect_before": lambda c: (W / f"data/resect_before/{c}/{c}-MRI-beforeUS.tag", rt),
       "bite_g4": lambda c: (W / f"data/bite/group4/{c}/Tags.tag", bt)}
COHORTS = ["resect_after", "bite_g4", "resect_before"]


def cases(coh):
    return [(r["case"], float(r["rot_deg"]))
            for r in csv.DictReader(open(R / f"rigid_gate/search_tight/{coh}/PER_CASE.csv"))]


def pose_of(coh, c):
    return np.loadtxt(R / f"rigid_gate/search_polish/{coh}/{c}/aff.txt")


def run(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        rc = subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"{Path(str(cmd[1] if str(cmd[0]).endswith('python') else cmd[0])).name} rc={rc}")


def basis_for(out, seg_r, n, log, basis_src=None):
    """An n-mode basis for this run.  Modes are ascending in energy, so the
    first n modes of a larger solve are the n-mode basis: an existing basis
    with >= n modes -- in this run, in basis_src, or in any sibling
    configuration of the same case and arm (same SynthSeg, same pose) -- is
    truncated instead of re-solved.  Otherwise a 30-mode (80 if n > 30)
    basis is solved here."""
    sibs = [out] + ([basis_src] if basis_src is not None else [])
    rel = out.relative_to(OUT).parts if out.is_relative_to(OUT) else ()
    if len(rel) == 4:                                  # <cfg>/<cohort>/<case>/<arm>
        sibs += [OUT / k / rel[1] / rel[2] / rel[3] for k in CFG]
    for K in (n, 30, 80):
        if K < n:
            continue
        for d in sibs:
            p = d / f"basis{K}.npz"
            if p.exists():
                if K == n:
                    return p
                t = out / f"basis{n}.npz"
                run([PY, S / "basis_truncate.py", p, t, str(n)], log)
                return t
    K = 30 if n <= 30 else 80
    basis = out / f"basis{K}.npz"
    run([PY, S / "biomech_basis.py", "--synthseg", seg_r, "--out", basis, "--h", "5.0", "--n-modes", str(K)], log)
    if K == n:
        return basis
    t = out / f"basis{n}.npz"
    run([PY, S / "basis_truncate.py", basis, t, str(n)], log)
    return t


def modal_arm(fx, mv, seg, A, out, cfg, lam=LAM, basis_src=None, material=None, tumor=None, **meta):
    """basis_src: an existing run directory of the same case/arm whose basis
    (built on the identically relabelled SynthSeg) is reused (lambda sweep,
    moving-contrast check).  material: a basis_variant.py material variant
    (sensitivity analysis), always solved here; tumor: its tumour label map."""
    disp = out / "modal/disp_mm.nii.gz"
    if disp.exists():
        return disp
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    t0 = time.time()
    mv_r, seg_r = out / "moving_rig.nii.gz", out / "synthseg_rig.nii.gz"
    relabel(mv, mv_r, A)
    relabel(seg, seg_r, A)
    if material:
        b = out / "basis_mat.npz"
        targ = []
        if tumor is not None:
            relabel(tumor, out / "tumor_rig.nii.gz", A)
            targ = ["--tumor", out / "tumor_rig.nii.gz"]
        if not b.exists():
            run([PY, S / "basis_variant.py", material, "--synthseg", seg_r, *targ, "--out", b, "--h", "5.0",
                 "--n-modes", str(max(30, CFG[cfg]["modes"]))], log)
        if CFG[cfg]["modes"] != 30:
            run([PY, S / "basis_truncate.py", b, out / "basis_mat_k.npz", str(CFG[cfg]["modes"])], log)
            b = out / "basis_mat_k.npz"
    else:
        b = basis_for(out, seg_r, CFG[cfg]["modes"], log, basis_src)
    cmd = [PY, S / "reg_modal.py", "--fixed", fx, "--moving", mv_r, "--basis", b, "--out", out / "modal",
           "--spacing", "1.0", "--iters", "200", "--lambda-eig", lam]
    if CFG[cfg]["residual"]:
        cmd += ["--residual", "--residual-smooth-mm", "1.5"]
    run(cmd, log)
    for p in (mv_r, out / "modal/T2_warped.nii.gz"):
        p.unlink(missing_ok=True)
    (out / "arm.json").write_text(json.dumps(dict(**meta, cfg=cfg, A=np.asarray(A).tolist(), basis=str(b),
                                                  lambda_eig=lam, **CFG[cfg], wall_s=time.time() - t0)))
    return disp


def do_external(a):
    """External site (no landmarks; figures and visual export only): the pose
    of the injection study's uninjected run, then the same modal stage."""
    todo = sorted(p.name for p in DEMO.iterdir() if p.is_dir())
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        todo = todo[i::n]
    for sweep in todo:
        subj, sw = sweep.split("_", 1)
        stage = "postop" if sw.startswith("postop") else "preop"
        try:
            A = np.loadtxt(INJ / stage / f"base/bratislava/{sweep}/aff.txt")
            modal_arm(W / f"data/bratislava/{subj}/{sw}_in_t2.nii.gz", W / f"data/bratislava/{subj}/t2.nii.gz",
                      W / f"data/bratislava/synthseg_out/{subj}_t2_synthseg.nii.gz", A,
                      OUT / a.cfg / "external" / sweep / "pose", a.cfg, cohort="external", case=sweep)
            print(f"[single] {a.cfg} external {sweep} ok", flush=True)
        except Exception as e:                                       # noqa: BLE001
            print(f"[single] {a.cfg} external {sweep} FAILED: {e}", flush=True)
    print("SINGLE EXTERNAL DONE", flush=True)


def do_run(a):
    spec = rg_cohorts.install(a.cohort)
    todo = []
    for c, rot in cases(a.cohort):
        if a.cases and c not in a.cases:
            continue
        # pose arm on every case: on abstained cases it is the rejected
        # (counterfactual) pose + modal result, never the reported one.
        # --gated-only: just the arm the gate selects (sensitivity analyses)
        if "pose" in a.arms and not (a.gated_only and rot >= SAT):
            todo.append((c, "pose"))
        if "identity" in a.arms and not (a.gated_only and rot < SAT):
            todo.append((c, "identity"))
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        todo = todo[i::n]
    for c, arm in todo:
        out = a.root / a.cohort / c / arm
        try:
            A = pose_of(a.cohort, c) if arm == "pose" else np.eye(4)
            fx, mv = spec["paths"](c)[:2]
            if a.moving == "t1":     # RESECT ceT1, header composed with the measured T1->FLAIR rigid
                mv = W / f"data/resect_t1/{c}/{c}-T1-rig.nii.gz"
            tum = W / f"data/resect/{c}/{c}-US-before-tumor.nii.gz" if a.material == "ratio4" else None
            modal_arm(fx, mv, spec["seg"](c), A, out, a.cfg, lam=a.lam or LAM,
                      basis_src=(OUT / a.cfg / a.cohort / c / arm) if (a.lam or a.moving == "t1") else None,
                      material=a.material, tumor=tum, cohort=a.cohort, case=c)
            print(f"[single] {a.cfg} {a.cohort} {c} {arm} ok", flush=True)
        except Exception as e:                                       # noqa: BLE001
            print(f"[single] {a.cfg} {a.cohort} {c} {arm} FAILED: {e}", flush=True)
    print("SINGLE RUN DONE", flush=True)


# ---------------------------------------------------------------------- score
def disp_at(p, pts):
    u = nib.load(str(p))
    a = np.asarray(u.dataobj, np.float32)
    inv = np.linalg.inv(u.affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    return np.stack([map_coordinates(a[..., k], [v[:, 0], v[:, 1], v[:, 2]], order=1, mode="nearest")
                     for k in range(3)], -1)


def do_score(a):
    rows, lms = [], []
    for coh in a.cohort_list:
        for c, rot in cases(coh):
            tagp, loader = TAG[coh](c)
            mri, us = loader(str(tagp))
            mri, us = np.asarray(mri, float), np.asarray(us, float)
            base = a.root / coh / c
            e0 = np.linalg.norm(us - mri, axis=1)
            idp = base / "identity/modal/disp_mm.nii.gz"
            e_id = np.linalg.norm(us + disp_at(idp, us) - mri, axis=1) if idp.exists() else None
            ab = rot >= SAT
            A = pose_of(coh, c)
            e_pose_only = np.linalg.norm(us @ A[:3, :3].T + A[:3, 3] - mri, axis=1)
            pp = base / "pose/modal/disp_mm.nii.gz"
            e_pose = None
            if pp.exists():                                  # abstained: the rejected (counterfactual) pose
                y = us + disp_at(pp, us)
                e_pose = np.linalg.norm(y @ A[:3, :3].T + A[:3, 3] - mri, axis=1)
            e_fin = e_id if ab else e_pose
            if e_fin is None:
                print(f"[score] {coh} {c}: arms missing", flush=True)
                continue
            rows.append(dict(cohort=coh, case=c, n=len(us), rot_deg=round(rot, 3), abstained=ab,
                             initial=e0.mean(), final=e_fin.mean(),
                             deform_only=(e_id.mean() if e_id is not None else np.nan),
                             pose_only=e_pose_only.mean(),
                             pose_modal=(e_pose.mean() if e_pose is not None else np.nan),
                             initial_tre30=np.percentile(e0, 70), final_tre30=np.percentile(e_fin, 70)))
            for k in range(len(us)):
                lms.append(dict(cohort=coh, case=c, lm=k, initial=e0[k], final=e_fin[k]))
    a.root.mkdir(parents=True, exist_ok=True)
    with open(a.root / "SCORES.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(a.root / "LANDMARKS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lms[0]))
        w.writeheader()
        for r in lms:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    for coh in a.cohort_list:
        rr = [r for r in rows if r["cohort"] == coh]
        if not rr:
            continue
        f = np.array([r["final"] for r in rr])
        acc = [r for r in rr if not r["abstained"]]
        print(f"{a.cfg} {coh}: n={len(rr)} abst={sum(r['abstained'] for r in rr)} "
              f"initial {np.mean([r['initial'] for r in rr]):.3f} final {f.mean():.3f} +- {f.std(ddof=1):.2f} "
              f"(median {np.median(f):.2f}) | deform-only {np.mean([r['deform_only'] for r in rr]):.3f} | "
              f"accepted: pose {np.mean([r['pose_only'] for r in acc]):.3f} -> +modal "
              f"{np.mean([r['pose_modal'] for r in acc]):.3f} | TRE30 {np.mean([r['final_tre30'] for r in rr]):.2f}",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score", "external"])
    ap.add_argument("--cfg", required=True, choices=sorted(CFG))
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--arms", nargs="+", default=["pose", "identity"])
    ap.add_argument("--shard", default=None)
    ap.add_argument("--lam", default=None,
                    help="lambda_eig override (sensitivity sweep only): output in <cfg>_lam<lam>/, "
                         "basis reused from the <cfg> run of the same case and arm")
    ap.add_argument("--moving", choices=["default", "t1"], default="default",
                    help="t1: RESECT ceT1 as the moving image (contrast check); basis reused")
    ap.add_argument("--material", choices=["ratio4", "ratioCSFx3"], default=None,
                    help="material-variant basis (sensitivity analysis, basis_variant.py)")
    ap.add_argument("--gated-only", action="store_true", help="run only the arm the gate selects")
    ap.add_argument("--cases", nargs="*", default=None)
    a = ap.parse_args()
    a.root = OUT / (a.cfg + (f"_lam{a.lam}" if a.lam else "") + ("_t1" if a.moving == "t1" else "")
                    + (f"_mat{a.material}" if a.material else ""))
    if a.cmd == "run":
        do_run(a)
    elif a.cmd == "external":
        do_external(a)
    else:
        a.cohort_list = [a.cohort] if a.cohort else COHORTS
        do_score(a)
