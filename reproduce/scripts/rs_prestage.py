"""rs_prestage.py -- arm A used as a rigid PRE-STAGE, followed by the frozen
method, end to end.

**SEPARATE, DECLARED ARM.**  Nothing published is modified: every output goes
to `results/capture_range/armA_modal/`, and the frozen hyper-parameters are
reproduced verbatim (they are NOT re-tuned here).

Pipeline, mirroring `oracleinit` / `rigidinit` in `rigid_decomposition.py`:

  1. A  = the rigid returned by `rigid_search.py` (US world -> MR world),
         gated or raw (`--aff`).  A = I reproduces the frozen arm exactly.
  2. The moving MR, its SynthSeg and (ReMIND) the MR tumour segmentation are
     expressed in the pre-aligned frame.  This is done EXACTLY, by relabelling
     the world affine rather than resampling:

         M'(x) = M(A x)   <=>   same voxel array, affine  P = A^-1 . M_affine

     so the pre-stage introduces no interpolation blur and cannot crop the
     moving image (unlike `reg_resample -ref <MR> -flo <MR>`).
  3. The modal basis is REBUILT on the pre-aligned SynthSeg (h 5.0, 30 modes) --
     necessary because the basis is defined in the moving anatomy's frame.
  4. `reg_modal.py` is re-run with the identical frozen hyper-parameters.
  5. TRE = | A (x_us + u(x_us)) - x_mr |, u sampled from disp_mm.nii.gz with the
     same code path as every other arm (`rigid_decomposition.sample_disp`).

Usage:
    python rs_prestage.py --dataset resect --case Case12
    python rs_prestage.py --dataset resect --case Case3 --identity   # frozen ref
    python rs_prestage.py --dataset remind --case ReMIND-045
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigid_search import (WD, OUT as ARMA, case_paths, case_landmarks,      # noqa
                          RESECT_TUNE, RESECT_TEST, REMIND_PIDS)
from rigid_decomposition import sample_disp                                 # noqa

PY = "/home/ncrhurh/anaconda3/envs/monai/bin/python"
S = WD / "scripts"
ROOT = WD / "results/capture_range/armA_modal"

# frozen hyper-parameters, copied verbatim from the published batch scripts
FROZEN = {
    "resect": dict(  # batch_resect_s15.sh
        lam_eig="3e-4", exclude_labels="1", exclude_dilate="3",
        seg=lambda c: WD / f"data/resect/synthseg_out/{c}-FLAIR_synthseg.nii.gz",
        tumor=lambda c: None),
    "remind": dict(  # batch_smooth_test.sh
        lam_eig="3e-3", exclude_labels="1,2", exclude_dilate="3",
        seg=lambda c: Path("/mnt/c/ReMIND_work/archive") / f"data/synthseg_out/preop/{c}_T2_synthseg.nii.gz",
        tumor=lambda c: Path("/mnt/c/ReMIND_work/archive") / f"data/segmentations/MRI/{c}-pre-mri-segmentation.nii.gz"),
}


def relabel(src, dst, A):
    """Write an image whose world map is  x -> src(A x)  (exact, no resampling)."""
    img = nib.load(str(src))
    new = np.linalg.inv(A) @ img.affine
    out = nib.Nifti1Image(np.asarray(img.dataobj), new, img.header)
    out.set_sform(new, code=1)
    out.set_qform(new, code=1)
    nib.save(out, str(dst))


def run(cmd, log):
    with open(log, "a") as fh:
        return subprocess.run([str(x) for x in cmd], stdout=fh,
                              stderr=subprocess.STDOUT).returncode


def prestage(dataset, case, A, outdir, force=False):
    cfg = FROZEN[dataset]
    fx, mv, ex, _ = case_paths(dataset, case)
    seg, tum = cfg["seg"](case), cfg["tumor"](case)
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "prestage.log"
    t0 = time.time()

    mv_r, seg_r = outdir / "moving_rig.nii.gz", outdir / "synthseg_rig.nii.gz"
    relabel(mv, mv_r, A)
    relabel(seg, seg_r, A)
    targ = []
    if tum is not None and Path(tum).exists():
        relabel(tum, outdir / "tumor_rig.nii.gz", A)
        targ = ["--tumor", outdir / "tumor_rig.nii.gz"]

    basis = outdir / "basis_rig.npz"
    if force or not basis.exists():
        rc = run([PY, S / "biomech_basis.py", "--synthseg", seg_r, *targ,
                  "--out", basis, "--h", "5.0", "--n-modes", "30"], log)
        if rc != 0 or not basis.exists():
            return {"case": case, "status": "basis_failed"}

    m = outdir / "modal_residual_s15"
    if force or not (m / "disp_mm.nii.gz").exists():
        m.mkdir(exist_ok=True)
        cmd = [PY, S / "reg_modal.py", "--fixed", fx, "--moving", mv_r,
               "--basis", basis, "--out", m, "--spacing", "1.0",
               "--iters", "200", "--lambda-eig", cfg["lam_eig"],
               "--residual", "--residual-smooth-mm", "1.5"]
        if ex is not None:
            cmd += ["--exclude", ex, "--exclude-labels", cfg["exclude_labels"],
                    "--exclude-dilate-mm", cfg["exclude_dilate"]]
        rc = run(cmd, log)
        if rc != 0 or not (m / "disp_mm.nii.gz").exists():
            return {"case": case, "status": "modal_failed"}

    us, mri = case_landmarks(dataset, case)
    u, outside = sample_disp(nib.load(str(m / "disp_mm.nii.gz")), us)
    pred = (us + u) @ A[:3, :3].T + A[:3, 3]
    res = {"case": case, "dataset": dataset, "status": "ok",
           "n_landmarks": int(len(us)), "n_outside": int(outside.sum()),
           "tre_initial": float(np.linalg.norm(us - mri, axis=1).mean()),
           "tre": float(np.linalg.norm(pred - mri, axis=1).mean()),
           "tre_sd": float(np.linalg.norm(pred - mri, axis=1).std()),
           "rigid_only_tre": float(np.linalg.norm(
               us @ A[:3, :3].T + A[:3, 3] - mri, axis=1).mean()),
           "A": A.tolist(), "wall_s": time.time() - t0}
    (outdir / "result.json").write_text(json.dumps(res, indent=2))
    # the pre-aligned moving is large and reproducible; keep only the seg
    for p in (mv_r, m / "T2_warped.nii.gz"):
        if p.exists():
            p.unlink()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", choices=["resect", "remind"], default="resect")
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--split", choices=["TUNE", "TEST", "ALL"], default="ALL")
    ap.add_argument("--identity", action="store_true",
                    help="A = I: reproduces the frozen arm (reference)")
    ap.add_argument("--raw", action="store_true",
                    help="use the UNGATED rigid (aff.txt) instead of aff_gated.txt")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--src", default="armA",
                    help="which arm-A specification supplies the rigid "
                         "(directory name under results/capture_range/)")
    ap.add_argument("--arm", default=None,
                    help="output sub-directory under armA_modal/ "
                         "(default: frozen_ref / gated / ungated)")
    a = ap.parse_args()
    src_root = WD / "results/capture_range" / a.src

    if a.dataset == "resect":
        nums = {"TUNE": RESECT_TUNE, "TEST": RESECT_TEST,
                "ALL": sorted(RESECT_TUNE + RESECT_TEST)}[a.split]
        cases = a.case or [f"Case{c}" for c in nums]
    else:
        cases = a.case or REMIND_PIDS

    arm = a.arm or ("frozen_ref" if a.identity else
                    ("ungated" if a.raw else "gated"))
    for case in cases:
        d = ROOT / arm / a.dataset / case
        if (d / "result.json").exists() and not a.force:
            print(f"SKIP {case}", flush=True)
            continue
        if a.identity:
            A = np.eye(4)
        else:
            f = src_root / a.dataset / case / ("aff.txt" if a.raw
                                               else "aff_gated.txt")
            if not f.exists():
                print(f"MISS {case} (no {f.name})", flush=True)
                continue
            A = np.loadtxt(f)
            if np.allclose(A, np.eye(4), atol=1e-9):
                # the gate declined: the arm IS the frozen method here, so the
                # frozen reference is reused verbatim rather than recomputed
                d.mkdir(parents=True, exist_ok=True)
                (d / "result.json").write_text(json.dumps(
                    {"case": case, "dataset": a.dataset, "status": "abstain",
                     "use": str(ROOT / "frozen_ref" / a.dataset / case)},
                    indent=2))
                print(f"ABSTAIN {case} (gate declined; frozen result stands)",
                      flush=True)
                continue
        print(f"=== {arm} {case} ===", flush=True)
        r = prestage(a.dataset, case, A, d, force=a.force)
        print(json.dumps(r, default=str)[:400], flush=True)


if __name__ == "__main__":
    main()
