"""affine_cascade.py -- does an AFFINE stage between the global rigid and the
deformable stage help?  Classical cascade rigid -> affine -> deformable versus
our rigid -> modal, on RESECT, all 20 cases.

**Additional declared arm.  Nothing frozen or published is modified**: every
output goes to `results/capture_range/affine_cascade/`, and the deformable
hyper-parameters are reproduced verbatim from the published batch scripts
(lambda_eig 3e-4, 200 iters, residual smoothing 1.5 mm, cavity label 1 dilated
3 mm, spacing 1.0) -- they are NOT re-tuned here.

Arms (all six produced by THIS runner, so the code path is identical):

  frozen_modal          A = I                      + modal
  frozen_modal_res      A = I                      + modal + smoothed residual
  rigid_modal           A = arm A global rigid     + modal
  rigid_modal_res       A = arm A global rigid     + modal + smoothed residual
  affine_modal          A = rigid + bounded affine + modal
  affine_modal_res      A = rigid + bounded affine + modal + smoothed residual

plus the transform-only rows (no deformable stage), scored from the same
matrices: `rigid_only`, `affine_only`, `affine_unbounded_only`.

For every arm the moving FLAIR and its SynthSeg are expressed in the
pre-aligned frame and **the modal basis is rebuilt there** -- the basis is
defined in the moving anatomy's frame, so it must be consistent with the
pre-stage.  The frame change is applied EXACTLY by relabelling the world affine
(`rs_prestage.relabel`: M'(x) = M(A x)), never by resampling, so no
interpolation blur and no cropping is introduced -- this is valid for a general
affine A, not just a rigid one.

TRE = | A (x_us + u(x_us)) - x_mr |, u sampled with the same code path as every
other arm in the project (`rigid_decomposition.sample_disp`).

Usage:
    python affine_cascade.py --split ALL
    python affine_cascade.py --case Case18 --arms rigid_modal,affine_modal
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rigid_decomposition import sample_disp                            # noqa
from rigid_search import (WD, case_landmarks, case_paths,              # noqa
                          RESECT_TEST, RESECT_TUNE)
from rs_prestage import FROZEN                                         # noqa


def relabel(src, dst, A):
    """Write an image whose world map is x -> src(A x), EXACTLY (no resampling):
    same voxel array, affine P = A^-1 . src_affine.

    Generalises `rs_prestage.relabel` (which is rigid-only) to an arbitrary
    affine.  The difference matters: a NIfTI **qform cannot represent shear**,
    so writing one for a stretched/sheared P silently stores an orthogonalised
    matrix (measured: up to 0.069 mm off on Case18's bounded affine).  Nothing
    in this pipeline reads the qform -- nibabel's `.affine` returns the sform
    when sform_code > 0 -- but an inconsistent header is a trap for any other
    reader, so the qform is marked UNDEFINED (code 0) instead of wrong."""
    img = nib.load(str(src))
    P = np.linalg.inv(A) @ img.affine
    out = nib.Nifti1Image(np.asarray(img.dataobj), P, img.header)
    out.set_sform(P, code=1)
    out.set_qform(None, code=0)
    return nib.save(out, str(dst))

PY = "/home/ncrhurh/anaconda3/envs/monai/bin/python"
S = WD / "scripts"
OUT = WD / "results/capture_range/affine_cascade"
ARMA = WD / "results/capture_range/armA"

# arm -> (pre-stage transform key, use bounded dense residual)
ARMS = {
    "frozen_modal":     ("identity", False),
    "frozen_modal_res": ("identity", True),
    "rigid_modal":      ("rigid", False),
    "rigid_modal_res":  ("rigid", True),
    "affine_modal":     ("affine", False),
    "affine_modal_res": ("affine", True),
}


def transform(dataset, case, key):
    if key == "identity":
        return np.eye(4)
    if key == "rigid":
        return np.array(json.loads(
            (ARMA / dataset / case / "search.json").read_text())["A"])
    if key in ("affine", "affine_unbounded"):
        tag = "bounded" if key == "affine" else "unbounded"
        return np.loadtxt(OUT / dataset / case / f"aff_{tag}.txt")
    raise ValueError(key)


def frame(dataset, case, key, log):
    """Moving + SynthSeg (+ tumour) expressed in the pre-aligned frame, and the
    modal basis rebuilt there.  Returns (moving_path, basis_path)."""
    cfg = FROZEN[dataset]
    _, mv, _, _ = case_paths(dataset, case)
    if key == "identity":                      # frozen frame: reuse the shipped basis
        b = WD / f"data/{dataset}/basis/{case}_basis.npz"
        if not b.exists():
            b = WD / f"data/basis/{case}_basis.npz"
        return mv, b
    A = transform(dataset, case, key)
    d = OUT / dataset / case / f"frame_{key}"
    d.mkdir(parents=True, exist_ok=True)
    mv_r, seg_r = d / "moving_rig.nii.gz", d / "synthseg_rig.nii.gz"
    if not mv_r.exists():
        relabel(mv, mv_r, A)
    if not seg_r.exists():
        relabel(cfg["seg"](case), seg_r, A)
    targ = []
    tum = cfg["tumor"](case)
    if tum is not None and Path(tum).exists():
        if not (d / "tumor_rig.nii.gz").exists():
            relabel(tum, d / "tumor_rig.nii.gz", A)
        targ = ["--tumor", d / "tumor_rig.nii.gz"]
    basis = d / "basis_rig.npz"
    if not basis.exists():
        t0 = time.time()
        rc = run([PY, S / "biomech_basis.py", "--synthseg", seg_r, *targ,
                  "--out", basis, "--h", "5.0", "--n-modes", "30"],
                 d / "basis.log")
        print(f"    basis[{key}] rc={rc} ({time.time()-t0:.0f}s)", flush=True)
    return mv_r, basis


def run(cmd, log):
    with open(log, "a") as fh:
        return subprocess.run([str(x) for x in cmd], stdout=fh,
                              stderr=subprocess.STDOUT).returncode


def run_arm(dataset, case, arm, force=False):
    key, residual = ARMS[arm]
    cfg = FROZEN[dataset]
    fx, _, ex, _ = case_paths(dataset, case)
    d = OUT / dataset / case / arm
    d.mkdir(parents=True, exist_ok=True)
    if (d / "result.json").exists() and not force:
        return json.loads((d / "result.json").read_text())
    t0 = time.time()
    mv, basis = frame(dataset, case, key, d / "run.log")
    if not Path(basis).exists():
        return {"case": case, "arm": arm, "status": "basis_failed"}
    if force or not (d / "disp_mm.nii.gz").exists():
        cmd = [PY, S / "reg_modal.py", "--fixed", fx, "--moving", mv,
               "--basis", basis, "--out", d, "--spacing", "1.0",
               "--iters", "200", "--lambda-eig", cfg["lam_eig"]]
        if residual:
            cmd += ["--residual", "--residual-smooth-mm", "1.5"]
        if ex is not None:
            cmd += ["--exclude", ex, "--exclude-labels", cfg["exclude_labels"],
                    "--exclude-dilate-mm", cfg["exclude_dilate"]]
        rc = run(cmd, d / "run.log")
        if rc != 0 or not (d / "disp_mm.nii.gz").exists():
            return {"case": case, "arm": arm, "status": "modal_failed"}
    A = transform(dataset, case, key)
    us, mri = case_landmarks(dataset, case)
    u, outside = sample_disp(nib.load(str(d / "disp_mm.nii.gz")), us)
    pred = (us + u) @ A[:3, :3].T + A[:3, 3]
    tre = np.linalg.norm(pred - mri, axis=1)
    res = {"case": case, "dataset": dataset, "arm": arm, "status": "ok",
           "prestage": key, "residual": residual,
           "n_landmarks": int(len(us)), "n_outside": int(outside.sum()),
           "tre": float(tre.mean()), "tre_sd": float(tre.std()),
           "tre_per_landmark": tre.tolist(), "wall_s": time.time() - t0}
    (d / "result.json").write_text(json.dumps(res, indent=2))
    for p in (d / "T2_warped.nii.gz", d / "US_fixed_grid.nii.gz",
              d / "jacobian.nii.gz"):
        if p.exists():
            p.unlink()
    return res


def cleanup(dataset, case):
    """Drop the relabelled moving volumes once every arm of the case is done
    (they are large and exactly reproducible from aff_*.txt by `relabel`)."""
    for key in ("rigid", "affine"):
        p = OUT / dataset / case / f"frame_{key}" / "moving_rig.nii.gz"
        if p.exists():
            p.unlink()


def transform_only(dataset, case):
    """TRE of the pre-stage transforms alone (no deformable stage)."""
    us, mri = case_landmarks(dataset, case)
    out = {"initial": float(np.linalg.norm(us - mri, axis=1).mean())}
    for lab, key in [("rigid_only", "rigid"), ("affine_only", "affine"),
                     ("affine_unbounded_only", "affine_unbounded")]:
        try:
            A = transform(dataset, case, key)
        except (FileNotFoundError, OSError):
            continue
        out[lab] = float(np.linalg.norm(
            us @ A[:3, :3].T + A[:3, 3] - mri, axis=1).mean())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="resect")
    ap.add_argument("--case", nargs="*")
    ap.add_argument("--split", choices=["TUNE", "TEST", "ALL"], default="ALL")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    nums = {"TUNE": RESECT_TUNE, "TEST": RESECT_TEST,
            "ALL": sorted(RESECT_TUNE + RESECT_TEST)}[a.split]
    cases = a.case or [f"Case{c}" for c in nums]
    for c in cases:
        print(f"=== {c} ===", flush=True)
        row = transform_only(a.dataset, c)
        print("    " + "  ".join(f"{k} {v:.3f}" for k, v in row.items()),
              flush=True)
        for arm in a.arms.split(","):
            r = run_arm(a.dataset, c, arm, force=a.force)
            print(f"    {arm:18s} {r.get('status')} "
                  f"TRE {r.get('tre', float('nan')):.3f} "
                  f"({r.get('wall_s', 0):.0f}s)", flush=True)
        (OUT / a.dataset / c / "transform_only.json").write_text(
            json.dumps(row, indent=2))
        cleanup(a.dataset, c)


if __name__ == "__main__":
    main()
