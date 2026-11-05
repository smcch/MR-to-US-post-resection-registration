#!/usr/bin/env python
"""The "challenge affine": the NiftyReg affine baseline of the Learn2Reg 2024 ReMIND2Reg challenge, as used in
the paper to initialize the FE refinement.

Baseline recipe (github.com/ReubenDo/ReMIND2Reg, niftyreg/run_niftyreg.py), verbatim flags:
    reg_aladin -ref iUS -flo MR -rmask iUS_mask -noSym -nac -ln 2 -lp 10        (rigid, then affine)
As in the challenge data, the iUS is first resampled to a 0.5 mm isotropic box covering its field of view and
the reference mask is the iUS field of view (iUS > 0).

Inputs:  --us iUS sweep (NIfTI, navigation frame), --mr preoperative MR (NIfTI), --out directory.
Output:  <out>/aff.txt, a 4x4 matrix mapping iUS world coordinates to MR world coordinates (NiftyReg convention
         Affine * Ref = Flo); this is the --affine of fe_refine.py.  <out>/reg_aladin.log keeps the command and
         its output.
Options: --crop-fov  compute the 0.5 mm box from the non-zero iUS voxels (+2 mm) instead of the image extent;
         use it when the sweep is stored on a larger zero-padded grid (the paper used it for ReMIND and the
         external site, not for RESECT/BITE whose volumes are already bounded by the field of view).
NiftyReg 1.5.69 binaries are found in $NIFTYREG_BIN or on the PATH.  reg_aladin occasionally segfaults with
many OpenMP threads; the run is retried, the last attempts with 2 threads (identical result).
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fe_refinement"))
from reg_core import resample_to_grid                                # noqa: E402


def half_mm_box(img, crop_fov=False, margin=2.0):
    """0.5 mm isotropic box over the image extent, or over the non-zero voxels + margin when crop_fov."""
    if crop_fov:
        msk = np.asarray(img.dataobj) > 0
        idx = [np.where(msk.any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
        lo_i = [float(i.min()) for i in idx]
        hi_i = [float(i.max() + 1) for i in idx]
        c = np.array([[i, j, k] for i in (lo_i[0], hi_i[0]) for j in (lo_i[1], hi_i[1]) for k in (lo_i[2], hi_i[2])])
    else:
        sh = np.array(img.shape[:3], float)
        c = np.array([[i * sh[0], j * sh[1], k * sh[2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    w = c @ img.affine[:3, :3].T + img.affine[:3, 3]
    lo, hi = w.min(0), w.max(0)
    if crop_fov:
        lo, hi = lo - margin, hi + margin
    dims = np.maximum(np.ceil((hi - lo) / 0.5).astype(int), 8)
    aff = np.diag([0.5, 0.5, 0.5, 1.0])
    aff[:3, 3] = lo
    arr = np.asarray(img.dataobj, np.float32)
    us = resample_to_grid(img, aff, dims).astype(np.float32)
    m = resample_to_grid(img, aff, dims, mode="nearest", data=(arr > 0).astype(np.float32)) > 0.5
    return us, m.astype(np.uint8), aff


def reg_aladin_bin():
    b = os.environ.get("NIFTYREG_BIN")
    if b and (Path(b) / "reg_aladin").exists():
        return str(Path(b) / "reg_aladin")
    exe = shutil.which("reg_aladin")
    if exe is None:
        sys.exit("reg_aladin not found: install NiftyReg 1.5.69 and set NIFTYREG_BIN or the PATH")
    return exe


def run_baseline(usp, mrp, d, crop_fov=False):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    affp = d / "aff.txt"
    if affp.exists():
        return np.loadtxt(affp)
    us05, m05, aff05 = half_mm_box(nib.load(str(usp)), crop_fov=crop_fov)
    nib.save(nib.Nifti1Image(us05, aff05), str(d / "us05.nii.gz"))
    nib.save(nib.Nifti1Image(m05, aff05), str(d / "us05_mask.nii.gz"))
    cmd = [reg_aladin_bin(), "-ref", d / "us05.nii.gz", "-flo", mrp, "-rmask", d / "us05_mask.nii.gz",
           "-noSym", "-aff", d / "aff_tmp.txt", "-res", d / "res.nii.gz", "-nac", "-ln", "2", "-lp", "10"]
    for k in range(8):
        run = cmd + (["-omp", "2"] if k >= 4 else [])
        with open(d / "reg_aladin.log", "a") as f:
            f.write("\n$ " + " ".join(map(str, run)) + "\n")
            rc = subprocess.run([str(x) for x in run], stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0 and (d / "aff_tmp.txt").exists():
            break
        print(f"    reg_aladin rc={rc}, retry {k + 1}", flush=True)
    else:
        raise RuntimeError("reg_aladin failed 8x")
    (d / "aff_tmp.txt").rename(affp)
    for p in ("res.nii.gz", "us05.nii.gz"):
        (d / p).unlink(missing_ok=True)
    return np.loadtxt(affp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--us", required=True)
    ap.add_argument("--mr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-fov", action="store_true")
    a = ap.parse_args()
    A = run_baseline(a.us, a.mr, a.out, crop_fov=a.crop_fov)
    print(f"[affine_init] -> {Path(a.out) / 'aff.txt'}\n{A}")


if __name__ == "__main__":
    main()
