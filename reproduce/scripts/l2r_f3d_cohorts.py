"""NiftyReg f3d MIND-SSC initialised with the Learn2Reg NiftyReg baseline
affine, on the paper's three landmark cohorts: the standard NiftyReg
deformable pipeline (reg_aladin -> reg_f3d) with the challenge baseline as
its first stage.

Per case:
    reg_f3d -ref WG -flo MR -rmask FOV -aff <baseline aff.txt>
            --mindssc 1 -sx 5 -be 0.005
WG  = the 1 mm working grid of the iUS used by every arm of the paper
      (<cohort results>/<case>/baseline/US_fixed_grid.nii.gz);
FOV = the iUS field of view on that grid (iUS > 0, nearest-neighbour), the
      same mask convention as the challenge baseline.
The control-point spacing (5 mm) and bending-energy weight (5e-3) are fixed
a priori, not selected on these cohorts.  Deformation field via
reg_transform -def, converted to a displacement (niftyreg_prep.def2disp);
prediction for an iUS landmark: x + u(x).

Resumable.  Outputs: results/l2r_niftyreg_baseline/<cohort>/<case>/f3d_mind/
and results/l2r_niftyreg_baseline/PER_CASE_F3D.csv.
Usage: python l2r_f3d_cohorts.py
"""
import csv
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import niftyreg_prep                                                 # noqa
import rg_cohorts                                                    # noqa
from reg_core import resample_to_grid                                # noqa
from l2r_baseline_cohorts import COH, NR, OUT, R, disp_at            # noqa

WG = {"resect_before": lambda c: R / f"resect_before/{c}/baseline/US_fixed_grid.nii.gz",
      "resect_after": lambda c: R / f"resect_tre/{c}/baseline/US_fixed_grid.nii.gz",
      "bite_g4": lambda c: R / f"bite/g4/{c}/baseline/US_fixed_grid.nii.gz"}
F3D = ["--mindssc", "1", "-sx", "5", "-be", "0.005"]


def run(cmd, log):
    for k in range(4):
        with open(log, "a") as f:
            f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
            rc = subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            return
        print(f"    {Path(str(cmd[0])).name} rc={rc}, retry {k + 1}", flush=True)
    raise RuntimeError(f"{Path(str(cmd[0])).name} failed 4x")


def fov_on_grid(usp, wgp, out):
    us = nib.load(str(usp))
    g = nib.load(str(wgp))
    m = resample_to_grid(us, g.affine, g.shape[:3], mode="nearest",
                         data=(np.asarray(us.dataobj) > 0).astype(np.float32)) > 0.5
    nib.save(nib.Nifti1Image(m.astype(np.uint8), g.affine), str(out))


def f3d_case(usp, mrp, wgp, affp, fd):
    disp = fd / "disp_mm.nii.gz"
    if disp.exists():
        return disp
    fd.mkdir(parents=True, exist_ok=True)
    log = fd / "reg_f3d.log"
    fov_on_grid(usp, wgp, fd / "fov_mask.nii.gz")
    run([NR / "reg_f3d", "-ref", wgp, "-flo", mrp, "-rmask", fd / "fov_mask.nii.gz", "-aff", affp,
         *F3D, "-cpp", fd / "cpp.nii.gz", "-res", fd / "res.nii.gz", "-voff"], log)
    run([NR / "reg_transform", "-ref", wgp, "-def", fd / "cpp.nii.gz", fd / "def.nii.gz"], log)
    niftyreg_prep.def2disp(fd / "def.nii.gz", fd)
    for p in ("def.nii.gz", "res.nii.gz"):
        (fd / p).unlink(missing_ok=True)
    return disp


def main():
    rows = []
    for coh, P in COH.items():
        spec = rg_cohorts.install(coh)
        for r in csv.DictReader(open(R / f"rigid_gate/search_tight/{coh}/PER_CASE.csv")):
            c = r["case"]
            try:
                affp = OUT / coh / c / "aff.txt"
                tagp, loader = P["tag"](c)
                if not affp.exists() or not tagp.exists():
                    print(f"[f3d] {coh} {c}: missing baseline affine or tags, skipped", flush=True)
                    continue
                usp, mrp = spec["paths"](c)[:2]
                wgp = WG[coh](c)
                disp = f3d_case(usp, mrp, wgp, affp, OUT / coh / c / "f3d_mind")
                mri, us = loader(str(tagp))
                mri, us = np.asarray(mri, float), np.asarray(us, float)
                e = np.linalg.norm(us + disp_at(disp, us) - mri, axis=1)
                rows.append(dict(cohort=coh, case=c, l2r_f3d=round(float(e.mean()), 4),
                                 l2r_f3d_median=round(float(np.median(e)), 4),
                                 l2r_f3d_tre30=round(float(np.percentile(e, 70)), 4)))
                print(f"[f3d] {coh} {c}: L2R + f3d MIND-SSC {e.mean():.2f} mm", flush=True)
            except Exception as ex:                                  # noqa: BLE001
                print(f"[f3d] {coh} {c} FAILED: {ex}", flush=True)
    with open(OUT / "PER_CASE_F3D.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for coh in COH:
        v = np.array([x["l2r_f3d"] for x in rows if x["cohort"] == coh])
        if len(v):
            print(f"{coh}: n={len(v)} mean {v.mean():.2f} +- {v.std(ddof=1):.2f}")
    print("F3D DONE")


if __name__ == "__main__":
    main()
