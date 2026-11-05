"""Two stored challenge affines (ReMIND-006, ReMIND-036) are not reproduced by re-running reg_aladin with the same
inputs and command (fer_verify_affines.py: 70 of 72 identical; these two differ by 0.68 / 0.45 mm at the landmarks),
whereas fresh runs agree among themselves (three runs each, _chk_aff006.py).  So that every transform of the
manuscript is the one the released code reproduces, the affine of these two cases is replaced by the reproducible one.

  1. the transform in use is kept as remind/arms/<case>/nr_aff_first_run.txt
  2. reg_aladin is run again (persistent copy in fer/affine_rerun/remind/<case>/) and written to nr_aff.txt
  3. everything computed from the old transform for these cases is MOVED (not deleted) to fer/first_affine/:
     fer/run/remind/<case>, fer/f3d/<kind>/remind/<case>.json, fer/steps/remind/<case>.json
  4. fer/tables/AFFINE_FIRST_RUN.csv records the event (read by paper/tools/build_numbers.py)
Afterwards: fer_run.py --cohorts remind --cases ...; fer_f3d.py (three kinds); fer_score.py; fer_steps.py run.
Usage: python fer_fix_affines.py
"""
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa
from l2r_baseline_cohorts import run_baseline                        # noqa

CASES = ("ReMIND-006", "ReMIND-036")
W = C.W
rec = []
for c in CASES:
    arm = W / "remind/arms" / c
    first = arm / "nr_aff_first_run.txt"
    if first.exists():
        print(f"{c}: already replaced")
        A = np.loadtxt(first)
    else:
        A = np.loadtxt(arm / "nr_aff.txt")
    us, mr = W / f"remind/data/{c}_USpreimri.nii.gz", W / f"remind/data/{c}_T2.nii.gz"
    d = C.FER / "affine_rerun/remind" / c
    B = run_baseline(us, mr, d, crop_fov=True)                       # resumable: reads aff.txt when present
    mri, usl, _ = C.landmarks("remind", c)
    shift = float(np.linalg.norm(C.apply(B, usl) - C.apply(A, usl), axis=1).max())
    t_first = float(np.linalg.norm(C.apply(A, usl) - mri, axis=1).mean())
    t_used = float(np.linalg.norm(C.apply(B, usl) - mri, axis=1).mean())
    rec.append(dict(cohort="remind", case=c, max_abs_diff=float(np.abs(B - A).max()), max_landmark_shift_mm=shift,
                    tre_first_run=t_first, tre_used=t_used))
    print(f"{c}: first run vs reproducible: landmark shift {shift:.3f} mm, TRE {t_first:.3f} -> {t_used:.3f}")
    if not first.exists():
        shutil.copy2(arm / "nr_aff.txt", first)
        np.savetxt(arm / "nr_aff.txt", B)
        old = C.FER / "first_affine"
        moves = [(C.RUN / "remind" / c, old / "run/remind" / c), (C.FER / "steps/remind" / f"{c}.json", old / "steps/remind" / f"{c}.json")]
        moves += [(C.FER / "f3d" / k / "remind" / f"{c}.json", old / "f3d" / k / "remind" / f"{c}.json")
                  for k in ("f3d_mind", "f3d_lncc", "f3d_mind_lncc")]
        for src, dst in moves:
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                print(f"   moved {src.relative_to(C.FER)} -> {dst.relative_to(C.FER)}")
    assert np.allclose(np.loadtxt(arm / "nr_aff.txt"), B)
with open(C.FER / "tables/AFFINE_FIRST_RUN.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rec[0])); w.writeheader(); w.writerows(rec)
print("AFFINE FIX DONE")
