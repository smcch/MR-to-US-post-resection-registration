"""Reproducibility of the challenge affine: reg_aladin (ReMIND2Reg baseline configuration) is re-run for EVERY
registration and compared with the stored transform that all the results use (maximum element difference, largest
displacement of the case's iUS landmarks between the two transforms, landmark error of each).  Also records the wall
time of every run.  Scratch in /tmp; nothing stored is modified.
Usage: python fer_verify_affines.py        Output: fer/tables/AFFINE_VERIFY.csv
"""
import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa
from l2r_baseline_cohorts import run_baseline                        # noqa

out = C.FER / "tables/AFFINE_VERIFY.csv"
rows = list(csv.DictReader(open(out))) if out.exists() else []
done = {(r["cohort"], r["case"]) for r in rows}
for coh in C.COHORTS:
    for c in C.cases(coh):
        if (coh, c) in done:
            continue
        us, mr, _, A, _ = C.inputs(coh, c)
        mri, usl, _ = C.landmarks(coh, c)
        d = Path("/tmp/feraffv") / coh / c
        shutil.rmtree(d, ignore_errors=True)
        B = None
        for attempt in range(5):                 # reg_aladin crashes sporadically (rc -11); a new call always succeeds
            shutil.rmtree(d, ignore_errors=True)
            t0 = time.time()
            try:
                B = run_baseline(us, mr, d, crop_fov=coh in ("remind", "external"))
                break
            except RuntimeError:
                print(f"[affv] {coh} {c}: reg_aladin failed 4x, new attempt", flush=True)
        if B is None:
            print(f"[affv] {coh} {c}: GAVE UP", flush=True)
            continue
        wall = time.time() - t0
        retries = (d / "reg_aladin.log").read_text(errors="replace").count("\n$ ") - 1
        rows.append(dict(cohort=coh, case=c, wall_s=round(wall, 1), retries=retries, max_abs_diff=float(np.abs(B - A).max()),
                         max_landmark_shift_mm=float(np.linalg.norm(C.apply(B, usl) - C.apply(A, usl), axis=1).max()),
                         tre_stored=float(np.linalg.norm(C.apply(A, usl) - mri, axis=1).mean()),
                         tre_rerun=float(np.linalg.norm(C.apply(B, usl) - mri, axis=1).mean())))
        r = rows[-1]
        print(f"[affv] {coh} {c}: diff {r['max_abs_diff']:.4f}, landmark shift {r['max_landmark_shift_mm']:.3f} mm, "
              f"TRE {r['tre_stored']:.3f} -> {r['tre_rerun']:.3f}, {wall:.0f} s, retries {retries}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
n = sum(float(r["max_abs_diff"]) > 1e-6 for r in rows)
print(f"AFFINE VERIFY DONE: {len(rows)} registrations, {n} not reproduced exactly")
