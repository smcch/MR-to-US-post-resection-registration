"""NiftyReg free-form stage on the SAME challenge affine as the proposed method (comparator of the 2026-10 version).
reg_f3d -ref <1 mm iUS working grid> -flo MR -rmask FoV -aff <challenge affine> <similarity> -sx 5 -be 0.005
  f3d_mind       --mindssc 1            (the identical MIND-SSC descriptor family)
  f3d_lncc       --lncc 5
  f3d_mind_lncc  --mindssc 1 --lncc 5
Recipe = l2r_f3d_cohorts.f3d_case (unchanged).  Deterministic (CPU).  Scratch in /tmp; per case a json with the
per-landmark errors, folding and SDLogJ over the FoV.
Usage: python fer_f3d.py [--kinds f3d_mind ...] [--cohorts ...]
Output: /mnt/c/ReMIND_work/fer/f3d/<kind>/<cohort>/<case>.json
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa
import l2r_f3d_cohorts as LF                                         # noqa

FLAGS = {"f3d_mind": ["--mindssc", "1", "-sx", "5", "-be", "0.005"],
         "f3d_lncc": ["--lncc", "5", "-sx", "5", "-be", "0.005"],
         "f3d_mind_lncc": ["--mindssc", "1", "--lncc", "5", "-sx", "5", "-be", "0.005"]}
OUT = C.FER / "f3d"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", nargs="*", default=["f3d_mind"])
    ap.add_argument("--cohorts", nargs="*", default=C.COHORTS)
    ap.add_argument("--keep", nargs="*", default=[], help="cohort/case whose displacement field is kept (figures)")
    a = ap.parse_args()
    for kind in a.kinds:
        for coh in a.cohorts:
            for c in C.cases(coh):
                f = OUT / kind / coh / f"{c}.json"
                keep = f"{coh}/{c}" in a.keep
                if f.exists() and not keep:
                    continue
                try:
                    us, mr, _, A, wg = C.inputs(coh, c)
                    fd = Path("/tmp/ferf3d") / kind / coh / c
                    shutil.rmtree(fd, ignore_errors=True)
                    fd.mkdir(parents=True)
                    np.savetxt(fd / "aff_init.txt", A)
                    t0 = time.time()
                    LF.F3D = FLAGS[kind]
                    disp = LF.f3d_case(us, mr, wg, fd / "aff_init.txt", fd)
                    wall = time.time() - t0
                    mri, usl, ids = C.landmarks(coh, c)
                    u = np.asarray(nib.load(str(disp)).dataobj, np.float32)
                    fold, sdlj = C.regularity(u, np.asarray(nib.load(str(wg)).dataobj) > 0)
                    err = np.linalg.norm(usl + C.disp_at(disp, usl) - mri, axis=1)
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(json.dumps(dict(cohort=coh, case=c, kind=kind, ids=[int(i) for i in ids], err=err.tolist(),
                                                 tre=float(err.mean()), fold_pct=fold, sdlogj=sdlj, wall_s=wall)))
                    if keep:
                        kd = OUT / "fields" / kind / coh / c
                        kd.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(disp, kd / "disp_mm.nii.gz")
                    shutil.rmtree(fd, ignore_errors=True)
                    print(f"[f3d] {kind} {coh} {c}: TRE {err.mean():.2f}, fold {fold:.3f} %, SDLogJ {sdlj:.3f}, {wall:.0f} s", flush=True)
                except Exception as e:                                # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    print(f"[f3d] {kind} {coh} {c} FAILED: {e}", flush=True)
    print("F3D DONE", flush=True)


if __name__ == "__main__":
    main()
