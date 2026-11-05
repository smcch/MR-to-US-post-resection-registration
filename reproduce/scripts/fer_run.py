"""Proposed method of the 2026-10 manuscript version, ONE configuration for every case:

    NiftyReg challenge affine (reg_aladin rigid + affine; the ReMIND2Reg baseline, stored transforms)
    -> patient-specific finite-element model built in the posed frame (two compartments, h 5 mm, 30 modes)
    -> frozen deformable stage v2_method.ACCEPTED (30 modes fitted on MIND-SSC + local FE response to
       symmetric block correspondences, bound 2 mm), through the frozen v2_remind_eval.deform

No tumour compartment, no cavity / tumour exclusion mask: nothing but the preoperative MR, its SynthSeg
segmentation and the iUS.  Landmark-blind (no landmark file is read here; scoring is fer_score.py).
The frozen modules are only imported; sub-stage timings are collected by wrapping their functions.

Usage: python fer_run.py [--cohorts resect_after bite_g4 resect_before remind external] [--cases ...]
Output: /mnt/c/ReMIND_work/fer/run/<cohort>/<case>/{synthseg_rig.nii.gz,basis30.npz,fer/disp_mm.nii.gz,fer/meta.json,
        timing.json}
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rg_cohorts                                                    # noqa
import method_arms as M                                              # noqa
import single_config as SC                                           # noqa
import affine_cascade as AC                                          # noqa
import v2_deform as V                                                # noqa
import v2_method as FM                                               # noqa
import v2_remind_eval as VE                                          # noqa

W = Path("/mnt/c/ReMIND_work")
OUT = W / "fer/run"
COHORTS = ["resect_after", "bite_g4", "resect_before", "remind", "external"]
REMIND = ["ReMIND-003", "ReMIND-006", "ReMIND-009", "ReMIND-015", "ReMIND-021", "ReMIND-023", "ReMIND-024", "ReMIND-029",
          "ReMIND-036", "ReMIND-038", "ReMIND-042", "ReMIND-043", "ReMIND-059", "ReMIND-062", "ReMIND-064"]   # ReMIND-065 temporarily excluded (user, 2026-10-05)
EXTERNAL = ["Subject-1", "Subject-2", "Subject-3", "Subject-4", "Subject-5", "Subject-6", "Subject-7", "Subject-9", "Subject-11", "Subject-12"]


def cases(coh):
    if coh == "remind":
        return list(REMIND)
    if coh == "external":
        return list(EXTERNAL)
    return sorted(c for (k, c) in M.scores() if k == coh)


def inputs(coh, c):
    """(iUS, MR, SynthSeg of the MR, NiftyReg challenge affine US world -> MR world)."""
    if coh == "remind":
        d = W / "remind"
        return (d / f"data/{c}_USpreimri.nii.gz", d / f"data/{c}_T2.nii.gz", d / f"segs/{c}_T2_synthseg.nii.gz",
                np.loadtxt(d / f"arms/{c}/nr_aff.txt"))
    if coh == "external":
        d = W / "brat" / c
        return (d / "us.nii.gz", d / "mr.nii.gz", d / f"{c}_mr_synthseg.nii.gz",
                np.loadtxt(W / f"brat/out/{c}/niftyreg_l2r/aff.txt"))
    spec = rg_cohorts.COHORTS[coh]
    us, mr = spec["paths"](c)[:2]
    return us, mr, spec["seg"](c), np.loadtxt(W / f"results/l2r_niftyreg_baseline/{coh}/{c}/aff.txt")


TIMES = {}


def timed(name, fn):
    def wrap(*a, **k):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = fn(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        TIMES[name] = TIMES.get(name, 0.0) + time.time() - t0
        return out
    return wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="*", default=COHORTS)
    ap.add_argument("--cases", nargs="*")
    a = ap.parse_args()
    VE.relabel = AC.relabel                                           # the pose is affine: sform-exact relabelling
    V.register_modal_from = timed("modal_fit_s", V.register_modal_from)
    V.residual_correspondences = timed("correspondences_s", V.residual_correspondences)
    V.fem_residual = timed("fe_response_s", V.fem_residual)
    for coh in a.cohorts:
        for c in cases(coh):
            if a.cases and c not in a.cases:
                continue
            d = OUT / coh / c
            if (d / "fer/disp_mm.nii.gz").exists() and (d / "timing.json").exists():
                continue
            try:
                d.mkdir(parents=True, exist_ok=True)
                us, mr, seg, A = inputs(coh, c)
                for p in (us, mr, seg):
                    if not Path(p).exists():
                        raise FileNotFoundError(p)
                t_all = time.time()
                seg_r, basis = d / "synthseg_rig.nii.gz", d / "basis30.npz"
                AC.relabel(seg, seg_r, A)
                t0 = time.time()
                if not basis.exists():
                    with open(d / "run.log", "a") as fh:
                        rc = subprocess.run([SC.PY, str(SC.S / "biomech_basis.py"), "--synthseg", str(seg_r), "--out", str(basis),
                                             "--h", "5.0", "--n-modes", "30"], stdout=fh, stderr=subprocess.STDOUT).returncode
                    if rc != 0 or not basis.exists():
                        raise RuntimeError("basis failed")
                t_basis = time.time() - t0
                TIMES.clear()
                VE.paths = lambda case, us=us, mr=mr: (us, mr)
                VE.OUT = OUT / coh
                t0 = time.time()
                VE.deform(c, FM.ACCEPTED, "fer", A, basis)
                t_def = time.time() - t0
                meta = json.loads((d / "fer/meta.json").read_text())
                sv = np.linalg.svd(A[:3, :3], compute_uv=False)
                (d / "timing.json").write_text(json.dumps(dict(
                    cohort=coh, case=c, basis_s=t_basis, deform_total_s=t_def, **{k: float(v) for k, v in TIMES.items()},
                    total_s=time.time() - t_all, peak_mb=meta["peak_mb"], fold_pct=meta["fold_pct"], sdlogj=meta["sdlogj"],
                    affine_singular_values=sv.tolist(), version=FM.VERSION)))
                print(f"[fer] {coh} {c}: basis {t_basis:.0f} s, deformable {t_def:.0f} s (modal {TIMES.get('modal_fit_s', 0):.0f}, "
                      f"corr {TIMES.get('correspondences_s', 0):.0f}, FE {TIMES.get('fe_response_s', 0):.0f}), "
                      f"fold {meta['fold_pct']:.3f} %, SDLogJ {meta['sdlogj']:.3f}", flush=True)
            except Exception as e:                                    # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[fer] {coh} {c} FAILED: {e}", flush=True)
    print("FER RUN DONE", flush=True)


if __name__ == "__main__":
    main()
