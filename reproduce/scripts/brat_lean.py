"""External-site case, manuscript arms only (no pose search / paper / v2): 1 mm working grid of the iUS, NiftyReg
challenge baseline (+ f3d MIND-SSC) and MCPO original, in the layout brat_eval.py uses, so that the fer_* scripts
find them (fer_common.inputs: brat/out/<case>/niftyreg_l2r/aff.txt and identity/modal_residual_s15/US_fixed_grid.nii.gz;
fer_mcpo: brat/out/<case>/mcpo/disp.nii.gz).  Used when the old pose-search chain is not wanted or stalls
(Subject-7: the polish stage filled the 24 GB GPU).   Usage: python brat_lean.py Subject-7"""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

case = sys.argv[1]
sys.argv = [sys.argv[0], "run", case]                                 # brat_eval reads the case from argv at import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import brat_eval as BE                                               # noqa  (registers the brat_set cohort for this case)
import annotation_eval as E                                          # noqa
import rg_cohorts                                                    # noqa
from reg_core import load_to_grid                                    # noqa

out = BE.OUT / case
out.mkdir(parents=True, exist_ok=True)
wg = out / "identity/modal_residual_s15/US_fixed_grid.nii.gz"
if not wg.exists():
    f_arr, _, grid_aff, dims = load_to_grid(BE.D / "us.nii.gz", BE.D / "us.nii.gz", 1.0)
    wg.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(f_arr.astype(np.float32), grid_aff), str(wg))
    print(f"[lean] working grid {dims.tolist()} written", flush=True)
spec = rg_cohorts.install("brat_set")
E.niftyreg_arms(case, spec, wg, out)
print("[lean] NiftyReg arms done", flush=True)
BE.mcpo_original(out)
print("[lean] MCPO original done", flush=True)
(out / "LEAN_DONE").write_text("manuscript arms only")
