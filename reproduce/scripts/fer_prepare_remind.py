"""Prepare NEW ReMIND landmark cases for the FE-refinement pipeline without going through the annotation-study arms:
the challenge affine (reg_aladin, ReMIND2Reg baseline configuration, iUS box cropped to its field of view; the same
call as annotation_eval.niftyreg_arms) and the 1 mm working grid of the iUS.
Inputs must be staged first (scripts/stage_remind_new.sh): remind/data/<case>_USpreimri.nii.gz, <case>_T2.nii.gz,
remind/segs/<case>_T2_synthseg.nii.gz.  Then add the case to REMIND in fer_common.py and fer_run.py.
Usage: python fer_prepare_remind.py ReMIND-066 ReMIND-068 ...
Output: remind/arms/<case>/{nr_aff.txt,US_fixed_grid.nii.gz}
"""
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l2r_baseline_cohorts import run_baseline                        # noqa
from reg_core import load_to_grid, normalize                         # noqa

D = Path("/mnt/c/ReMIND_work/remind")
for c in sys.argv[1:]:
    us, mr = D / f"data/{c}_USpreimri.nii.gz", D / f"data/{c}_T2.nii.gz"
    out = D / "arms" / c
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "nr_aff.txt").exists():
        tmp = Path("/tmp/ferprep") / c
        shutil.rmtree(tmp, ignore_errors=True)
        A = run_baseline(us, mr, tmp, crop_fov=True)
        np.savetxt(out / "nr_aff.txt", A)
        shutil.rmtree(tmp, ignore_errors=True)
    if not (out / "US_fixed_grid.nii.gz").exists():
        f_arr, _, grid_aff, _ = load_to_grid(us, mr, 1.0)
        f_n = normalize(f_arr, f_arr > 0)                              # as reg_modal writes it (intensities in [0, 1])
        nib.save(nib.Nifti1Image(f_n, grid_aff), str(out / "US_fixed_grid.nii.gz"))
    print(f"[prepare] {c}: affine and working grid ready", flush=True)
