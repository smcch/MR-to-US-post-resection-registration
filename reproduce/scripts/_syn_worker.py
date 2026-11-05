"""SyN worker for fer_extra_deformables.py, run with anaconda3/envs/mortal (antspyx 0.3.8; no torch there).
Usage: python _syn_worker.py <workdir>   (expects fixed.nii.gz, moving.nii.gz, cone.nii.gz, ci/cj/ck/one.nii.gz;
writes ci_w/cj_w/ck_w/one_w.nii.gz and syn.json with the wall time and the transform names).
ants.registration type_of_transform='SyNOnly' at its defaults (Mattes MI), fixed mask = cone."""
import json
import sys
import time
from pathlib import Path

import ants
import nibabel as nib
import numpy as np

wd = Path(sys.argv[1])


def load(name):
    im = nib.load(str(wd / f"{name}.nii.gz"))
    sp = tuple(float(abs(x)) for x in np.diag(im.affine)[:3])
    return ants.from_numpy(np.asarray(im.dataobj, np.float32), spacing=sp), im.affine


fi, aff = load("fixed")
mi, _ = load("moving")
mk, _ = load("cone")
t0 = time.time()
reg = ants.registration(fixed=fi, moving=mi, type_of_transform="SyNOnly", mask=mk, verbose=False)
wall = time.time() - t0
for name in ("ci", "cj", "ck", "one"):
    im, _ = load(name)
    w = ants.apply_transforms(fixed=fi, moving=im, transformlist=reg["fwdtransforms"], interpolator="linear")
    nib.save(nib.Nifti1Image(w.numpy().astype(np.float32), aff), str(wd / f"{name}_w.nii.gz"))
(wd / "syn.json").write_text(json.dumps(dict(wall_s=wall, transforms=[Path(t).name for t in reg["fwdtransforms"]])))
print(f"syn done {wall:.0f} s")
