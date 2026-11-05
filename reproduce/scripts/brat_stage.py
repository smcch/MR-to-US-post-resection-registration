"""Stage one external-site pair (local use only, never redistributed).  Source since 2026-11-05: the author's copy on C:
(C:/ReMIND_biomech/annotation_remind/SET-Bratislava; G: is failing).
mr.nii.gz = the preoperative MR verbatim: <case>-T2-pre.nii.gz or <case>-MR-pre.nii.gz (3D FLAIR), EXCEPT Subject-3,
whose landmarks were placed on the 3D T1 (author, 2026-11-05): cs_T1W_3D_TFE_s301.nii.gz with the sweep exported in
that image's frame (Scan6_in_cs_T1W_3D_TFE.nii.gz).  The FLAIR of Subject-3 lies ~16 mm / 4 deg from the T1 frame.
us.nii.gz = the post-resection sweep cropped to its non-zero bounding box (+2 voxels), world coordinates preserved
(pure crop of zero padding, no resampling).  The author's first-round landmark files are copied for the record; the
scored landmarks are the final two-observer set (fer_common.landmarks, landmarks_final/).
Usage: python brat_stage.py Subject-2 [Subject-3 ...]"""
import shutil
import sys
import numpy as np, nibabel as nib
from pathlib import Path
G = Path("/mnt/c/ReMIND_biomech/annotation_remind")
SPECIAL = {"Subject-3": ("cs_T1W_3D_TFE_s301.nii.gz", "Scan6_in_cs_T1W_3D_TFE.nii.gz")}
for case in sys.argv[1:]:
    D = Path("/mnt/c/ReMIND_work/brat") / case
    D.mkdir(parents=True, exist_ok=True)
    S = G / f"SET-Bratislava/{case}"
    if case in SPECIAL:
        src, usrc = S / SPECIAL[case][0], S / SPECIAL[case][1]
    else:
        src = next(p for p in (S / f"{case}-T2-pre.nii.gz", S / f"{case}-MR-pre.nii.gz") if p.exists())
        usrc = S / f"{case}-US-post.nii.gz"
    shutil.copy2(src, D / "mr.nii.gz")
    for f in (f"{case}-MR.mrk.json", f"{case}-US-SANTI.mrk.json"):
        shutil.copy2(G / "LANDMARKS-BRATISLAVA" / f, D / f)
    u = nib.load(str(usrc))
    a = np.asarray(u.dataobj)
    nz = np.argwhere(a > 0)
    lo = np.maximum(nz.min(0) - 2, 0); hi = np.minimum(nz.max(0) + 3, a.shape)
    c = a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    aff = u.affine.copy(); aff[:3, 3] = nib.affines.apply_affine(u.affine, lo)
    o = nib.Nifti1Image(c, aff, u.header); o.set_sform(aff, 1); o.set_qform(aff, 1)
    nib.save(o, str(D / "us.nii.gz"))
    m = nib.load(str(D / "mr.nii.gz"))
    print(f"{case}: MR {src.name} {m.shape} zooms {tuple(round(float(z), 2) for z in m.header.get_zooms()[:3])}; "
          f"US {usrc.name} {a.shape} -> {c.shape} zooms {tuple(round(float(z), 2) for z in u.header.get_zooms()[:3])}, "
          f"nonzero kept {int((c > 0).sum())} of {int((a > 0).sum())}")
