"""Shared definitions of the 2026-10 manuscript version (proposed method = NiftyReg challenge affine + finite-element
refinement, "FER").  Cohorts, inputs, landmarks, error helpers.  C: working copy; nothing here writes to G:."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))

W = Path("/mnt/c/ReMIND_work")
FER = W / "fer"
RUN = FER / "run"
# ReMIND / external landmarks: "final" = two-observer consensus set of 2026-11-05 (landmarks_final/, 196 pairs);
# "provisional" = the author's first round (results of that version in fer/superseded_landmarks_2026-11-05/).
LANDMARK_SET = "final"
COHORTS = ["resect_after", "bite_g4", "resect_before", "remind", "external"]
PUBLIC = ["resect_after", "bite_g4", "resect_before"]
LABEL = {"resect_after": "RESECT post-resection", "bite_g4": "BITE post-resection", "resect_before": "RESECT pre-resection",
         "remind": "ReMIND post-resection", "external": "External site"}
REMIND = ["ReMIND-003", "ReMIND-006", "ReMIND-009", "ReMIND-015", "ReMIND-021", "ReMIND-023", "ReMIND-024", "ReMIND-029",
          "ReMIND-036", "ReMIND-038", "ReMIND-042", "ReMIND-043", "ReMIND-059", "ReMIND-062", "ReMIND-064"]   # ReMIND-065 temporarily excluded (user, 2026-10-05)
# cases annotated after the pairing "challenge affine + FE refinement" was first examined (2026-09-24, 8 cases)
REMIND_AFTER_PAIRING = ["ReMIND-036", "ReMIND-038", "ReMIND-042", "ReMIND-043", "ReMIND-059", "ReMIND-062", "ReMIND-064",
                        "ReMIND-065"]
EXTERNAL = ["Subject-1", "Subject-2", "Subject-3", "Subject-4", "Subject-5", "Subject-6", "Subject-7", "Subject-9", "Subject-11", "Subject-12"]


def cases(coh):
    if coh == "remind":
        return list(REMIND)
    if coh == "external":
        return list(EXTERNAL)
    import method_arms as M
    return sorted((c for (k, c) in M.scores() if k == coh), key=lambda s: int("".join(ch for ch in s if ch.isdigit())))


def inputs(coh, c):
    """(iUS, MR, SynthSeg of the MR, NiftyReg challenge affine US world -> MR world, 1 mm working grid of the iUS)."""
    if coh == "remind":
        d = W / "remind"
        return (d / f"data/{c}_USpreimri.nii.gz", d / f"data/{c}_T2.nii.gz", d / f"segs/{c}_T2_synthseg.nii.gz",
                np.loadtxt(d / f"arms/{c}/nr_aff.txt"), d / f"arms/{c}/US_fixed_grid.nii.gz")
    if coh == "external":
        d = W / "brat" / c
        return (d / "us.nii.gz", d / "mr.nii.gz", d / f"{c}_mr_synthseg.nii.gz",
                np.loadtxt(W / f"brat/out/{c}/niftyreg_l2r/aff.txt"),
                W / f"brat/out/{c}/identity/modal_residual_s15/US_fixed_grid.nii.gz")
    import rg_cohorts
    import l2r_f3d_cohorts as LF
    spec = rg_cohorts.COHORTS[coh]
    us, mr = spec["paths"](c)[:2]
    return us, mr, spec["seg"](c), np.loadtxt(W / f"results/l2r_niftyreg_baseline/{coh}/{c}/aff.txt"), LF.WG[coh](c)


def landmarks(coh, c):
    """(mr_pts, us_pts, ids) in world mm.  Third-party tags for RESECT / BITE; the author's Slicer markups for ReMIND
    and the external site (LPS files -> RAS by annotation_qc.read_markups)."""
    if coh in ("remind", "external"):
        from annotation_qc import read_markups
        if LANDMARK_SET == "final":                    # two-observer consensus (scripts/landmarks_final_build.py)
            L = W / "landmarks_final" / coh
            mr, _ = read_markups(L / f"{c}-MR.mrk.json")
            us, _ = read_markups(L / f"{c}-US.mrk.json")
        elif coh == "remind":                          # provisional one-observer set (until 2026-11-05)
            L = W / "annotation_remind/LANDMARKS"
            mr, _ = read_markups(L / f"{c}-MR.mrk.json")
            us, _ = read_markups(L / f"{c}-US-SANTI.mrk.json")
        else:
            L = W / "brat" / c
            mr, _ = read_markups(L / f"{c}-MR.mrk.json")
            us, _ = read_markups(L / f"{c}-US-SANTI.mrk.json")
        ids = sorted(set(mr) & set(us))
        return np.array([mr[i] for i in ids], float), np.array([us[i] for i in ids], float), ids
    import single_config as SC
    tagp, loader = SC.TAG[coh](c)
    mri, us = loader(str(tagp))
    mri, us = np.asarray(mri, float), np.asarray(us, float)
    return mri, us, list(range(1, len(us) + 1))


def apply(A, p):
    return p @ A[:3, :3].T + A[:3, 3]


def disp_at(path_or_img, pts):
    """Trilinear sample of a displacement image (X,Y,Z,3) at world points."""
    u = nib.load(str(path_or_img)) if not hasattr(path_or_img, "affine") else path_or_img
    a = np.asarray(u.dataobj, np.float32)
    inv = np.linalg.inv(u.affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    return np.stack([map_coordinates(a[..., k], [v[:, 0], v[:, 1], v[:, 2]], order=1, mode="nearest") for k in range(3)], -1)


def horn(P, Q):
    """Least-squares rigid (R, t) with Q ~ P R^T + t."""
    cp, cq = P.mean(0), Q.mean(0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cq - cp @ R.T


def regularity(u_vox_xyz3, mask):
    """Folding (% of mask voxels with det J <= 0) and SD of log det J over the mask, for x + u with u in voxels of
    its own grid, array (X,Y,Z,3)."""
    import torch
    import v2_deform as V
    u = torch.from_numpy(np.moveaxis(np.asarray(u_vox_xyz3, np.float32), -1, 0))[None]
    return V.regularity(u, np.asarray(mask, bool))
