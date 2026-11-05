"""rg_cohorts.py -- dataset adapters for the gated-rigid extension.

**Nothing frozen is modified.**  `rigid_search.py`, `rigid_search_fft.py`,
`rs_prestage.py`, `reg_modal.py`, `biomech_basis.py` are imported read-only and
their `case_paths` / `case_landmarks` / `FROZEN` symbols are *rebound at
runtime* so that exactly the same code objects serve cohorts they were never
written for.  The cost, the mask convention, the pose convention, the candidate
rule and the whole stage-2 refinement are therefore literally identical across
cohorts -- which is the only way a gate calibrated on one can be tested on
another.

Cohorts
  resect_after   RESECT, preop FLAIR -> POST-resection US, 20 cases  (the
                 cohort the concurrent workflow already ran; included so the
                 adapter can be verified to reproduce its numbers exactly)
  bite_g4        BITE group 4, preop T1+Gd -> POST-resection US, 13 patients
  remind         ReMIND, preop T2 -> post-dura US (landmark subset, 14)
  remind113      ReMIND, every case with the required inputs (landmark-free)
  resect_before  RESECT, preop FLAIR -> PRE-resection US, 22 cases  (context)
  bite_g2        BITE group 2, preop T1+Gd -> PRE-resection US, 14  (context)

The one non-obvious choice, stated plainly: for **bite_g4 the moving image is
the full-head MR** (`data/bite/synthseg_in_g4/<k>_mrfull.nii.gz`), not the
shipped `group4/<k>/mr.nii.gz`.  The shipped file is cropped to the US-cone
bounding box, so *any* rigid excursion would sample zeros outside it and the
search would be scoring padding rather than anatomy.  The two images are the
same object -- BITE_RESULTS.md 1b verifies the rebuild reproduces the shipped
crop at NCC = 1.000000, max |d| <= 0.026 -- and the modal basis the frozen BITE
arm used was *already* built on the full-head rebuild.  The A = I reference arm
is re-run with the same full-head moving so every paired comparison has the
identical moving image on both sides.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

WD = Path("/mnt/c/ReMIND_work")
TAGS = Path("/mnt/g/ECOGRAFIA/ReMIND_downloads/RESECT/NIFTI")
# ReMIND cold data archived 2026-08-19 (see ARCHIVE_MAP.md at repo root)
ARCH = Path("/mnt/c/ReMIND_work/archive")
OUT = WD / "results/rigid_gate"

RESECT_AFTER = [1, 2, 3, 4, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 21, 23,
                24, 25, 27]
RESECT_BEFORE = [1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 21,
                 23, 24, 25, 26, 27]
BITE_G4 = [f"{k:02d}" for k in range(1, 14)]
BITE_G2 = [f"{k:02d}" for k in range(1, 15)]
REMIND_LM = ["ReMIND-003", "ReMIND-023", "ReMIND-031", "ReMIND-032",
             "ReMIND-034", "ReMIND-045", "ReMIND-056", "ReMIND-077",
             "ReMIND-079", "ReMIND-087", "ReMIND-091", "ReMIND-096",
             "ReMIND-107", "ReMIND-109"]
MRC = ["mr_x_RAS_mm", "mr_y_RAS_mm", "mr_z_RAS_mm"]
USC = ["us_x_RAS_mm", "us_y_RAS_mm", "us_z_RAS_mm"]


# ------------------------------------------------------------------ landmarks

def _resect_tag(case, which):
    from resect_tags import load_tag_pairs
    tag = TAGS / case / "Landmarks" / f"{case}-MRI-{which}US.tag"
    if not tag.exists():
        return None, None
    mri, us = load_tag_pairs(str(tag))
    return us, mri


def _resect_before_tag(case):
    from resect_tags import parse_tag
    tag = WD / "data/resect_before" / case / f"{case}-MRI-beforeUS.tag"
    if not tag.exists():
        return None, None
    mri, us = parse_tag(str(tag))
    return np.asarray(us, float), np.asarray(mri, float)


def _bite_tag(group, case):
    from bite_tags import load_tag_pairs
    d = WD / "data/bite" / f"group{group}" / case
    tag = d / ("Tags.tag" if group == 4 else f"{case}_all.tag")
    if not tag.exists():
        return None, None
    mri, us = load_tag_pairs(str(tag))
    return np.asarray(us, float), np.asarray(mri, float)


def _remind_lm(case):
    p = WD / "release/landmarks/ReMIND-BS" / case / f"landmarks_{case}.csv"
    if not p.exists():
        return None, None
    t = pd.read_csv(p, comment="#")
    return t[USC].to_numpy(float), t[MRC].to_numpy(float)


# ------------------------------------------------------------------- cohorts

def _remind113_cases():
    """Every ReMIND case with the three inputs the frozen arm needs."""
    us = sorted((ARCH / "data/us").glob("ReMIND-*_USpreimri.nii.gz"))
    out = []
    for p in us:
        c = p.name.split("_")[0]
        if (ARCH / f"data/synthseg_in/preop/{c}_T2.nii.gz").exists() and \
           (ARCH / f"data/synthseg_out/preop/{c}_T2_synthseg.nii.gz").exists():
            out.append(c)
    return out


COHORTS = {
    "resect_after": dict(
        cases=lambda: [f"Case{c}" for c in RESECT_AFTER],
        paths=lambda c: (WD / f"data/resect/{c}/{c}-US-after.nii.gz",
                         WD / f"data/resect/{c}/{c}-FLAIR.nii.gz",
                         (WD / f"data/resect/{c}/{c}-US-after-cavity.nii.gz"
                          if (WD / f"data/resect/{c}/{c}-US-after-cavity.nii.gz"
                              ).exists() else None), [1]),
        landmarks=lambda c: _resect_tag(c, "after"),
        seg=lambda c: WD / f"data/resect/synthseg_out/{c}-FLAIR_synthseg.nii.gz",
        tumor=lambda c: None,
        basis=lambda c: WD / f"data/resect/basis/{c}_basis.npz",
        frozen_arm=lambda c: WD / f"results/resect_tre/{c}/modal_residual_s15",
        lam_eig="3e-4", exclude_labels="1", exclude_dilate="3",
        regime="post", floor=0.48, label="RESECT after-US"),

    # RESECT BEFORE-resection: the one regime with a strong published
    # comparator we can actually run against (MCPO-deform 1.798 mm on the
    # identical landmark set, identical initial error 5.374).  No cavity, so no
    # exclusion mask; tags ship next to the images in data/resect_before/.
    "resect_before": dict(
        cases=lambda: sorted(
            (p.name for p in (WD / "data/resect_before").iterdir()
             if p.is_dir() and p.name.startswith("Case")
             and (p / f"{p.name}-US-before.nii.gz").exists()
             and (p / f"{p.name}-MRI-beforeUS.tag").exists()),
            key=lambda s: int(s.replace("Case", ""))),
        paths=lambda c: (WD / f"data/resect_before/{c}/{c}-US-before.nii.gz",
                         WD / f"data/resect_before/{c}/{c}-FLAIR.nii.gz",
                         None, [1]),
        landmarks=lambda c: _resect_before_tag(c),
        seg=lambda c: WD / f"data/resect_before/synthseg_out/{c}-FLAIR_synthseg.nii.gz",
        tumor=lambda c: None,
        basis=lambda c: WD / f"data/resect_before/basis/{c}_basis.npz",
        frozen_arm=lambda c: WD / f"results/resect_before/{c}/modal_residual_s15",
        lam_eig="3e-4", exclude_labels="1", exclude_dilate="3",
        regime="pre", floor=0.48, label="RESECT before-US"),

    "bite_g4": dict(
        cases=lambda: list(BITE_G4),
        paths=lambda c: (WD / f"data/bite/group4/{c}/us.nii.gz",
                         WD / f"data/bite/synthseg_in_g4/{c}_mrfull.nii.gz",
                         None, [1]),
        landmarks=lambda c: _bite_tag(4, c),
        seg=lambda c: WD / f"data/bite/synthseg_out_g4/{c}_mrfull_synthseg.nii.gz",
        tumor=lambda c: None,
        basis=lambda c: WD / f"data/bite/basis_g4/{c}_basis.npz",
        frozen_arm=lambda c: WD / f"results/bite/g4/{c}/modal_residual_s15",
        lam_eig="3e-4", exclude_labels="1", exclude_dilate="3",
        regime="post", floor=1.45, label="BITE group 4"),

    "remind": dict(
        cases=lambda: list(REMIND_LM),
        paths=lambda c: (ARCH / f"data/us/{c}_USpreimri.nii.gz",
                         ARCH / f"data/synthseg_in/preop/{c}_T2.nii.gz",
                         (ARCH / f"data/segmentations/US/{c}-post-us-segmentation.nii.gz"
                          if (ARCH / f"data/segmentations/US/{c}-post-us-segmentation.nii.gz"
                              ).exists() else None), [1, 2]),
        landmarks=_remind_lm,
        seg=lambda c: ARCH / f"data/synthseg_out/preop/{c}_T2_synthseg.nii.gz",
        tumor=lambda c: ARCH / f"data/segmentations/MRI/{c}-pre-mri-segmentation.nii.gz",
        basis=lambda c: ARCH / f"data/basis/{c}_basis.npz",
        frozen_arm=lambda c: None,
        lam_eig="3e-3", exclude_labels="1,2", exclude_dilate="3",
        regime="post", floor=None, label="ReMIND (landmark subset)"),

    "resect_before": dict(
        cases=lambda: [f"Case{c}" for c in RESECT_BEFORE],
        paths=lambda c: (WD / f"data/resect_before/{c}/{c}-US-before.nii.gz",
                         WD / f"data/resect_before/{c}/{c}-FLAIR.nii.gz",
                         None, [1]),
        landmarks=_resect_before_tag,
        seg=lambda c: WD / f"data/resect_before/synthseg_out/{c}-FLAIR_synthseg.nii.gz",
        tumor=lambda c: None,
        basis=lambda c: WD / f"data/resect_before/basis/{c}_basis.npz",
        frozen_arm=lambda c: WD / f"results/resect_before/{c}/modal_residual_s15",
        lam_eig="3e-4", exclude_labels="1", exclude_dilate="3",
        regime="pre", floor=0.48, label="RESECT before-US (context)"),

    "bite_g2": dict(
        cases=lambda: list(BITE_G2),
        paths=lambda c: (WD / f"data/bite/group2/{c}/us.nii.gz",
                         WD / f"data/bite/group2/{c}/mr.nii.gz",
                         None, [1]),
        landmarks=lambda c: _bite_tag(2, c),
        seg=lambda c: WD / f"data/bite/synthseg_out_g2/{c}_mrfull_synthseg.nii.gz",
        tumor=lambda c: None,
        basis=lambda c: WD / f"data/bite/basis_g2/{c}_basis.npz",
        frozen_arm=lambda c: WD / f"results/bite/g2/{c}/modal_residual_s15",
        lam_eig="3e-4", exclude_labels="1", exclude_dilate="3",
        regime="pre", floor=1.45, label="BITE group 2 (context)"),
}
COHORTS["remind113"] = dict(COHORTS["remind"],
                            cases=_remind113_cases,
                            label="ReMIND (all available, landmark-free)")


# -------------------------------------------------------------------- install

def cohort_cases(name):
    return COHORTS[name]["cases"]()


def install(name):
    """Rebind `case_paths` / `case_landmarks` (and rs_prestage's FROZEN) in the
    frozen modules so they serve `name`.  Returns the cohort spec."""
    spec = COHORTS[name]

    def case_paths(dataset, case):
        return spec["paths"](case)

    def case_landmarks(dataset, case):
        return spec["landmarks"](case)

    import rigid_search
    rigid_search.case_paths = case_paths
    rigid_search.case_landmarks = case_landmarks
    for mod in ("rigid_search_fft", "rs_prestage"):
        if mod in sys.modules:
            m = sys.modules[mod]
            if hasattr(m, "case_paths"):
                m.case_paths = case_paths
            if hasattr(m, "case_landmarks"):
                m.case_landmarks = case_landmarks
    if "rs_prestage" in sys.modules:
        sys.modules["rs_prestage"].FROZEN[name] = dict(
            lam_eig=spec["lam_eig"], exclude_labels=spec["exclude_labels"],
            exclude_dilate=spec["exclude_dilate"],
            seg=spec["seg"], tumor=spec["tumor"])
    return spec
