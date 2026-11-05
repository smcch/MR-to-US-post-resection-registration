"""The shipped method, one configuration, one source.

Every consumer of "our" per-case result (statistics, tables, figures, the
visual export, the comparator scripts) reads it from here, so the manuscript
cannot mix configurations again.  The runs are produced by single_config.py:

  pose      A (x_us + u_pose(x_us))   A = frozen search + guarded polish
            (results/rigid_gate/search_polish/<cohort>/<case>/aff.txt)
  identity  x_us + u_identity(x_us)   the deformation-only arm and the
                                      abstention fallback
  final     identity if the search rotation >= 9.5 deg (abstention) else pose

CFG names the configuration selected on the BITE development set
(single_config.py: m20 = 20 modes, m30r = 30 modes + smoothed residual;
lambda_eig 3e-3 and no segmentation mask in both).
"""
import csv
from pathlib import Path

W = Path("/mnt/c/ReMIND_work")
R = W / "results"
CFG = "m30r"
ROOT = R / "single_config" / CFG
SAT = 9.5
COHORTS = ["resect_after", "bite_g4", "resect_before"]


def pose(coh, c):
    return R / f"rigid_gate/search_polish/{coh}/{c}/aff.txt"


def u_pose(coh, c):
    return ROOT / coh / c / "pose/modal/disp_mm.nii.gz"


def u_identity(coh, c):
    return ROOT / coh / c / "identity/modal/disp_mm.nii.gz"


def grid(coh, c):
    return ROOT / coh / c / "identity/modal/US_fixed_grid.nii.gz"


def _f(v):
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except ValueError:
        return v


def scores():
    """{(cohort, case): row} from single_config.py score (SCORES.csv)."""
    return {(r["cohort"], r["case"]): {k: _f(v) for k, v in r.items()}
            for r in csv.DictReader(open(ROOT / "SCORES.csv"))}


def cohort(coh):
    """{case: row} for one cohort."""
    return {c: r for (k, c), r in scores().items() if k == coh}
