"""External site: one table over all staged cases.  Manuscript arms from fer/tables/PER_CASE.csv (initial, affine,
affine + f3d, affine + FE refinement, MCPO pose, MCPO pose + FE refinement, MCPO original, best-fit rigid) and the
older arms of brat_eval.py from brat/out/<case>/TRE_landmarks.csv (our pose, paper method, v2, affine + paper stage,
affine + v2 stage).  Output: brat/out/REPORT_ALL.md (printed).  Usage: python brat_summary.py"""
import csv
from pathlib import Path

import numpy as np

W = Path("/mnt/c/ReMIND_work")
rows = [r for r in csv.DictReader(open(W / "fer/tables/PER_CASE.csv")) if r["cohort"] == "external"]
ARMS = [("initial", "navigation (initial)"), ("affine", "NiftyReg affine (baseline)"), ("f3d_mind", "affine + f3d MIND-SSC"),
        ("fer", "affine + FE refinement (proposed)"), ("mcpo_pose", "MCPO pose"), ("mcpo_pose_fer", "MCPO pose + FE refinement"),
        ("mcpo", "MCPO original"), ("bestfit", "best-fit rigid (oracle)")]
OLD = [("pose", "our pose only"), ("paper", "paper method"), ("v2", "v2"), ("nr_paper", "affine + paper stage"), ("nr_v2", "affine + v2 stage (earlier run)")]
cases = [r["case"] for r in rows]
L = [f"# External site: {len(cases)} cases ({', '.join(cases)}), observer SANTI, preoperative MR (FLAIR) to post-resection iUS", "",
     "Mean TRE over the landmarks of each case (mm).", "",
     "| arm | " + " | ".join(cases) + " | mean |", "|---|" + "---|" * (len(cases) + 1)]


def f(v):
    return "—" if v in ("", None) or (isinstance(v, float) and np.isnan(v)) else f"{float(v):.2f}"


for k, name in ARMS:
    vals = [float(r[k]) if r.get(k) not in ("", None) else np.nan for r in rows]
    L.append(f"| {name} | " + " | ".join(f(v) for v in vals) + f" | {np.nanmean(vals):.2f} |")
old = {}
for c in cases:
    p = W / f"brat/out/{c}/TRE_landmarks.csv"
    if p.exists():
        rr = list(csv.DictReader(open(p)))
        old[c] = {k: float(np.mean([float(x[k]) for x in rr])) for k in rr[0] if k != "lm"}
L += ["", "Older arms (brat_eval.py, not in the manuscript):", "", "| arm | " + " | ".join(cases) + " | mean |", "|---|" + "---|" * (len(cases) + 1)]
for k, name in OLD:
    vals = [old.get(c, {}).get(k, np.nan) for c in cases]
    L.append(f"| {name} | " + " | ".join(f(v) for v in vals) + f" | {np.nanmean(vals):.2f} |")
L += ["", "Regularity (SDLogJ at 1 mm): " + "; ".join(f"{r['case']}: FE refinement {float(r['fer_sdlogj']):.2f}, MCPO {float(r['mcpo_sdlogj']):.2f}" for r in rows if r.get("mcpo_sdlogj") not in ("", None)),
      "Affine rotation (deg): " + ", ".join(f"{r['case']} {float(r['aff_rot_deg']):.1f}" for r in rows)]
(W / "brat/out/REPORT_ALL.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))
