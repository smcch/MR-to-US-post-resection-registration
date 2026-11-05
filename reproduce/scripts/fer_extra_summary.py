"""Summary of the additional deformable comparison (fer_extra_deformables.py): per cohort and overall, mean TRE of
affine | convexadam | deeds | syn | f3d_crop | f3d (manuscript, uncropped) | FER (proposed) | MCPO (reference, own
initialisation); cases improved from the affine; Wilcoxon (zsplit) of each comparator vs the affine and vs FER;
regularity (fold %, SDLogJ) and median wall time.  Internal, not in the manuscript.
Usage: python fer_extra_summary.py [cone|box]   -> fer/EXTRA_DEFORMABLES[_box].md (and printed)"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

W = Path("/mnt/c/ReMIND_work")
CROP = sys.argv[1] if len(sys.argv) > 1 else "cone"
EX = W / ("fer/extra_deformables" if CROP == "cone" else "fer/extra_deformables_box")
METHODS = ["convexadam", "convexadam_mask", "deeds", "syn", "f3d_crop"]
METHODS = [m for m in METHODS if (EX / m).exists()]
LABEL = {"resect_after": "RESECT post", "bite_g4": "BITE", "remind": "ReMIND", "external": "External"}
per = {(r["cohort"], r["case"]): r for r in csv.DictReader(open(W / "fer/tables/PER_CASE.csv"))}
rows = []
for (coh, c), r in per.items():
    if coh not in LABEL:
        continue
    d = dict(cohort=coh, case=c, affine=float(r["affine"]), fer=float(r["fer"]), f3d=float(r["f3d_mind"]),
             mcpo=float(r["mcpo"]) if r["mcpo"] not in ("", "nan") else np.nan, fer_sdlogj=float(r["fer_sdlogj"]),
             fer_fold=float(r["fer_fold"]))
    for m in METHODS:
        f = EX / m / coh / f"{c}.json"
        if f.exists():
            j = json.loads(f.read_text())
            d[m] = j["tre"]; d[m + "_fold"] = j["fold_pct"]; d[m + "_sdlogj"] = j["sdlogj"]; d[m + "_wall"] = j["wall_s"]
        else:
            d[m] = np.nan
    rows.append(d)


def col(rr, k):
    return np.array([r.get(k, np.nan) for r in rr], float)


def p(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5 or np.allclose(x[ok], y[ok]):
        return "-"
    return f"{stats.wilcoxon(x[ok], y[ok], zero_method='zsplit').pvalue:.3f}"


L = [f"# Additional deformable comparators on the challenge affine (MR cut to the iUS box, crop = {CROP}; 1 mm grid; internal, not in the paper)", "",
     "Mean TRE (mm) over cases; n = cases with a result for every method of the row.", "",
     "| set | n | affine | " + " | ".join(METHODS) + " | f3d (paper) | FER (ours) | MCPO (own init) |", "|---|---|---|" + "---|" * (len(METHODS) + 3)]
groups = [(LABEL[c], [r for r in rows if r["cohort"] == c]) for c in LABEL] + [("All post-resection", rows)]
for lab, rr in groups:
    ok = np.all([np.isfinite(col(rr, m)) for m in METHODS], axis=0)
    sub = [r for r, o in zip(rr, ok) if o]
    if not sub:
        continue
    L.append(f"| {lab} | {len(sub)} | " + " | ".join(f"{np.nanmean(col(sub, k)):.2f}" for k in ["affine"] + METHODS + ["f3d", "fer", "mcpo"]) + " |")
L += ["", "Cases improved from the affine (and Wilcoxon p vs affine); p vs FER (ours):", "",
      "| set | " + " | ".join(METHODS + ["f3d (paper)", "FER"]) + " |", "|---|" + "---|" * (len(METHODS) + 2)]
for lab, rr in groups:
    ok = np.all([np.isfinite(col(rr, m)) for m in METHODS], axis=0)
    sub = [r for r, o in zip(rr, ok) if o]
    if not sub:
        continue
    a, fer = col(sub, "affine"), col(sub, "fer")
    cells = []
    for m in METHODS + ["f3d", "fer"]:
        x = col(sub, m)
        cells.append(f"{int((x < a).sum())}/{len(sub)} (p {p(x, a)}; vs FER p {p(x, fer)})" if m != "fer" else f"{int((x < a).sum())}/{len(sub)} (p {p(x, a)})")
    L.append(f"| {lab} | " + " | ".join(cells) + " |")
L += ["", "Regularity over the cone (mean fold %, mean SDLogJ) and median wall time (s), all post-resection cases:", "",
      "| method | fold % | SDLogJ | wall s |", "|---|---|---|---|"]
for m in METHODS:
    ok = np.isfinite(col(rows, m))
    if ok.any():
        L.append(f"| {m} | {np.nanmean(col(rows, m + '_fold')):.3f} | {np.nanmean(col(rows, m + '_sdlogj')):.3f} | {np.nanmedian(col(rows, m + '_wall')):.0f} |")
L.append(f"| FER (ours) | {np.nanmean(col(rows, 'fer_fold')):.3f} | {np.nanmean(col(rows, 'fer_sdlogj')):.3f} | 41 |")
L += ["", "Per case (mm):", "", "| set | case | affine | " + " | ".join(METHODS) + " | f3d | FER | MCPO |", "|---|---|---|" + "---|" * (len(METHODS) + 3)]
for r in rows:
    L.append(f"| {LABEL[r['cohort']]} | {r['case']} | {r['affine']:.2f} | " + " | ".join(f"{r[m]:.2f}" if np.isfinite(r[m]) else "-" for m in METHODS)
             + f" | {r['f3d']:.2f} | {r['fer']:.2f} | " + (f"{r['mcpo']:.2f}" if np.isfinite(r['mcpo']) else "-") + " |")
(W / ("fer/EXTRA_DEFORMABLES.md" if CROP == "cone" else "fer/EXTRA_DEFORMABLES_box.md")).write_text("\n".join(L), encoding="utf-8")
print("\n".join(L[:40]))
