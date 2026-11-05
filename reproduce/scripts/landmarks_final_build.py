"""Build the final two-observer landmark set (ReMIND + external site) and the observer-variability tables.

Inputs (never modified): shared MR points (SANTI, one pick), US picks SANTI round 1 + round 2, OLGA round 1.
Rules (fixed 2026-11-05):
 1. MR duplicates: landmarks are visited in ascending number; a landmark whose MR point lies < 3 mm from the MR
    point of an already kept landmark is the same feature and is merged into it (dropped).
 2. Outlier pick: a US pick lying more than T = 5.79 mm from both other picks of the same landmark is removed.
    T = Maxwell P99.9 of SANTI's intra-observer distances (robust sigma), fixed on 2026-10-15 before OLGA's data.
 3. A landmark is kept when its remaining picks include both observers and lie within T of each other.
 4. Final US point = mean of the observer means (SANTI = mean of his kept rounds; OLGA = her pick), so both
    observers weigh the same.  Final MR point = the shared MR point.
Variability (after rule 1, before rules 2-3): intra = SANTI R1 vs R2; inter = SANTI R1 vs OLGA and SANTI R2 vs OLGA;
all pairs, and without pairs > T.  Mean per case, then mean +- SD over cases (literature format).
Outputs: C:\\ReMIND_work\\landmarks_final\\ (see README.md there).
"""
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import shutil

import numpy as np

A = "/mnt/c/ReMIND_biomech/annotation_remind"
OUT = "/mnt/c/ReMIND_work/landmarks_final"
T, DUP = 5.79, 3.0
TODAY = datetime.date.today().isoformat()
LPS = np.array([-1.0, -1.0, 1.0])
COH = {
    "ReMIND": dict(r1=f"{A}/LANDMARKS", olga=f"{A}/OLGA-1/remind", mr=f"{A}/para_compartir/REMIND_landmarks"),
    "External": dict(r1=f"{A}/LANDMARKS-BRATISLAVA", olga=f"{A}/OLGA-1/bratislava", mr=f"{A}/para_compartir/BRATISLAVA_landmarks"),
}


def load(p):
    j = json.load(open(p, encoding="utf-8"))
    pts = {}
    for m in j["markups"]:
        s = LPS if m.get("coordinateSystem", "LPS").upper() == "LPS" else np.ones(3)
        for c in m["controlPoints"]:
            pts[int(re.search(r"(\d+)\s*$", c["label"]).group(1))] = np.array(c["position"], float) * s  # RAS
    return pts


def write_mrk(path, prefix, pts):
    """Slicer markups (schema 1.0.3), LPS, labels <prefix>-<original number>."""
    cps = [dict(id=str(k), label=f"{prefix}-{k}", description="", associatedNodeID="",
                position=[float(v) for v in pts[k] * LPS], orientation=[-1.0, -0.0, -0.0, -0.0, -1.0, -0.0, 0.0, 0.0, 1.0],
                selected=True, locked=True, visibility=True, positionStatus="defined") for k in sorted(pts)]
    j = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json#",
         "markups": [dict(type="Fiducial", coordinateSystem="LPS", coordinateUnits="mm", locked=True,
                          fixedNumberOfControlPoints=False, labelFormat="%N-%d", lastUsedControlPointNumber=max(pts),
                          controlPoints=cps)]}
    json.dump(j, open(path, "w", encoding="utf-8"), indent=4)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


dist = lambda a, b: float(np.linalg.norm(a - b))
audit, final, var = [], {}, []
for coh, P in COH.items():
    files = sorted(glob.glob(f"{P['olga']}/*-US-olga.mrk.json"), key=lambda f: int(os.path.basename(f).split("-")[1]))
    for f in files:
        c = os.path.basename(f).replace("-US-olga.mrk.json", "")
        O, MR = load(f), load(f"{P['mr']}/{c}-MR.mrk.json")
        S1, S2 = load(f"{P['r1']}/{c}-US-SANTI.mrk.json"), load(f"{A}/para_compartir/SANTI-2/{c}-US-SANTI-2.mrk.json")
        kept_mr, fin = [], {}
        for k in sorted(MR):
            rec = dict(cohort=coh, case=c, landmark=k, S1_S2=dist(S1[k], S2[k]), S1_O=dist(S1[k], O[k]), S2_O=dist(S2[k], O[k]),
                       Sm_O=dist((S1[k] + S2[k]) / 2, O[k]), Sk_O=float("nan"))   # OLGA vs mean of SANTI's rounds
            near = [(dist(MR[k], MR[j]), j) for j in kept_mr if dist(MR[k], MR[j]) < DUP]
            if near:
                dd, j = min(near)
                rec.update(decision="merged", reason=f"MR point {dd:.1f} mm from MR-{j} (same feature)", picks="")
                audit.append(rec)
                continue
            kept_mr.append(k)
            var.append(dict(cohort=coh, case=c, landmark=k, intra=rec["S1_S2"], inter1=rec["S1_O"], inter2=rec["S2_O"],
                            interm=rec["Sm_O"]))
            picks = {"S1": S1[k], "S2": S2[k], "O": O[k]}
            out = [p for p in picks if all(dist(picks[p], picks[q]) > T for q in picks if q != p)]
            for p in out:
                picks.pop(p)
            ok = "O" in picks and ({"S1", "S2"} & set(picks)) and all(
                dist(picks[a], picks[b]) <= T for a in picks for b in picks if a < b)
            removed = ", ".join({"S1": "SANTI round 1", "S2": "SANTI round 2", "O": "OLGA"}[p] for p in out)
            if ok:
                s = np.mean([picks[p] for p in ("S1", "S2") if p in picks], axis=0)
                fin[k] = (MR[k], (s + picks["O"]) / 2, "+".join(sorted(picks)))
                rec["Sk_O"] = dist(s, picks["O"])       # OLGA vs mean of SANTI's rounds kept (final set)
                rec.update(decision="kept", reason=(f"outlier pick removed: {removed}" if out else "all picks agree"),
                           picks="+".join(sorted(picks)))
            else:
                why = ("OLGA pick is an outlier (SANTI's rounds agree, no second-observer confirmation)" if out == ["O"]
                       else "picks spread wider than T without a single outlier pick" if not out
                       else "all three picks disagree" if len(out) > 1
                       else f"picks still disagree after removing {removed}")
                rec.update(decision="excluded", reason=why, picks="")
            audit.append(rec)
        final[(coh, c)] = fin

# ---------------- outputs ----------------
for sub in ("remind", "external", "public_ReMIND/landmarks"):  # regenerated folders only; README files are kept
    if os.path.isdir(f"{OUT}/{sub}"):
        shutil.rmtree(f"{OUT}/{sub}")
for sub in ("remind", "external", "public_ReMIND/landmarks"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
rows_csv = []
for (coh, c), fin in final.items():
    sub = "remind" if coh == "ReMIND" else "external"
    write_mrk(f"{OUT}/{sub}/{c}-MR.mrk.json", "MR", {k: v[0] for k, v in fin.items()})
    write_mrk(f"{OUT}/{sub}/{c}-US.mrk.json", "US", {k: v[1] for k, v in fin.items()})
    if coh == "ReMIND":
        shutil.copy(f"{OUT}/remind/{c}-MR.mrk.json", f"{OUT}/public_ReMIND/landmarks/")
        shutil.copy(f"{OUT}/remind/{c}-US.mrk.json", f"{OUT}/public_ReMIND/landmarks/")
    for k, (m, u, p) in sorted(fin.items()):
        rows_csv.append([coh, c, k] + [f"{v:.3f}" for v in (*(m * LPS), *(u * LPS))] + [p])
hdr = ["cohort", "case", "landmark", "MR_L", "MR_P", "MR_S", "US_L", "US_P", "US_S", "US_picks_averaged"]
with open(f"{OUT}/landmarks_final_LPS.csv", "w", newline="", encoding="utf-8") as fh:
    cw = csv.writer(fh); cw.writerow(hdr); cw.writerows(rows_csv)
with open(f"{OUT}/public_ReMIND/landmarks_ReMIND_LPS.csv", "w", newline="", encoding="utf-8") as fh:
    cw = csv.writer(fh); cw.writerow(["case", "landmark", "MR_L", "MR_P", "MR_S", "US_L", "US_P", "US_S"])
    cw.writerows([r[1:9] for r in rows_csv if r[0] == "ReMIND"])
with open(f"{OUT}/landmarks_audit.csv", "w", newline="", encoding="utf-8") as fh:
    cw = csv.writer(fh, delimiter=";")
    cw.writerow(["cohort", "case", "landmark", "decision", "reason", "picks_averaged", "SANTI_R1_vs_R2_mm",
                 "SANTI_R1_vs_OLGA_mm", "SANTI_R2_vs_OLGA_mm", "SANTI_mean_vs_OLGA_mm", "SANTI_kept_mean_vs_OLGA_mm"])
    for r in audit:
        cw.writerow([r["cohort"], r["case"], r["landmark"], r["decision"], r["reason"], r["picks"]] +
                    [("" if r[x] != r[x] else f"{r[x]:.2f}".replace(".", ",")) for x in ("S1_S2", "S1_O", "S2_O", "Sm_O", "Sk_O")])

# image manifest (the images the coordinates refer to)
img_rows = []
for (coh, c) in final:
    if coh == "ReMIND":
        d = f"{A}/SET2/{c}"
        mr = sorted(glob.glob(f"{d}/*_pre_T2.nii.gz") + glob.glob(f"{d}/*_MR_T2_preop.nii.gz"))[0]
        us = glob.glob(f"{d}/*_US_postresection.nii.gz")[0]
    else:
        d = f"{A}/SET-Bratislava/{c}"
        if c == "Subject-3":  # the landmarks were placed on the 3D T1 (author, 2026-11-05)
            mr, us = f"{d}/cs_T1W_3D_TFE_s301.nii.gz", f"{d}/Scan6_in_cs_T1W_3D_TFE.nii.gz"
        else:
            mr = (glob.glob(f"{d}/{c}-MR-pre.nii.gz") + glob.glob(f"{d}/{c}-T2-pre.nii.gz"))[0]
            us = f"{d}/{c}-US-post.nii.gz"
    img_rows.append([coh, c, mr.replace("/mnt/c/", "C:/"), sha(mr), us.replace("/mnt/c/", "C:/"), sha(us)])
with open(f"{OUT}/images_manifest.csv", "w", newline="", encoding="utf-8") as fh:
    cw = csv.writer(fh); cw.writerow(["cohort", "case", "MR_image", "MR_sha256", "US_image", "US_sha256"]); cw.writerows(img_rows)
man = {r[0]: r for r in csv.reader(open(f"{A}/SET2/MANIFEST.csv", encoding="utf-8"))}
SERIES = {"ReMIND-006": "3DAXT2SPACE", "ReMIND-023": "2DAXT2BLADE"}  # not in SET2/MANIFEST.csv; matched by geometry
with open(f"{OUT}/public_ReMIND/images.csv", "w", newline="", encoding="utf-8") as fh:
    cw = csv.writer(fh); cw.writerow(["case", "MR_series", "US_series", "n_landmarks"])
    for (coh, c), fin in final.items():
        if coh == "ReMIND":
            cw.writerow([c, man[c][1] if c in man else SERIES[c], "USpreimri", len(fin)])

# ---------------- report ----------------
L = []
w = L.append
w(f"# Final two-observer landmark set and observer variability ({TODAY})")
w("")
w("Script `scripts/landmarks_final_build.sh`; rules in the script header and in README.md. Distances in mm.")
for coh in ("ReMIND", "External", "Pooled"):
    R = audit if coh == "Pooled" else [r for r in audit if r["cohort"] == coh]
    cases = list(dict.fromkeys(r["case"] for r in R))
    n_dec = {d: sum(r["decision"] == d for r in R) for d in ("kept", "merged", "excluded")}
    per = [sum(1 for r in R if r["case"] == c and r["decision"] == "kept") for c in cases]
    w("")
    w(f"## {coh} ({len(cases)} cases, {len(R)} annotated landmarks)")
    w(f"- kept {n_dec['kept']}, merged as MR duplicates {n_dec['merged']}, excluded {n_dec['excluded']}; "
      f"kept per case median {int(np.median(per))} (range {min(per)}-{max(per)})")
    for why in sorted(set(r["reason"] for r in R if r["decision"] != "merged")):
        w(f"  - {why}: {sum(r['reason'] == why for r in R)}")
    V = var if coh == "Pooled" else [v for v in var if v["cohort"] == coh]
    w("")
    w("| comparison | all pairs: mean per case ± SD | median of pairs | without pairs > T: mean per case ± SD | pairs > T |")
    w("|---|---|---|---|---|")
    for lab, key in (("intra-observer, SANTI R1 vs R2", "intra"), ("inter-observer, mean of SANTI's rounds vs OLGA", "interm"),
                     ("inter-observer, SANTI R1 vs OLGA", "inter1"), ("inter-observer, SANTI R2 vs OLGA", "inter2")):
        cm = np.array([np.mean([v[key] for v in V if v["case"] == c]) for c in cases])
        cmk = np.array([np.mean([v[key] for v in V if v["case"] == c and v[key] <= T]) for c in cases])
        allp = np.array([v[key] for v in V])
        w(f"| {lab} | {cm.mean():.2f} ± {cm.std(ddof=1):.2f} | {np.median(allp):.2f} | {cmk.mean():.2f} ± {cmk.std(ddof=1):.2f} | "
          f"{(allp > T).sum()} of {len(allp)} |")
w("")
w("## Per case")
w("")
w("Landmarks: annotated 10 per case; merged = MR duplicates; kept = final set. Agreement = mean distance (mm) "
  "between US picks. 'All pairs' = every landmark after merging duplicates (before consensus); 'final set' = only "
  "the picks averaged into the released points (intra needs both SANTI rounds kept; n in brackets).")
w("")
w("| Case | merged | excluded | kept | intra all pairs | inter mean-OLGA all | inter R1-OLGA all | inter R2-OLGA all | "
  "intra final set | inter mean-OLGA final | inter R1-OLGA final | inter R2-OLGA final |")
w("|---|---|---|---|---|---|---|---|---|---|---|---|")
picks = {(r["case"], r["landmark"]): r["picks"].split("+") for r in audit if r["decision"] == "kept"}
skept = {(r["case"], r["landmark"]): r["Sk_O"] for r in audit if r["decision"] == "kept"}
for v in var:                                           # final set: OLGA vs the mean of SANTI's rounds kept
    v["intermk"] = skept.get((v["case"], v["landmark"]), float("nan"))
for (coh, c), fin in final.items():
    R = [r for r in audit if r["case"] == c]
    V = [v for v in var if v["case"] == c]
    def fm(sel, key):
        x = [v[key] for v in V if sel(v)]
        return f"{np.mean(x):.2f} ({len(x)})" if x else "-"
    kp = lambda v, need: (c, v["landmark"]) in picks and set(need) <= set(picks[(c, v["landmark"])])
    w(f"| {c} | {sum(r['decision'] == 'merged' for r in R)} | {sum(r['decision'] == 'excluded' for r in R)} | {len(fin)} | "
      f"{np.mean([v['intra'] for v in V]):.2f} | {np.mean([v['interm'] for v in V]):.2f} | {np.mean([v['inter1'] for v in V]):.2f} | "
      f"{np.mean([v['inter2'] for v in V]):.2f} | "
      f"{fm(lambda v: kp(v, ['S1', 'S2', 'O']), 'intra')} | {fm(lambda v: kp(v, ['O']), 'intermk')} | "
      f"{fm(lambda v: kp(v, ['S1', 'O']), 'inter1')} | {fm(lambda v: kp(v, ['S2', 'O']), 'inter2')} |")
for coh in ("ReMIND", "External", "Pooled"):
    Fk = {k: v for k, v in final.items() if coh == "Pooled" or k[0] == coh}
    w("")
    vals = {}
    for key, need in (("intra", ["S1", "S2", "O"]), ("intermk", ["O"]), ("inter1", ["S1", "O"]), ("inter2", ["S2", "O"])):
        cm = [np.mean(x) for (_, c) in Fk for x in [[v[key] for v in var if v["case"] == c and (c, v["landmark"]) in picks
                                                     and set(need) <= set(picks[(c, v["landmark"])])]] if x]
        vals[key] = f"{np.mean(cm):.2f} ± {np.std(cm, ddof=1):.2f}"
    w(f"Final set, {coh}: {sum(len(f) for f in Fk.values())} landmarks; intra {vals['intra']}, inter mean-OLGA {vals['intermk']}, "
      f"inter R1-OLGA {vals['inter1']}, inter R2-OLGA {vals['inter2']} (mean per case ± SD)")
open(f"{OUT}/REPORT.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))

# counts and observer variability for the manuscript macros (landmarks_final/STATS.json)
import runpy
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "landmarks_final_stats.py"))
