"""Inter-observer analysis of the US landmarks: OLGA (round 1) vs SANTI (rounds 1 and 2).

Design: the MR point of every pair was placed once by SANTI and is shared (fixed); each observer placed the
US counterpart independently. SANTI placed his US points twice (round 1, round 2); OLGA once.
  intra  = |SANTI R1 - SANTI R2|                (same observer, single picks)
  inter  = mean(|OLGA - R1|, |OLGA - R2|)       (different observers, single picks; comparable to intra)
  nav    = |MR - US| per observer               (unregistered navigation alignment = initial misalignment)
Coordinates: Slicer markups in LPS -> RAS. Pairs matched by the trailing integer of the label.
Flag threshold THR = 5.79 mm: Maxwell P99.9 of SANTI's intra-observer distances (robust sigma from the median),
fixed on 2026-10-15, before OLGA's data arrived. A landmark is flagged when any of the three picks
(R1, R2, OLGA) lies more than THR from another one.
Outputs (C:\\ReMIND_work\\annotation): inter_observer_<date>.md, inter_observer_landmarks_<date>.csv,
consensus_review_list_<date>.csv.
"""
import csv
import datetime
import glob
import json
import os
import re
from collections import Counter

import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import maxwell, wilcoxon

A = "/mnt/c/ReMIND_biomech/annotation_remind"
OUT = "/mnt/c/ReMIND_work/annotation"
THR = 5.79
TODAY = datetime.date.today().isoformat()
COH = {
    "ReMIND": dict(r1="LANDMARKS", olga="OLGA-1/remind", mr="para_compartir/REMIND_landmarks",
                   img="para_compartir/REMIND_imagenes"),
    "External": dict(r1="LANDMARKS-BRATISLAVA", olga="OLGA-1/bratislava", mr="para_compartir/BRATISLAVA_landmarks",
                     img="para_compartir/BRATISLAVA_imagenes"),
}


def load(p):
    """Return {landmark number: RAS point}, checking that numbers are unique."""
    j = json.load(open(p, encoding="utf-8"))
    pts = {}
    for m in j["markups"]:
        s = np.array([-1.0, -1.0, 1.0]) if m.get("coordinateSystem", "LPS").upper() == "LPS" else np.ones(3)
        for c in m["controlPoints"]:
            if c.get("positionStatus", "defined") != "defined":
                continue
            k = int(re.search(r"(\d+)\s*$", c["label"]).group(1))
            assert k not in pts, f"duplicate landmark number {k} in {p}"
            pts[k] = np.array(c["position"], float) * s
    return pts


def us_check(img, x):
    """Point inside the US volume and on non-zero data (max over a 3x3x3 voxel neighbourhood)."""
    v = np.linalg.inv(img.affine) @ np.append(x, 1.0)
    i = np.round(v[:3]).astype(int)
    sh = np.array(img.shape[:3])
    if np.any(i < 0) or np.any(i >= sh):
        return "outside volume"
    lo, hi = np.maximum(i - 1, 0), np.minimum(i + 2, sh)
    d = np.asarray(img.dataobj[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]])
    return "ok" if d.max() > 0 else "zero intensity (outside cone)"


def f2(x):
    return f"{x:.2f}"


def fc(x):  # decimal comma for the CSV files (Spanish Excel)
    return f"{x:.1f}".replace(".", ",")


rows, warn, NOTES = [], [], {}
for coh, P in COH.items():
    files = sorted(glob.glob(f"{A}/{P['olga']}/*-US-olga.mrk.json"),
                   key=lambda f: int(os.path.basename(f).split("-")[1]))
    for f in files:
        c = os.path.basename(f).replace("-US-olga.mrk.json", "")
        O = load(f)
        R1 = load(f"{A}/{P['r1']}/{c}-US-SANTI.mrk.json")
        R2 = load(f"{A}/para_compartir/SANTI-2/{c}-US-SANTI-2.mrk.json")
        MR = load(f"{A}/{P['mr']}/{c}-MR.mrk.json")
        ks = sorted(MR)
        for name, S in (("OLGA", O), ("SANTI R1", R1), ("SANTI R2", R2)):
            if sorted(S) != ks:
                warn.append(f"{c}: {name} landmark numbers {sorted(S)} differ from MR {ks}")
        img = nib.load(glob.glob(f"{A}/{P['img']}/{c}/*US*post*.nii.gz")[0])
        for k in ks:
            if k not in O or k not in R1 or k not in R2:
                continue
            o, r1, r2, mr = O[k], R1[k], R2[k], MR[k]
            sm = (r1 + r2) / 2
            rows.append(dict(cohort=coh, case=c, k=k, o=o, r1=r1, r2=r2, mr=mr,
                             intra=np.linalg.norm(r1 - r2), o1=np.linalg.norm(o - r1), o2=np.linalg.norm(o - r2),
                             om=np.linalg.norm(o - sm), nav_o=np.linalg.norm(mr - o),
                             nav_s=np.linalg.norm(mr - sm), qc=us_check(img, o)))
        # label-swap check: Olga points far from SANTI's same label but close to another SANTI label
        for k in ks:
            if min(np.linalg.norm(O[k] - R1[k]), np.linalg.norm(O[k] - R2[k])) > THR:
                for j in ks:
                    if j != k and min(np.linalg.norm(O[k] - R1[j]), np.linalg.norm(O[k] - R2[j])) < 3.0:
                        own = np.linalg.norm(O[k] - O[j])
                        kind = (f"duplicate: OLGA US-{k} and US-{j} are {own:.1f} mm apart" if own < 3.0
                                else "possible numbering swap")
                        warn.append(f"{c}: OLGA US-{k} lies < 3 mm from SANTI US-{j} ({kind})")
        # near-duplicates inside each file (two landmarks on the same structure)
        for name, S in (("MR", MR), ("OLGA", O), ("SANTI R1", R1), ("SANTI R2", R2)):
            for i, a in enumerate(ks):
                for b in ks[i + 1:]:
                    if a in S and b in S and np.linalg.norm(S[a] - S[b]) < 3.0:
                        dd = np.linalg.norm(S[a] - S[b])
                        warn.append(f"{c}: {name} points {a} and {b} only {dd:.1f} mm apart")
                        for x, y in ((a, b), (b, a)):
                            NOTES.setdefault((c, x), []).append(f"{name} duplicate of US-{y} ({dd:.1f} mm)".replace("MR duplicate of US-", "MR duplicate of MR-"))

for r in rows:
    d = (r["intra"], r["o1"], r["o2"])
    if max(d) <= THR:
        r["cat"] = "agree"
    elif r["intra"] <= THR:
        r["cat"] = "OLGA differs from both SANTI rounds" if min(r["o1"], r["o2"]) > THR else "OLGA near one SANTI round only"
    elif r["o1"] <= THR and r["o2"] > THR:
        r["cat"] = "SANTI rounds differ, OLGA supports R1"
    elif r["o2"] <= THR and r["o1"] > THR:
        r["cat"] = "SANTI rounds differ, OLGA supports R2"
    elif r["o1"] <= THR and r["o2"] <= THR:
        r["cat"] = "SANTI rounds differ, OLGA between them"
    else:
        r["cat"] = "all three differ"
    r["inter"] = (r["o1"] + r["o2"]) / 2
    r["flag"] = r["cat"] != "agree"

L = []  # report lines
w = L.append
w(f"# Inter-observer variability of the US landmarks (OLGA vs SANTI), computed {TODAY}")
w("")
w("MR points fixed and shared (placed by SANTI); US points placed independently by each observer. "
  "SANTI: two rounds; OLGA: one. Distances in mm, RAS. Script: `scripts/_inter_observer.sh`.")
w(f"Flag threshold {THR} mm, fixed on 2026-10-15 from SANTI's intra-observer data (before OLGA's data).")
w("")
w("## Checks")
qc = Counter(r["qc"] for r in rows)
w(f"- Landmarks compared: {len(rows)} in {len(set(r['case'] for r in rows))} cases; OLGA points on US data: {dict(qc)}")
w("- Numbering / label warnings: " + ("none" if not warn else ""))
for x in warn:
    w(f"  - {x}")
bad_qc = [r for r in rows if r["qc"] != "ok"]
for r in bad_qc:
    w(f"  - {r['case']} US-{r['k']}: OLGA point {r['qc']}")

# per-case table
w("")
w("## Per case")
w("")
w("| Case | intra (S1 vs S2) | inter (OLGA vs S) | OLGA vs S mean | systematic offset | nav OLGA | nav SANTI | reorderings | flagged |")
w("|---|---|---|---|---|---|---|---|---|")
case_stats = []
for coh in COH:
    for c in dict.fromkeys(r["case"] for r in rows if r["cohort"] == coh):
        R = [r for r in rows if r["case"] == c]
        O = np.array([r["o"] for r in R]); S = np.array([(r["r1"] + r["r2"]) / 2 for r in R])
        t = (O - S).mean(0)
        D = np.linalg.norm(O[:, None] - S[None], axis=-1); ri, ci = linear_sum_assignment(D)
        cs = dict(cohort=coh, case=c, intra=np.mean([r["intra"] for r in R]), inter=np.mean([r["inter"] for r in R]),
                  om=np.mean([r["om"] for r in R]), shift=np.linalg.norm(t),
                  resid=np.linalg.norm(O - S - t, axis=1).mean(), nav_o=np.mean([r["nav_o"] for r in R]),
                  nav_s=np.mean([r["nav_s"] for r in R]), reord=int((ri != ci).sum()), nflag=sum(r["flag"] for r in R))
        case_stats.append(cs)
        w(f"| {c} | {f2(cs['intra'])} | {f2(cs['inter'])} | {f2(cs['om'])} | {f2(cs['shift'])} (resid. {f2(cs['resid'])}) | "
          f"{f2(cs['nav_o'])} | {f2(cs['nav_s'])} | {cs['reord']} | {cs['nflag']} |")

# summaries
w("")
w("## Summary (mean per case, then mean ± SD over cases = literature format; pairs pooled in brackets)")
w("")
sig_rows = []
for coh in list(COH) + ["Pooled"]:
    R = rows if coh == "Pooled" else [r for r in rows if r["cohort"] == coh]
    C = case_stats if coh == "Pooled" else [s for s in case_stats if s["cohort"] == coh]
    intra = np.array([r["intra"] for r in R]); inter = np.array([r["inter"] for r in R])
    ci_ = np.array([s["intra"] for s in C]); ce_ = np.array([s["inter"] for s in C])
    p = wilcoxon(ce_, ci_).pvalue if len(C) >= 6 else float("nan")
    keep = np.array([not r["flag"] for r in R])
    ci_k = np.array([np.mean([r["intra"] for r in R if r["case"] == s["case"] and not r["flag"]] or [np.nan]) for s in C])
    ce_k = np.array([np.mean([r["inter"] for r in R if r["case"] == s["case"] and not r["flag"]] or [np.nan]) for s in C])
    sig_i = np.median(intra) / maxwell.median(); sig_e = np.median(inter) / maxwell.median()
    sig_rows.append((coh, sig_i, sig_e))
    w(f"**{coh}** ({len(C)} cases, {len(R)} landmarks)")
    w(f"- intra-observer (SANTI): {ci_.mean():.2f} ± {ci_.std(ddof=1):.2f} [pairs {intra.mean():.2f} ± {intra.std(ddof=1):.2f}, median {np.median(intra):.2f}]")
    w(f"- inter-observer (OLGA vs SANTI): {ce_.mean():.2f} ± {ce_.std(ddof=1):.2f} [pairs {inter.mean():.2f} ± {inter.std(ddof=1):.2f}, median {np.median(inter):.2f}]; "
      f"inter > intra in {(ce_ > ci_).sum()}/{len(C)} cases, Wilcoxon p = {p:.3g}")
    w(f"- landmarks agreeing (all three picks within {THR} mm): {keep.sum()}/{len(R)} ({100 * keep.mean():.0f} %); "
      f"on these, intra {np.nanmean(ci_k):.2f} ± {np.nanstd(ci_k, ddof=1):.2f}, inter {np.nanmean(ce_k):.2f} ± {np.nanstd(ce_k, ddof=1):.2f} (mean per case)")
    w(f"- navigation misalignment MR vs US: OLGA {np.mean([s['nav_o'] for s in C]):.2f}, SANTI (mean of rounds) {np.mean([s['nav_s'] for s in C]):.2f}")
    w("")
mrdup = sorted({(c, max(a, b)) for (c, a), v in NOTES.items() for n in v if n.startswith("MR duplicate")
                for b in [int(re.search(r"MR-(\d+)", n).group(1))]})
R = [r for r in rows if (r["case"], r["k"]) not in mrdup]
ci_ = np.array([np.mean([r["intra"] for r in R if r["case"] == s["case"]]) for s in case_stats])
ce_ = np.array([np.mean([r["inter"] for r in R if r["case"] == s["case"]]) for s in case_stats])
w(f"Sensitivity, MR reference points closer than 3 mm to another MR point ({len(mrdup)} pairs in "
  f"{len(set(c for c, _ in mrdup))} cases: {', '.join(f'{c} MR-{k}' for c, k in mrdup)}) counted once "
  f"(higher number dropped, {len(R)} landmarks): intra {ci_.mean():.2f} ± {ci_.std(ddof=1):.2f}, "
  f"inter {ce_.mean():.2f} ± {ce_.std(ddof=1):.2f} (mean per case, pooled)")
w("")
w("Robust localisation spread (Maxwell sigma per axis from the median distance; per-pick sigma = sigma / sqrt 2 when both picks are equally noisy):")
for coh, si, se in sig_rows:
    w(f"- {coh}: intra sigma {si:.2f} mm, inter sigma {se:.2f} mm; extra inter-observer component sqrt(inter² - intra²) = {np.sqrt(max(se**2 - si**2, 0)):.2f} mm per axis")

# categories
w("")
w("## Consensus triage (per landmark)")
w("")
cats = Counter(r["cat"] for r in rows)
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    w(f"- {k}: {v}")
w("")
w("Flagged landmarks by case: " + ", ".join(f"{c} {n}" for c, n in sorted(Counter(r['case'] for r in rows if r['flag']).items(),
                                                                         key=lambda x: (-x[1], x[0]))))
open(f"{OUT}/inter_observer_{TODAY}.md", "w", encoding="utf-8").write("\n".join(L) + "\n")

hdr = ["cohort", "case", "landmark", "category", "intra_S1_S2", "OLGA_S1", "OLGA_S2", "MR_OLGA", "MR_Smean",
       "S1_R", "S1_A", "S1_S", "S2_R", "S2_A", "S2_S", "OLGA_R", "OLGA_A", "OLGA_S", "MR_R", "MR_A", "MR_S", "OLGA_qc",
       "notes"]


def line(r):
    return [r["cohort"], r["case"], f"US-{r['k']}", r["cat"], fc(r["intra"]), fc(r["o1"]), fc(r["o2"]), fc(r["nav_o"]),
            fc(r["nav_s"])] + [fc(v) for v in (*r["r1"], *r["r2"], *r["o"], *r["mr"])] + [r["qc"],
                                                                                             " | ".join(NOTES.get((r["case"], r["k"]), []))]


with open(f"{OUT}/inter_observer_landmarks_{TODAY}.csv", "w", newline="", encoding="utf-8") as fh:
    cw = csv.writer(fh, delimiter=";"); cw.writerow(hdr); cw.writerows(line(r) for r in rows)
nf = Counter(r["case"] for r in rows if r["flag"])
flag = sorted([r for r in rows if r["flag"]],
              key=lambda r: (r["cohort"] != "ReMIND", -nf[r["case"]], r["case"], -max(r["intra"], r["o1"], r["o2"])))
with open(f"{OUT}/consensus_review_list_{TODAY}.csv", "w", newline="", encoding="utf-8") as fh:
    cw = csv.writer(fh, delimiter=";"); cw.writerow(hdr); cw.writerows(line(r) for r in flag)
print("\n".join(L))
print(f"\n{len(flag)} flagged landmarks -> {OUT}/consensus_review_list_{TODAY}.csv")
