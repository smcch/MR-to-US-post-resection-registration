"""Agreement indices between observers (OLGA vs SANTI) for the US landmarks; reads the per-landmark CSV
written by _inter_observer.py (inter_observer_landmarks_<date>.csv, RAS coordinates, decimal comma).
1. Percentage agreement: share of landmarks whose two observers' US points lie within d mm.
2. ICC(2,1) absolute agreement per RAS axis (two-way random, single measure); coordinates centred on each
   case's centroid so the case position in scanner space does not inflate the ICC.
3. Reliability of the measured initial misalignment |MR - US| (the quantity a TRE is built on): ICC(2,1),
   Bland-Altman bias and 95 % limits of agreement, per landmark and per case mean.
SANTI = mean of his two rounds for inter-observer indices; SANTI R1 vs R2 gives the intra-observer reference."""
import csv
import glob
import sys

import numpy as np

OUT = "/mnt/c/ReMIND_work/annotation"
f = sorted(glob.glob(f"{OUT}/inter_observer_landmarks_*.csv"))[-1]
date = f.rsplit("_", 1)[1].replace(".csv", "")
rows = list(csv.DictReader(open(f, encoding="utf-8"), delimiter=";"))
num = lambda s: float(s.replace(",", "."))
P = {}
for r in rows:
    for who in ("S1", "S2", "OLGA", "MR"):
        r[who] = np.array([num(r[f"{who}_{a}"]) for a in "RAS"])
    r["S"] = (r["S1"] + r["S2"]) / 2


def icc21(Y):
    """ICC(2,1), absolute agreement. Y: n subjects x k raters."""
    n, k = Y.shape
    gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def boot_ci(fun, idx_groups, B=2000, seed=0):
    """Case-level bootstrap CI (resample cases, keep their landmarks together)."""
    rng = np.random.default_rng(seed)
    cases = list(idx_groups)
    vals = []
    for _ in range(B):
        pick = rng.choice(len(cases), len(cases))
        idx = np.concatenate([idx_groups[cases[i]] for i in pick])
        vals.append(fun(idx))
    return np.percentile(vals, [2.5, 97.5])


L = []
w = L.append
w(f"# Agreement between observers (OLGA vs SANTI) for the US landmarks ({date})")
w("")
w("SANTI = mean of his two rounds (inter-observer); SANTI round 1 vs round 2 = intra-observer reference. "
  "95 % CI by case-level bootstrap. Script: `scripts/_inter_agreement.sh`.")
for coh in ("ReMIND", "External", "Pooled"):
    R = rows if coh == "Pooled" else [r for r in rows if r["cohort"] == coh]
    cases = list(dict.fromkeys(r["case"] for r in R))
    groups = {c: np.array([i for i, r in enumerate(R) if r["case"] == c]) for c in cases}
    d_inter = np.array([np.linalg.norm(r["OLGA"] - r["S"]) for r in R])
    d_inter1 = np.array([(np.linalg.norm(r["OLGA"] - r["S1"]) + np.linalg.norm(r["OLGA"] - r["S2"])) / 2 for r in R])
    d_intra = np.array([np.linalg.norm(r["S1"] - r["S2"]) for r in R])
    w("")
    w(f"## {coh} ({len(cases)} cases, {len(R)} landmarks)")
    w("")
    w("**1. Percentage agreement** (share of landmarks within d mm)")
    w("")
    w("| d (mm) | OLGA vs SANTI mean | OLGA vs single SANTI round | SANTI R1 vs R2 |")
    w("|---|---|---|---|")
    for d in (2, 3, 4, 5, 5.79):
        w(f"| {d} | {100 * (d_inter <= d).mean():.0f} % | {100 * (d_inter1 <= d).mean():.0f} % | {100 * (d_intra <= d).mean():.0f} % |")
    # ICC per axis, centred per case on SANTI's centroid
    w("")
    w("**2. ICC(2,1) per axis, coordinates centred per case**")
    w("")
    for lab, A_, B_ in (("inter (OLGA vs SANTI mean)", "OLGA", "S"), ("intra (SANTI R1 vs R2)", "S1", "S2")):
        out = []
        for ax, an in enumerate("RAS"):
            def fun(idx, ax=ax):
                Y = []
                for i in idx:
                    r = R[i]
                    cen = np.mean([R[j]["S"] for j in groups[r["case"]]], axis=0)
                    Y.append([r[A_][ax] - cen[ax], r[B_][ax] - cen[ax]])
                return icc21(np.array(Y))
            v = fun(np.arange(len(R)))
            lo, hi = boot_ci(fun, groups, B=500)
            out.append(f"{an} {v:.3f} ({lo:.3f}-{hi:.3f})")
        w(f"- {lab}: " + "; ".join(out))
    # reliability of the misalignment measure
    nav_o = np.array([np.linalg.norm(r["MR"] - r["OLGA"]) for r in R])
    nav_s = np.array([np.linalg.norm(r["MR"] - r["S"]) for r in R])
    nav_1 = np.array([np.linalg.norm(r["MR"] - r["S1"]) for r in R])
    nav_2 = np.array([np.linalg.norm(r["MR"] - r["S2"]) for r in R])
    w("")
    w("**3. Reliability of the measured initial misalignment |MR - US|**")
    w("")
    for lab, a, b in (("inter (OLGA vs SANTI mean)", nav_o, nav_s), ("intra (SANTI R1 vs R2)", nav_1, nav_2)):
        icc_l = icc21(np.column_stack([a, b]))
        lo, hi = boot_ci(lambda idx: icc21(np.column_stack([a[idx], b[idx]])), groups)
        ca = np.array([a[groups[c]].mean() for c in cases]); cb = np.array([b[groups[c]].mean() for c in cases])
        icc_c = icc21(np.column_stack([ca, cb]))
        diff = a - b; dc = ca - cb
        w(f"- {lab}: per landmark ICC {icc_l:.2f} ({lo:.2f}-{hi:.2f}), bias {diff.mean():+.2f} mm, "
          f"limits of agreement {diff.mean() - 1.96 * diff.std(ddof=1):+.2f} to {diff.mean() + 1.96 * diff.std(ddof=1):+.2f}; "
          f"per case mean ICC {icc_c:.2f}, bias {dc.mean():+.2f}, limits {dc.mean() - 1.96 * dc.std(ddof=1):+.2f} to "
          f"{dc.mean() + 1.96 * dc.std(ddof=1):+.2f} (case means {ca.mean():.2f} vs {cb.mean():.2f})")
open(f"{OUT}/inter_agreement_{date}.md", "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
