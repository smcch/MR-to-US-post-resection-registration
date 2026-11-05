"""For the candidate cases of each cohort, find the axial slices (working grid) in which the landmark pairs within
6 mm (>= 3 pairs) show the clearest advantage of the proposed method over BOTH the affine and MCPO:
margin = min(mean affine error - mean FER error, mean MCPO error - mean FER error) over those pairs.
Usage: python _slice_search.py [cohort ...]   -> prints, per case, the best slices (k, n pairs, a/f/m, margin)."""
import csv
import sys

import numpy as np

import fig_lib as L

rows = list(csv.DictReader(open(L.W / "fer/tables/PER_CASE.csv")))
lmrows = list(csv.DictReader(open(L.W / "fer/tables/LANDMARKS.csv")))
cohorts = sys.argv[1:] or ["resect_after", "bite_g4", "remind", "external"]
for coh in cohorts:
    rr = [r for r in rows if r["cohort"] == coh and r["mcpo"] not in ("", "nan")]
    rr.sort(key=lambda r: min(float(r["affine"]) - float(r["fer"]), float(r["mcpo"]) - float(r["fer"])), reverse=True)
    print(f"== {coh}")
    for r in rr[:6]:
        c = r["case"]
        cm = min(float(r["affine"]) - float(r["fer"]), float(r["mcpo"]) - float(r["fer"]))
        if cm <= 0:
            continue
        d = L.load_case(coh, c)
        E = [x for x in lmrows if x["cohort"] == coh and x["case"] == c]
        err = {a: np.array([float(x[a]) for x in E]) for a in ("affine", "fer", "mcpo")}
        inv = np.linalg.inv(d["aff"])
        lm_v = d["lm_us"] @ inv[:3, :3].T + inv[:3, 3]
        nz = [int(k) for k in np.unique(np.where(d["us"] > 0.02)[2])]
        best = []
        for k in nz:
            near = np.abs(lm_v[:, 2] - k) < 6
            if near.sum() < 3:
                continue
            a, f, m = (err[x][near].mean() for x in ("affine", "fer", "mcpo"))
            best.append((min(a - f, m - f), k, int(near.sum()), a, f, m))
        best.sort(reverse=True)
        # keep distinct slices (>= 4 apart)
        kept = []
        for b in best:
            if all(abs(b[1] - x[1]) >= 4 for x in kept):
                kept.append(b)
            if len(kept) == 4:
                break
        print(f"  {L.case_label(coh, c):14s} case: affine {float(r['affine']):.2f} FER {float(r['fer']):.2f} MCPO {float(r['mcpo']):.2f} | slices: "
              + "; ".join(f"k{b[1]} n{b[2]} a{b[3]:.1f} f{b[4]:.1f} m{b[5]:.1f} (+{b[0]:.1f})" for b in kept))
