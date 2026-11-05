"""Outlier pairs in the intra-observer US differences (round 1 vs round 2).
Model: localization error isotropic Gaussian per axis -> 3D distance between two picks ~ Maxwell(sigma).
sigma fitted robustly from the median (Maxwell median = 1.5382 sigma), so outliers do not inflate it.
Outlier = distance above the Maxwell P99.9 (expected false positives ~0.25 in 250 pairs).
Sensitivity: Tukey upper fence Q3 + 1.5 IQR and far fence Q3 + 3 IQR on the pooled distances."""
import glob, os, json
import numpy as np
from scipy.stats import maxwell
B = "/mnt/c/ReMIND_biomech/annotation_remind"
def load(p):
    j = json.load(open(p, encoding="utf-8")); pts, lab = [], []
    for m in j["markups"]:
        s = np.array([-1, -1, 1]) if m.get("coordinateSystem", "LPS").upper() == "LPS" else np.ones(3)
        for c in m["controlPoints"]:
            pts.append(np.array(c["position"], float) * s); lab.append(c.get("label", ""))
    return np.array(pts), lab
rows = []
for f in glob.glob(f"{B}/para_compartir/SANTI-2/*-US-SANTI-2.mrk.json"):
    c = os.path.basename(f).split("-US")[0]
    r1dir = "LANDMARKS-BRATISLAVA" if c.startswith("Subject") else "LANDMARKS"
    mrdir = "BRATISLAVA_landmarks" if c.startswith("Subject") else "REMIND_landmarks"
    r1, _ = load(f"{B}/{r1dir}/{c}-US-SANTI.mrk.json"); r2, lab = load(f)
    mr, _ = load(f"{B}/para_compartir/{mrdir}/{c}-MR.mrk.json")
    for i in range(len(r2)):
        rows.append((c, lab[i], np.linalg.norm(r1[i] - r2[i]), np.linalg.norm(mr[i] - r1[i]), np.linalg.norm(mr[i] - r2[i])))
d = np.array([r[2] for r in rows]); n = len(d)
sig = np.median(d) / maxwell.median()
thr = {p: maxwell.ppf(p, scale=sig) for p in (0.99, 0.999)}
q1, q3 = np.percentile(d, [25, 75]); iqr = q3 - q1
tk, tk3 = q3 + 1.5 * iqr, q3 + 3 * iqr
print(f"pairs {n}; robust sigma per axis {sig:.2f} mm (per-pick sigma {sig/np.sqrt(2):.2f}); "
      f"Maxwell P99 {thr[0.99]:.2f} mm, P99.9 {thr[0.999]:.2f} mm; Tukey fence {tk:.2f}, far fence {tk3:.2f}")
for name, t in [("Maxwell P99.9", thr[0.999]), ("Maxwell P99", thr[0.99]), ("Tukey 1.5 IQR", tk), ("Tukey 3 IQR", tk3)]:
    k = d <= t
    print(f"  {name:14s} > {t:5.2f} mm: {(~k).sum():3d} pairs ({100*(~k).mean():.0f}%; expected by chance {n*(1-maxwell.cdf(t, scale=sig)):.1f}); "
          f"remaining mean {d[k].mean():.2f} ± {d[k].std(ddof=1):.2f}, median {np.median(d[k]):.2f}")
# goodness of fit of the bulk: observed vs expected share below thresholds
for t in (1, 2, 3, 4):
    print(f"  share <= {t} mm: observed {np.mean(d <= t):.2f}, Maxwell {maxwell.cdf(t, scale=sig):.2f}")
print("\nOutliers (> Maxwell P99.9), sorted: case, label, intra, MR-US round 1, MR-US round 2")
out = sorted([r for r in rows if r[2] > thr[0.999]], key=lambda r: -r[2])
for r in out:
    print(f"  {r[0]:11s} {r[1]:6s} {r[2]:5.1f}   nav R1 {r[3]:5.1f}  nav R2 {r[4]:5.1f}")
from collections import Counter
print("\nper case:", dict(sorted(Counter(r[0] for r in out).items(), key=lambda x: -x[1])))

# Two-component Maxwell mixture (core localization noise + gross misidentification), EM fit.
s = np.sort(d); print("\nlargest 30 distances:", np.round(s[-30:][::-1], 1).tolist())
w, s1, s2 = 0.8, 1.0, 4.0
for _ in range(500):
    p1 = w * maxwell.pdf(d, scale=s1); p2 = (1 - w) * maxwell.pdf(d, scale=s2)
    r = p1 / (p1 + p2)
    w = r.mean(); s1 = np.sqrt((r * d**2).sum() / (3 * r.sum())); s2 = np.sqrt(((1 - r) * d**2).sum() / (3 * (1 - r).sum()))
ll2 = np.log(w * maxwell.pdf(d, scale=s1) + (1 - w) * maxwell.pdf(d, scale=s2)).sum()
s0 = np.sqrt((d**2).sum() / (3 * n)); ll1 = np.log(maxwell.pdf(d, scale=s0)).sum()
grid = np.linspace(0.5, 15, 2901)
post = (1 - w) * maxwell.pdf(grid, scale=s2) / (w * maxwell.pdf(grid, scale=s1) + (1 - w) * maxwell.pdf(grid, scale=s2))
cut = grid[np.argmax(post > 0.5)]; cut9 = grid[np.argmax(post > 0.9)]
print(f"mixture: core {100*w:.0f}% sigma {s1:.2f} mm (core mean distance {s1*2*np.sqrt(2/np.pi):.2f}), "
      f"tail {100*(1-w):.0f}% sigma {s2:.2f} mm; BIC 1-comp {-2*ll1+np.log(n):.1f} vs 2-comp {-2*ll2+3*np.log(n):.1f}")
print(f"posterior P(gross) > 0.5 above {cut:.2f} mm ({(d > cut).sum()} pairs); > 0.9 above {cut9:.2f} mm ({(d > cut9).sum()} pairs)")
for t in (cut, cut9):
    k = d <= t; print(f"  without pairs > {t:.2f}: mean {d[k].mean():.2f} ± {d[k].std(ddof=1):.2f}, median {np.median(d[k]):.2f} (n={k.sum()})")
