"""Intra-observer variability after removing outlier pairs, per cohort (thresholds from _intra_outliers.py)."""
import glob, os, json
import numpy as np
B = "/mnt/c/ReMIND_biomech/annotation_remind"
def load(p):
    j = json.load(open(p, encoding="utf-8")); pts = []
    for m in j["markups"]:
        s = np.array([-1, -1, 1]) if m.get("coordinateSystem", "LPS").upper() == "LPS" else np.ones(3)
        pts += [np.array(c["position"], float) * s for c in m["controlPoints"]]
    return np.array(pts)
data = {"ReMIND": {}, "External": {}}
for f in glob.glob(f"{B}/para_compartir/SANTI-2/*-US-SANTI-2.mrk.json"):
    c = os.path.basename(f).split("-US")[0]; ext = c.startswith("Subject")
    r1 = load(f"{B}/{'LANDMARKS-BRATISLAVA' if ext else 'LANDMARKS'}/{c}-US-SANTI.mrk.json"); r2 = load(f)
    data["External" if ext else "ReMIND"][c] = np.linalg.norm(r1 - r2, axis=1)
data["Pooled"] = {**data["ReMIND"], **data["External"]}
for name, t in [("Maxwell P99.9 (> 5.79 mm)", 5.79), ("Tukey (> 8.46 mm)", 8.46)]:
    print(f"\n{name}")
    for coh, dd in data.items():
        allp = np.concatenate(list(dd.values())); k = allp[allp <= t]
        cm = np.array([v[v <= t].mean() for v in dd.values()])
        print(f"  {coh:8s}: pairs {len(k)}/{len(allp)}; mean {k.mean():.2f} ± {k.std(ddof=1):.2f} mm, median {np.median(k):.2f}, "
              f"P90 {np.percentile(k,90):.2f}; case means {cm.mean():.2f} ± {cm.std(ddof=1):.2f} (range {cm.min():.2f}-{cm.max():.2f})")
