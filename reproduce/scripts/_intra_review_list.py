"""Checklist of US landmark pairs to review (round 1 vs round 2 > 5.79 mm, Maxwell P99.9 criterion).
Coordinates in RAS (as shown by 3D Slicer), both rounds, plus distance to the shared MR point."""
import glob, os, json, csv
import numpy as np
B = "/mnt/c/ReMIND_biomech/annotation_remind"; THR = 5.79
def load(p):
    j = json.load(open(p, encoding="utf-8")); pts, lab = [], []
    for m in j["markups"]:
        s = np.array([-1, -1, 1]) if m.get("coordinateSystem", "LPS").upper() == "LPS" else np.ones(3)
        for c in m["controlPoints"]:
            pts.append(np.array(c["position"], float) * s); lab.append(c.get("label", ""))
    return np.array(pts), lab
rows = []
for f in glob.glob(f"{B}/para_compartir/SANTI-2/*-US-SANTI-2.mrk.json"):
    c = os.path.basename(f).split("-US")[0]; ext = c.startswith("Subject")
    r1, _ = load(f"{B}/{'LANDMARKS-BRATISLAVA' if ext else 'LANDMARKS'}/{c}-US-SANTI.mrk.json"); r2, lab = load(f)
    for i in range(len(r2)):
        d = np.linalg.norm(r1[i] - r2[i])
        if d > THR:
            rows.append([c, lab[i], round(d, 1)] + [round(x, 1) for x in r1[i]] + [round(x, 1) for x in r2[i]])
from collections import Counter
n_case = Counter(r[0] for r in rows)
rows.sort(key=lambda r: (not r[0].startswith("ReMIND"), -n_case[r[0]], r[0], -r[2]))
out = "/mnt/c/ReMIND_work/annotation/review_list_SANTI_2026-10-16.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh, delimiter=";")
    w.writerow(["case", "landmark", "distance_mm", "R1_R", "R1_A", "R1_S", "R2_R", "R2_A", "R2_S"])
    w.writerows([[str(x).replace(".", ",") if isinstance(x, float) else x for x in r] for r in rows])
for r in rows: print(r)
print(len(rows), "pairs ->", out)
