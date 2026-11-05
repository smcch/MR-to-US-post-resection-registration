"""Near-duplicate MR reference landmarks (pairs < 5 mm apart in the shared MR set) and the distance between the
US counterparts of each pair for every observer (SANTI R1, R2, OLGA). Also ReMIND-065 (excluded case) for completeness."""
import glob, json, os, re
import numpy as np
A = "/mnt/c/ReMIND_biomech/annotation_remind"
def load(p):
    if not os.path.exists(p):
        return None
    j = json.load(open(p, encoding="utf-8")); pts = {}
    for m in j["markups"]:
        s = np.array([-1., -1, 1]) if m.get("coordinateSystem", "LPS").upper() == "LPS" else np.ones(3)
        for c in m["controlPoints"]:
            pts[int(re.search(r"(\d+)\s*$", c["label"]).group(1))] = np.array(c["position"], float) * s
    return pts
cases = [(os.path.basename(f).replace("-MR.mrk.json", ""), f) for f in
         sorted(glob.glob(f"{A}/para_compartir/REMIND_landmarks/*-MR.mrk.json")) + sorted(glob.glob(f"{A}/para_compartir/BRATISLAVA_landmarks/*-MR.mrk.json"))]
cases.append(("ReMIND-065", f"{A}/ReMIND-065-MR.mrk.json"))
print("case        MR pair   MR dist | US dist S1   S2   OLGA")
for c, f in cases:
    MR = load(f); ext = c.startswith("Subject")
    S1 = load(f"{A}/{'LANDMARKS-BRATISLAVA' if ext else 'LANDMARKS'}/{c}-US-SANTI.mrk.json") or load(f"{A}/{c}-US-SANTI.mrk.json")
    S2 = load(f"{A}/para_compartir/SANTI-2/{c}-US-SANTI-2.mrk.json")
    O = load(f"{A}/OLGA-1/{'bratislava' if ext else 'remind'}/{c}-US-olga.mrk.json")
    ks = sorted(MR)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            d = np.linalg.norm(MR[a] - MR[b])
            if d < 5:
                us = [f"{np.linalg.norm(S[a] - S[b]):4.1f}" if S else "  - " for S in (S1, S2, O)]
                print(f"{c:11s} {a:>2d}-{b:<2d}     {d:4.1f}   |   {'  '.join(us)}")
