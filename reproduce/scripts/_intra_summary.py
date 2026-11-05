"""Intra-observer variability of US landmarks: SANTI round 1 vs round 2, ReMIND + external site.
Round 1 US: LANDMARKS/ (ReMIND) or LANDMARKS-BRATISLAVA/ (external); round 2: para_compartir/SANTI-2/;
shared MR: para_compartir/REMIND_landmarks or BRATISLAVA_landmarks. Coordinates LPS->RAS; pairs matched by order."""
import glob, os, json
import numpy as np
B = "/mnt/c/ReMIND_biomech/annotation_remind"
COH = {"ReMIND": ("ReMIND-*", "LANDMARKS", "REMIND_landmarks"),
       "External": ("Subject-*", "LANDMARKS-BRATISLAVA", "BRATISLAVA_landmarks")}
def load(p):
    j = json.load(open(p, encoding="utf-8")); pts, lab = [], []
    for m in j["markups"]:
        s = np.array([-1, -1, 1]) if m.get("coordinateSystem", "LPS").upper() == "LPS" else np.ones(3)
        for c in m["controlPoints"]:
            pts.append(np.array(c["position"], float) * s); lab.append(c.get("label", ""))
    return np.array(pts), lab
def key(c):
    return int(c.split("-")[1])
pooled = {}
for coh, (pat, r1dir, mrdir) in COH.items():
    files = sorted(glob.glob(f"{B}/para_compartir/SANTI-2/{pat}-US-SANTI-2.mrk.json"),
                   key=lambda f: key(os.path.basename(f)))
    print(f"\n## {coh} ({len(files)} cases)\n")
    print("| Case | intra mean | SD | median | max | >3 mm | >5 mm | nav R1 | nav R2 | pairs >5 mm |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    D, cm, n1s, n2s, warn = [], [], [], [], []
    for f in files:
        c = os.path.basename(f).split("-US")[0]
        r1, l1 = load(f"{B}/{r1dir}/{c}-US-SANTI.mrk.json"); r2, l2 = load(f)
        mr, _ = load(f"{B}/para_compartir/{mrdir}/{c}-MR.mrk.json")
        if len(r1) != len(r2) or l1 != l2:
            warn.append(f"{c}: {len(r1)} vs {len(r2)} points, labels equal={l1 == l2}")
        n = min(len(r1), len(r2))
        d = np.linalg.norm(r1[:n] - r2[:n], axis=1)
        n1 = np.linalg.norm(mr[:n] - r1[:n], axis=1).mean(); n2 = np.linalg.norm(mr[:n] - r2[:n], axis=1).mean()
        D += list(d); cm.append(d.mean()); n1s.append(n1); n2s.append(n2)
        big = ", ".join(f"{l2[i]} ({d[i]:.1f})" for i in np.argsort(-d) if d[i] > 5)
        print(f"| {c} | {d.mean():.2f} | {d.std(ddof=1):.2f} | {np.median(d):.2f} | {d.max():.1f} | {(d>3).sum()} | {(d>5).sum()} | {n1:.2f} | {n2:.2f} | {big} |")
    D = np.array(D); cm = np.array(cm); n1s = np.array(n1s); n2s = np.array(n2s)
    pooled[coh] = (D, cm)
    print(f"\n{coh}: {len(D)} pairs; pair mean {D.mean():.2f} ± {D.std(ddof=1):.2f} mm, median {np.median(D):.2f}, "
          f"P90 {np.percentile(D,90):.2f}; >3 mm {(D>3).sum()} ({100*(D>3).mean():.0f}%), >5 mm {(D>5).sum()} ({100*(D>5).mean():.0f}%); "
          f"case means {cm.mean():.2f} ± {cm.std(ddof=1):.2f} (range {cm.min():.2f}-{cm.max():.2f}); "
          f"nav R1 {n1s.mean():.2f} vs R2 {n2s.mean():.2f} (R2 closer in {(n2s < n1s).sum()}/{len(n1s)})")
    if warn: print("WARNINGS: " + "; ".join(warn))
D = np.concatenate([v[0] for v in pooled.values()]); cm = np.concatenate([v[1] for v in pooled.values()])
print(f"\n## Pooled ({len(cm)} cases, {len(D)} pairs)\npair mean {D.mean():.2f} ± {D.std(ddof=1):.2f} mm, median {np.median(D):.2f}, "
      f">3 mm {(D>3).sum()} ({100*(D>3).mean():.0f}%), >5 mm {(D>5).sum()} ({100*(D>5).mean():.0f}%); case means {cm.mean():.2f} ± {cm.std(ddof=1):.2f}")
# sensitivity: excluding pairs > 5 mm (gross disagreements to be resolved at consensus)
for coh, (Dc, _) in list(pooled.items()) + [("Pooled", (D, None))]:
    k = Dc[Dc <= 5]
    print(f"{coh} without pairs > 5 mm: {k.mean():.2f} ± {k.std(ddof=1):.2f} mm (n={len(k)})")
