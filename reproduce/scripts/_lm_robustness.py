"""Quick check: does the ranking on ReMIND / external survive if the worst landmarks are discounted?
Per-case mean vs median vs 70th percentile (tre30 columns of PER_CASE.csv), and a leave-worst-landmark-out mean
from LANDMARKS.csv (drop, per case, the landmark with the largest error under the best-fit rigid = most suspect pair)."""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

W = "/mnt/c/ReMIND_work/fer/tables/"
per = [r for r in csv.DictReader(open(W + "PER_CASE.csv"))]
ARMS = ["affine", "fer", "mcpo_pose", "mcpo_pose_fer", "mcpo"]
for coh in ("remind", "external"):
    rr = [r for r in per if r["cohort"] == coh]
    print(f"\n== {coh} (n {len(rr)})")
    for stat in ("", "_median", "_tre30"):
        vals = {a: np.array([float(r[a + stat]) if r[a + stat] not in ("", "nan") else np.nan for r in rr]) for a in ARMS}
        line = " | ".join(f"{a} {np.nanmean(v):.2f}" for a, v in vals.items())
        ok = np.isfinite(vals["mcpo"])
        p = stats.wilcoxon(vals["fer"][ok], vals["mcpo"][ok], zero_method="zsplit").pvalue
        print(f"  case {stat or '_mean':8s}: {line} | FER<MCPO {(vals['fer'][ok] < vals['mcpo'][ok]).sum()}/{ok.sum()} p {p:.3f}")

lm = [r for r in csv.DictReader(open(W + "LANDMARKS.csv"))]
print("\nLANDMARKS.csv columns:", list(lm[0])[:20])

print("\nPrimary claim (refinement vs its initialization) under the three per-case statistics:")
for coh in ("remind", "external"):
    rr = [r for r in per if r["cohort"] == coh]
    for stat in ("", "_median", "_tre30"):
        v = lambda a: np.array([float(r[a + stat]) for r in rr])
        print(f"  {coh:8s} {stat or '_mean':8s}: FER<affine {(v('fer') < v('affine')).sum()}/{len(rr)} "
              f"(p {stats.wilcoxon(v('fer'), v('affine'), zero_method='zsplit').pvalue:.4f}) | "
              f"pose+FER<pose {(v('mcpo_pose_fer') < v('mcpo_pose')).sum()}/{len(rr)} "
              f"(p {stats.wilcoxon(v('mcpo_pose_fer'), v('mcpo_pose'), zero_method='zsplit').pvalue:.4f})")
