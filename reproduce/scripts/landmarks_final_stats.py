"""Counts and observer variability of the final two-observer landmark set, for the manuscript macros.
Reads landmarks_final/landmarks_audit.csv (written by landmarks_final_build.py); writes landmarks_final/STATS.json.
Variability = mean distance between iUS picks per case, then mean and SD over cases:
  'final' = only the picks averaged into the released points (intra needs both SANTI rounds kept);
  'all'   = every landmark after merging MR duplicates, before removing outliers (pre-consensus).
intra = SANTI round 1 vs round 2; inter_mean = mean of SANTI rounds vs OLGA (main text); inter_one / inter_two =
SANTI round 1 / round 2 vs OLGA (supplement)."""
import csv
import json
from pathlib import Path

import numpy as np

F = Path("/mnt/c/ReMIND_work/landmarks_final")
rows = list(csv.DictReader(open(F / "landmarks_audit.csv", encoding="utf-8"), delimiter=";"))
num = lambda s: float(s.replace(",", "."))
KEY = {"intra": ("SANTI_R1_vs_R2_mm", {"S1", "S2", "O"}), "inter_one": ("SANTI_R1_vs_OLGA_mm", {"S1", "O"}),
       "inter_two": ("SANTI_R2_vs_OLGA_mm", {"S2", "O"}),
       # OLGA vs the mean of SANTI's rounds: both rounds before the consensus, the rounds kept in the final set
       "inter_mean": ("SANTI_mean_vs_OLGA_mm", {"O"})}
FINAL_COL = {"inter_mean": "SANTI_kept_mean_vs_OLGA_mm"}
out = dict(dup_mm=3.0, outlier_mm=5.79)
for coh in ("ReMIND", "External", "Pooled"):
    R = rows if coh == "Pooled" else [r for r in rows if r["cohort"] == coh]
    cases = list(dict.fromkeys(r["case"] for r in R))
    kept = [sum(r["case"] == c and r["decision"] == "kept" for r in R) for c in cases]
    s = dict(cases=len(cases), annotated=len(R), merged=sum(r["decision"] == "merged" for r in R),
             excluded=sum(r["decision"] == "excluded" for r in R), kept=sum(kept), kept_min=min(kept), kept_max=max(kept))
    for name, (col, need) in KEY.items():
        for mode in ("final", "all"):
            cm = []
            for c in cases:
                if mode == "all":
                    v = [num(r[col]) for r in R if r["case"] == c and r["decision"] != "merged"]
                else:
                    fc = FINAL_COL.get(name, col)
                    v = [num(r[fc]) for r in R if r["case"] == c and r["decision"] == "kept"
                         and need <= set(r["picks_averaged"].split("+"))]
                if v:
                    cm.append(np.mean(v))
            s[f"{name}_{mode}_mean"] = float(np.mean(cm))
            s[f"{name}_{mode}_sd"] = float(np.std(cm, ddof=1))
    out[coh] = s
(F / "STATS.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
