"""Rank cases by how clearly the proposed method (affine + FER) beats BOTH its affine and MCPO: min(gain over affine,
margin over MCPO), per cohort."""
import csv

rows = list(csv.DictReader(open("/mnt/c/ReMIND_work/fer/tables/PER_CASE.csv")))
for coh in ("resect_after", "bite_g4", "remind", "external"):
    rr = [r for r in rows if r["cohort"] == coh and r["mcpo"] not in ("", "nan")]
    out = []
    for r in rr:
        a, f, m, p = (float(r[k]) for k in ("affine", "fer", "mcpo", "mcpo_pose_fer"))
        out.append((min(a - f, m - f), r["case"], a, f, m, p))
    out.sort(reverse=True)
    print(f"== {coh}")
    for s, c, a, f, m, p in out[:7]:
        print(f"  {c:12s} min margin {s:5.2f} | affine {a:5.2f} -> FER {f:5.2f} | MCPO {m:5.2f} | pose+FER {p:5.2f}")
