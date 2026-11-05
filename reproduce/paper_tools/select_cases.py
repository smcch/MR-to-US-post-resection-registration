"""Objective choice of the cases shown in the image figures (rules stated in the captions).
  Fig. 4 (qualitative), Fig. 5 (fields), Fig. 6 (external): since 2026-10-03 the AUTHOR chooses illustrative cases among
                        those in which the proposed method beats both its affine and MCPO (Case15, 03, 024, 021; external
                        S4, S1; see _rank_wins.py), and since 2026-10-05 also the slice (cavity visible, landmarks clearly
                        better: _slice_search.py, _contact_sheet.py; slices in _make_figs.sh). The rules below are kept
                        for reference.
  Fig. 4-old (qualitative)  largest gain of the refinement over its affine in RESECT post-resection, in BITE, and the two
                        largest in ReMIND
  Fig. 3 (fields)       the most typical registration for regularity (see median_mcpo) in RESECT post-resection and ReMIND
  Fig. 1 (method)       the ReMIND case with the largest gain
"""
import csv
from pathlib import Path

rows = list(csv.DictReader(open("/mnt/c/ReMIND_work/fer/tables/PER_CASE.csv")))


def gain(r):
    return float(r["affine"]) - float(r["fer"])


def top(coh, n):
    rr = sorted((r for r in rows if r["cohort"] == coh), key=gain, reverse=True)
    return [(r["case"], round(gain(r), 2)) for r in rr[:n]]


def median_mcpo(coh):
    """The most typical registration of the cohort for regularity: smallest sum, over the four deformable methods, of the
    distance between the case's SDLogJ rank and the median rank (no method is shown on its best or worst case)."""
    keys = ["f3d_mind_sdlogj", "f3d_lncc_sdlogj", "mcpo_sdlogj", "fer_sdlogj"]
    rr = [r for r in rows if r["cohort"] == coh and all(r.get(k) not in ("", None) for k in keys)]
    score = {r["case"]: 0.0 for r in rr}
    for k in keys:
        order = sorted(rr, key=lambda r: float(r[k]))
        for rank, r in enumerate(order):
            score[r["case"]] += abs(rank - (len(rr) - 1) / 2)
    best = min(rr, key=lambda r: score[r["case"]])
    return best["case"], {k.replace("_sdlogj", ""): round(float(best[k]), 3) for k in keys}, score[best["case"]]


q = [("resect_after", top("resect_after", 3)), ("bite_g4", top("bite_g4", 3)), ("remind", top("remind", 4))]
print("largest gains:", q)
f = [("resect_after", median_mcpo("resect_after")), ("remind", median_mcpo("remind"))]
print("most typical case for regularity:", f)
qual = [f"resect_after:{q[0][1][0][0]}", f"bite_g4:{q[1][1][0][0]}", f"remind:{q[2][1][0][0]}", f"remind:{q[2][1][1][0]}"]
fields = [f"{c}:{v[0]}" for c, v in f]
Path("/mnt/c/ReMIND_work/paper/build/figure_cases.txt").write_text(
    "QUAL=" + " ".join(qual) + "\nFIELDS=" + " ".join(fields) + "\nMETHOD=" + f"remind:{q[2][1][0][0]}" + "\n")
print(Path("/mnt/c/ReMIND_work/paper/build/figure_cases.txt").read_text())
