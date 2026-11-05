"""Contact sheet of the axial iUS slices of a case (working grid, iUS alone in gray so that the resection cavity is
visible), with the landmark pairs of the proposed method, to choose by eye the slice that shows the cavity AND in which
the landmarks clearly improve.  Per slice: k, number of pairs within 6 mm and, over those pairs, the mean error for
affine / proposed / MCPO (from fer/tables/LANDMARKS.csv).  The automatic choice (choose_slice) is framed in red.
Usage: python _contact_sheet.py cohort:case [step] [kmin kmax]
Output: paper/build/contact_<case>.png"""
import csv
import sys

import numpy as np

import fig_lib as L
from fig_lib import plt, pe

coh, c = sys.argv[1].split(":")
step = int(sys.argv[2]) if len(sys.argv) > 2 else 4
d = L.load_case(coh, c)
k0, lm_v, how = L.choose_slice(d)
E = [r for r in csv.DictReader(open(L.W / "fer/tables/LANDMARKS.csv")) if r["cohort"] == coh and r["case"] == c]
err = {a: np.array([float(r[a]) for r in E]) for a in ("affine", "fer", "mcpo")}
assert len(E) == len(lm_v)
nz = [int(k) for k in np.unique(np.where(d["us"] > 0.02)[2])]
if len(sys.argv) > 4:
    nz = [k for k in nz if int(sys.argv[3]) <= k <= int(sys.argv[4])]
ks = [k for k in nz if (k - nz[0]) % step == 0]
if k0 in nz and k0 not in ks:
    ks = sorted(ks + [k0])
ncol = 5
nrow = int(np.ceil(len(ks) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.6, nrow * 2.6))
axes = np.atleast_2d(axes)
for i, ax in enumerate(axes.flat):
    ax.set_xticks([]); ax.set_yticks([])
    if i >= len(ks):
        ax.axis("off")
        continue
    k = ks[i]
    us_sl = d["us"][:, :, k]
    cone = us_sl > 0.02
    ax.imshow(L.norm01(us_sl, cone).T, cmap="gray", vmin=0, vmax=0.9, origin="lower", interpolation="nearest")
    near = np.abs(lm_v[:, 2] - k) < 6
    L.draw_pairs(ax, d, "fer", lm_v, near)
    x0, x1, y0, y1 = L.crop_limits(cone)
    if (coh, c) in L.ROT180:
        ax.set_xlim(x1, x0); ax.set_ylim(y1, y0)
    else:
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    lab = f"k={k} ({int(near.sum())})"
    if near.any():
        lab += f" a{err['affine'][near].mean():.1f} f{err['fer'][near].mean():.1f} m{err['mcpo'][near].mean():.1f}"
    t = ax.text(0.02, 0.03, lab, transform=ax.transAxes, color="w", fontsize=7.5, fontweight="bold")
    t.set_path_effects([pe.Stroke(linewidth=2, foreground="black"), pe.Normal()])
    if k == k0:
        for s in ax.spines.values():
            s.set_edgecolor("red"); s.set_linewidth(3)
fig.suptitle(f"{L.case_label(coh, c)}: automatic slice {k0} ({how}) in red; (n) pairs within 6 mm; a/f/m = mean error affine/proposed/MCPO", fontsize=9)
fig.tight_layout(pad=0.2)
out = L.W / f"paper/build/contact_{c}.png"
fig.savefig(out, dpi=100)
print(out, "slices", ks)
