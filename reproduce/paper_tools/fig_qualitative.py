"""Fig. 4 of the FE-refinement manuscript: fused iUS on warped MR with landmark pairs, one row per case.
Columns (author's choice, 2026-10-03: only the baseline, the proposed method and MCPO appear in the figures):
challenge affine (baseline) | affine + FE refinement (proposed) | MCPO (complete).
Usage: python fig_qualitative.py cohort:case[:slice] [...] [--stem fig4_qualitative]   (slice: author's choice, else automatic)
Needs MCPO's field for public cases (scripts/fer_keep_mcpo_fields.py cohort/case).
"""
import argparse
import csv

import numpy as np

import fig_lib as L
from fig_lib import plt, pe

ARMS = [("affine", "NiftyReg affine (baseline)"), ("fer", "Affine + FE refinement (proposed)"), ("mcpo", "MCPO")]
KEY = {"initial": "initial", "affine": "affine", "f3d": "f3d_mind", "fer": "fer", "mcpo": "mcpo", "mcpo_pose": "mcpo_pose",
       "mcpo_pose_fer": "mcpo_pose_fer"}

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--stem", default="fig4_qualitative")
a = ap.parse_args()
T = {(r["cohort"], r["case"]): r for r in csv.DictReader(open(L.W / "fer/tables/PER_CASE.csv"))}

items = L.parse_items(a.cases)
fig, axes = plt.subplots(len(items), len(ARMS), figsize=(170 * L.MM, 55 * len(items) * L.MM + 6 * L.MM), squeeze=False)
for r, (coh, c, kf) in enumerate(items):
    d = L.load_case(coh, c)
    assert d["u_mcpo"] is not None, (coh, c, "MCPO field missing (scripts/fer_keep_mcpo_fields.py)")
    k, lm_v, how = L.choose_slice(d, k_fixed=kf)
    xw = L.world_grid(d["aff"], d["us"].shape, k)
    us_sl = d["us"][:, :, k]
    cone = us_sl > 0.02
    x0, x1, y0, y1 = L.crop_limits(cone)
    near = np.abs(lm_v[:, 2] - k) < 6
    row = T[(coh, c)]
    print(f"[{a.stem}] {L.case_label(coh, c)}: slice {k} ({how}), pairs within 6 mm: {int(near.sum())}; "
          + ", ".join(f"{arm} {float(row[KEY[arm]]):.2f}" for arm, _ in ARMS), flush=True)
    for cc, (arm, title) in enumerate(ARMS):
        ax = axes[r, cc]
        L.fuse(ax, L.norm01(L.warped_mr(d, arm, xw)), us_sl, cone)
        L.draw_pairs(ax, d, arm, lm_v, near)
        if (coh, c) in L.ROT180:
            ax.set_xlim(x1, x0); ax.set_ylim(y1, y0)
        else:
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(title, fontsize=9)
        if cc == 0:
            ax.set_ylabel(L.case_label(coh, c), fontsize=9)
        t = ax.text(0.03, 0.04, f"{float(row[KEY[arm]]):.1f} mm", transform=ax.transAxes, ha="left", va="bottom", color="w", fontsize=9,
                    fontweight="bold")
        t.set_path_effects([pe.Stroke(linewidth=2, foreground="black"), pe.Normal()])
        if cc == 0 and r == 0:
            for txt, xy, ha, va in (("A", (0.5, 0.97), "center", "top"), ("R", (0.97, 0.5), "right", "center")):
                tt = ax.text(*xy, txt, transform=ax.transAxes, ha=ha, va=va, color="w", fontsize=8, fontweight="bold")
                tt.set_path_effects([pe.Stroke(linewidth=2, foreground="black"), pe.Normal()])
fig.tight_layout(pad=0.3)
L.OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(L.OUT / f"{a.stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)   # one-row figures clipped titles
fig.savefig(L.OUT / f"{a.stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print(a.stem, "ok")
