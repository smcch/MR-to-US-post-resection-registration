"""Fig. 1 of the FE-refinement manuscript: the pipeline as a schematic with small captures of one real case.
Stages: inputs (preoperative MR + SynthSeg, iUS sweep) -> initialization A -> patient-specific FE model -> modal fit ->
block correspondences -> local FE response -> output.  Captures from fig_method_lib (cached fields of the case).
Usage: python fig_pipeline.py cohort:case [--stem fig1_pipeline]
"""
import argparse

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import fig_method_lib as M
from fig_method_lib import plt, L

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--stem", default="fig1_pipeline")
a = ap.parse_args()
coh, c = a.case.split(":")
P = M.prepare(coh, c)

W, H = 190.0, 108.0                                                   # mm
fig = plt.figure(figsize=(W * L.MM, H * L.MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
INK, BOX, EDGE = "#222222", "#f3f3f3", "#888888"
FS_T, FS_B, FS_L = 7.5, 6.3, 6.0                                      # box title, box body, capture label


def thumb(x, y, w, h):
    return fig.add_axes([x / W, y / H, w / W, h / H])


def box(x0, x1, y0, y1, title, body, fill=BOX):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.4", fc=fill, ec=EDGE, lw=0.7))
    ax.text((x0 + x1) / 2, y1 - 1.3, title, ha="center", va="top", fontsize=FS_T, fontweight="bold", color=INK)
    ax.text((x0 + x1) / 2, y0 + 1.1, body, ha="center", va="bottom", fontsize=FS_B, color=INK, linespacing=1.12)


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=0.9, color=INK))


def label(x, y, s):
    ax.text(x, y, s, ha="center", va="bottom", fontsize=FS_L, color=INK)


# ------------------------------------------------------------------ row 1: inputs -> initialization -> FE model
TY, TH = 74.0, 30.0                                                   # captures of row 1
BY0, BY1 = 57.0, 71.0                                                 # boxes of row 1
t = thumb(5, TY + 5, 20, 20); M.panel_mr(t, P)
t = thumb(27, TY + 5, 20, 20); M.panel_ius(t, P)
label(15, TY + 25.6, "preoperative MR"); label(37, TY + 25.6, "iUS sweep")
box(4, 48, BY0, BY1, "Inputs", "preoperative MR and its SynthSeg\nsegmentation; iUS sweep in the\nnavigation frame")
t = thumb(71, TY, TH, TH); M.panel_fused(t, P, "affine")
label(86, TY + TH + 0.6, "iUS on the MR after $A$")
box(58, 114, BY0, BY1, "Initialization $A$", "NiftyReg affine (challenge baseline)\nor the pose of another method;\nMR and segmentation placed in the frame of $A$")
t = thumb(127, TY + 2, 26, 26); M.panel_stiffness(t, P, bar=False)
t = thumb(157, TY + 2, 26, 26); M.panel_mode(t, P, 0)
label(140, TY + 28.6, "stiffness"); label(170, TY + 28.6, "eigenmode 1")
box(124, 186, BY0, BY1, "Patient-specific FE model", "5 mm tetrahedral lattice, two compartments\n(parenchyma 3 kPa, CSF 0.1 kPa), foundation;\n$K$, $M$ and the 30 lowest-energy modes")
arrow((48, 64), (58, 64)); arrow((114, 64), (124, 64))

# ------------------------------------------------------------------ row 2: modal fit -> correspondences -> FE response -> output
TY2, BY20, BY21 = 16.0, 1.0, 14.0
t = thumb(9, TY2, TH, TH); M.panel_modal(t, P, bar=False)
label(24, TY2 + TH + 0.6, "$|u_{\\mathrm{m}}|$ and deformed grid")
box(4, 44, BY20, BY21, "1. Modal fit", "30 mode coefficients $c_i$ fitted to\nthe iUS: MIND-SSC similarity\n+ elastic energy")
t = thumb(59, TY2, TH, TH); M.panel_corr(t, P, bar=False)
label(74, TY2 + TH + 0.6, "block correspondences, confidence")
box(50, 98, BY20, BY21, "2. Block correspondences", "MIND blocks, two levels, both directions;\ndisplacement $d_j$ and confidence $w_j$\nper block")
t = thumb(113, TY2, TH, TH); M.panel_response(t, P, bar=False)
label(128, TY2 + TH + 0.6, "$|r|$ and deformed grid")
box(104, 152, BY20, BY21, "3. Local FE response", "nodal displacements through $K$:\nHuber weights, $\\lambda_{\\mathrm{rel}}$ to the data;\n$|r| \\leq 2$ mm")
t = thumb(157, TY2, TH, TH); M.panel_fused(t, P, "fer")
label(172, TY2 + TH + 0.6, "iUS on the refined MR")
box(158, 186, BY20, BY21, "Output", "$x_{\\mathrm{MR}} = A\\,(x + u_{\\mathrm{m}} + r)$\nregular field")
arrow((44, 7.5), (50, 7.5)); arrow((98, 7.5), (104, 7.5)); arrow((152, 7.5), (158, 7.5))
# the model feeds both steps of the refinement: elbow from the FE-model box down to row 2
ax.plot([155, 155, 24, 24], [BY0, 52.5, 52.5, 50.0], color=INK, lw=0.9, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((24, 52.5), (24, 49.6), arrowstyle="-|>", mutation_scale=9, lw=0.9, color=INK))
ax.text(90, 53.4, "FE refinement $u = u_{\\mathrm{m}} + r$: one frozen configuration for every case; no landmark is used",
        ha="center", va="bottom", fontsize=6.4, color=INK, style="italic")
fig.savefig(L.OUT / f"{a.stem}.pdf", dpi=300)
fig.savefig(L.OUT / f"{a.stem}.png", dpi=300)
print(f"{a.stem} ok: {L.case_label(coh, c)}, slice {P['k']}")
