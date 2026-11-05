"""Fig. 2 of the FE-refinement manuscript: from the preoperative MR and its SynthSeg segmentation to the patient-specific
finite-element model, on one real case, axial section of the whole model in the frame of the initialisation.
(a) preoperative MR after the initialisation   (b) SynthSeg labels      (c) the two compartments, lattice, surface nodes
(d) node-averaged Young's modulus              (e) eigenmode 1          (f) eigenmode 6
Usage: python fig_model.py cohort:case [--stem fig2_model]
"""
import argparse

import fig_method_lib as M
from fig_method_lib import plt, L

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--stem", default="fig2_model")
a = ap.parse_args()
coh, c = a.case.split(":")
P = M.prepare(coh, c)

fig, axes = plt.subplots(2, 3, figsize=(170 * L.MM, 112 * L.MM))
M.panel_mr(axes[0, 0], P, "(a) preoperative MR in the initialized frame")
M.panel_synthseg(axes[0, 1], P, "(b) SynthSeg segmentation")
M.panel_compartments(axes[0, 2], P, "(c) compartments, lattice, surface nodes")
M.panel_stiffness(axes[1, 0], P, "(d) Young's modulus per node (kPa)")
M.panel_mode(axes[1, 1], P, 0, "(e) eigenmode 1, $|\\varphi_1|$")
M.panel_mode(axes[1, 2], P, 5, "(f) eigenmode 6, $|\\varphi_6|$")
fig.tight_layout(pad=0.35)
fig.savefig(L.OUT / f"{a.stem}.pdf", dpi=300)
fig.savefig(L.OUT / f"{a.stem}.png", dpi=300)
print(f"{a.stem} ok: {L.case_label(coh, c)}, slice {P['k']} ({P['how']}); nodes {len(P['nodes'])}, tets {len(P['tets'])}, "
      f"surface nodes {len(P['surface'])}, h {P['h_mm']} mm")
