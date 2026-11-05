"""FROZEN v2 method (round 3, 2026-09-16) — candidate to the shipped method, NOT official.

Decision (author, 2026-09-16): the shipped method (single_config m30r, method_arms.py) stays the
official method of the manuscript; this configuration is kept frozen and will be re-evaluated on
the ReMIND landmark cases when they exist (scripts/v2_remind_eval.py).  The final choice between
the two is deferred to that evaluation.

The configurations below are LITERALS (not derived from v2_arms.py) so that later edits of the
development code cannot change them.  They correspond to v2_arms.arm("R3_sym") (accepted poses)
and v2_arms.arm("I4_fem") (fallback of the abstentions) at freeze time; frozen copies of the
implementing scripts and their md5 sums are in results/v2/frozen_scripts/.

Pipeline (pose stage unchanged from the shipped method):
  search (FFT, +-25 mm / +-10 deg, K12) -> guarded 0.5 mm polish -> gate at 9.5 deg
  accepted  : 30 FEM modes fitted by gradient on MIND-SSC (r2/d2, 1 mm grid, lambda_eig 3e-3,
              200 it)  ->  local FEM response (Clatz-style) to SYMMETRIC block correspondences
              (0.5 mm iUS box, pyramid (4,2)(2,1), MIND r3/d3, coupled convex, margin weights,
              forward/backward averaged), robust ridge on the case's stiffness K (Huber 1 mm,
              lambda_rel 10 of the covered-DoF data strength), interpolated with the basis
              lattice, bound 2 mm
  abstained : identity pose -> the same modal stage -> the same FEM response with one-way
              correspondences (the symmetric variant was below the adoption rule there)
"""
VERSION = "v2-r3-2026-09-16"

ACCEPTED = {
    "name": "R3_sym", "pose": "shipped", "mind": (2, 2), "modal": "mind", "modal_iters": 200, "residual": "fem",
    "res": {"levels": [(4, 2), (2, 1)], "mind": (3, 3), "coupled": True, "weight": "margin", "symmetric": True,
            "lam_rel": 10.0, "huber_mm": 1.0, "trim": 0.0, "max_pts": 300000, "max_mm": 2.0, "smooth_mm": 1.5},
}
FALLBACK = {
    "name": "I4_fem", "pose": "identity", "mind": (2, 2), "modal": "mind", "modal_iters": 200, "residual": "fem",
    "res": {"levels": [(4, 2), (2, 1)], "mind": (3, 3), "coupled": True, "weight": "margin", "symmetric": False,
            "lam_rel": 10.0, "huber_mm": 1.0, "trim": 0.0, "max_pts": 300000, "max_mm": 2.0, "smooth_mm": 1.5},
}
GATE_DEG = 9.5

# development-set results at freeze time (results/v2/PIPELINE.md, 55 pairs, shipped pose):
#   RESECT post 2.481 (shipped 2.560) | BITE 2.791 (3.025) | RESECT pre 1.920 (2.050)
#   pooled post-resection 2.603 vs 2.743 (p < 0.001, 25/33); folding 0.000 %; SDLogJ 0.09
# caveat: BITE was the development set and RESECT entered the round-3 selection -> the only
# untouched evaluation is the future ReMIND landmark set (first case: ReMIND-023, v2 2.01 vs
# shipped 2.20, NiftyReg 1.87, MCPO 2.16; results/v2/remind023/TRE.md)


def describe():
    import json
    return json.dumps(dict(version=VERSION, gate_deg=GATE_DEG, accepted=ACCEPTED, fallback=FALLBACK), indent=1, default=list)


if __name__ == "__main__":
    print(describe())
