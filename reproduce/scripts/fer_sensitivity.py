"""Sensitivity of the FE refinement to its three continuous constants, on a DEVELOPMENT cohort only (BITE, 13 cases):
one constant at a time around the frozen configuration; nothing is re-selected (supplementary table).
  lambda_eig   1e-3 | [3e-3] | 1e-2      weight of the modal energy prior
  lambda_rel   3    | [10]   | 30        relative weight of the stiffness in the local response
  bound        1    | [2]    | 3   mm    clip of the local response
The model (basis30.npz) of the frozen run is reused; ReMIND and the external site are not touched.
Usage: python fer_sensitivity.py run | score
Output: /mnt/c/ReMIND_work/fer/sens/<variant>/bite_g4/<case>/fer/, fer/tables/SENSITIVITY.csv
"""
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa

COH = "bite_g4"
VARIANTS = {"lam_eig_1e-3": dict(lam_eig=1e-3), "lam_eig_1e-2": dict(lam_eig=1e-2),
            "lam_rel_3": dict(lam_rel=3.0), "lam_rel_30": dict(lam_rel=30.0),
            "bound_1mm": dict(max_mm=1.0), "bound_3mm": dict(max_mm=3.0)}
OUT = C.FER / "sens"


def do_run():
    import affine_cascade as AC
    import v2_deform as V
    import v2_method as FM
    import v2_remind_eval as VE
    VE.relabel = AC.relabel
    orig_modal = V.register_modal_from
    for name, ch in VARIANTS.items():
        cfg = copy.deepcopy(FM.ACCEPTED)
        if "lam_rel" in ch:
            cfg["res"]["lam_rel"] = ch["lam_rel"]
        if "max_mm" in ch:
            cfg["res"]["max_mm"] = ch["max_mm"]
        lam_eig = ch.get("lam_eig", 3e-3)

        def modal(*a, _lam=lam_eig, **k):
            a = list(a)
            assert abs(a[7] - 3e-3) < 1e-12, "unexpected call signature"
            a[7] = _lam
            return orig_modal(*a, **k)
        V.register_modal_from = modal
        for c in C.cases(COH):
            us, mr, _, A, _ = C.inputs(COH, c)
            VE.paths = lambda case, us=us, mr=mr: (us, mr)
            VE.OUT = OUT / name / COH
            VE.deform(c, cfg, "fer", A, C.RUN / COH / c / "basis30.npz")
        print(f"[sens] {name} done", flush=True)
    V.register_modal_from = orig_modal
    print("SENS RUN DONE", flush=True)


def do_score():
    rows = []
    ref = {}
    for c in C.cases(COH):
        mri, us, _ = C.landmarks(COH, c)
        A = C.inputs(COH, c)[3]
        e = lambda p: float(np.linalg.norm(C.apply(A, us + C.disp_at(p, us)) - mri, axis=1).mean())   # noqa: E731
        ref[c] = e(C.RUN / COH / c / "fer/disp_mm.nii.gz")
        m = json.loads((C.RUN / COH / c / "fer/meta.json").read_text())
        rows.append(dict(variant="frozen", case=c, tre=ref[c], fold_pct=m["fold_pct"], sdlogj=m["sdlogj"]))
        for name in VARIANTS:
            d = OUT / name / COH / c / "fer"
            m = json.loads((d / "meta.json").read_text())
            rows.append(dict(variant=name, case=c, tre=e(d / "disp_mm.nii.gz"), fold_pct=m["fold_pct"], sdlogj=m["sdlogj"]))
    (C.FER / "tables").mkdir(parents=True, exist_ok=True)
    with open(C.FER / "tables/SENSITIVITY.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for name in ["frozen"] + list(VARIANTS):
        v = np.array([r["tre"] for r in rows if r["variant"] == name])
        r0 = np.array([ref[r["case"]] for r in rows if r["variant"] == name])
        print(f"{name:14s} TRE {v.mean():.3f}  diff vs frozen {np.mean(v - r0):+.3f}  max |diff| {np.abs(v - r0).max():.2f}  "
              f"SDLogJ {np.mean([r['sdlogj'] for r in rows if r['variant'] == name]):.3f}  "
              f"fold max {max(r['fold_pct'] for r in rows if r['variant'] == name):.3f}")


if __name__ == "__main__":
    {"run": do_run, "score": do_score}[sys.argv[1]]()
