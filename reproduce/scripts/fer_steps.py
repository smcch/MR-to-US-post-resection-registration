"""Internal breakdown of the two steps of the FE refinement on the challenge affine (NOT part of the manuscript unless
the author decides so): per case, the landmark error after
  modal     the modal fit alone (u = u_m)
  response  the local FE response alone, computed on the affine without the modal fit (u = r0)
  both      modal fit + local response (the proposed refinement, recomputed here; compared with the stored run)
Same frozen functions and constants as v2_remind_eval.deform / v2_method.ACCEPTED; models (basis30.npz) of fer_run.py.
Usage: python fer_steps.py run | report
Output: /mnt/c/ReMIND_work/fer/steps/<cohort>/<case>.json, fer/tables/STEPS.csv, fer/STEPS_REPORT.md
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa

OUT = C.FER / "steps"


def do_run():
    import nibabel as nib
    import torch
    import affine_cascade as AC
    import v2_deform as V
    import v2_method as FM
    from reg_core import load_to_grid, normalize, resample_to_grid
    from reg_modal import build_phi
    DEV = "cuda"
    cfg, rc = FM.ACCEPTED, FM.ACCEPTED["res"]
    tmp = Path("/tmp/fersteps")
    tmp.mkdir(exist_ok=True)

    def response(u_base, f05, mri_img, A, grid_aff, aff05, dims05, z, dims):
        mw05 = V.warp_mr_on_box(mri_img, A, u_base, grid_aff, aff05, dims05, DEV)
        mw05 = V.normalize_t(mw05, mw05 > 0)
        corr = V.residual_correspondences(f05, mw05, aff05, [tuple(l) for l in rc["levels"]], mind=tuple(rc["mind"]),
                                          coupled=rc["coupled"], weight=rc["weight"], dev=DEV, symmetric=rc["symmetric"],
                                          max_pts=rc["max_pts"])
        u_nodes, _ = V.fem_residual(z, corr, rc["lam_rel"], rc["huber_mm"], trim=rc["trim"])
        r = V.lattice_field(z, u_nodes, grid_aff, dims).to(DEV)
        n = r.norm(dim=1, keepdim=True)
        return r * (rc["max_mm"] / n.clamp(min=rc["max_mm"]))

    for coh in C.COHORTS:
        for c in C.cases(coh):
            f = OUT / coh / f"{c}.json"
            if f.exists():
                continue
            try:
                us, mr, _, A, _ = C.inputs(coh, c)
                basis = C.RUN / coh / c / "basis30.npz"
                AC.relabel(mr, tmp / "moving_rig.nii.gz", A)
                f_arr, m_arr, grid_aff, dims = load_to_grid(us, tmp / "moving_rig.nii.gz", 1.0)
                fov = f_arr > 0
                f_n, m_n = normalize(f_arr, fov), normalize(m_arr, m_arr > 0)
                phi, _ = build_phi(basis, grid_aff, dims, 1.0)
                eig = np.load(str(basis))["eigvals"]
                phi = phi.to(DEV)
                _, u = V.register_modal_from(f_n, m_n, fov, phi, eig, None, cfg["modal_iters"], 3e-3, 0.1, tuple(cfg["mind"]), DEV)
                usi, mri_img = nib.load(str(us)), nib.load(str(mr))
                aff05, dims05 = V.box05(usi)
                f05 = resample_to_grid(usi, aff05, dims05).astype(np.float32)
                f05 = torch.from_numpy(normalize(f05, f05 > 0))[None, None].to(DEV)
                z = np.load(str(basis), allow_pickle=True)
                r_both = response(u, f05, mri_img, A, grid_aff, aff05, dims05, z, dims)
                r_only = response(torch.zeros_like(u), f05, mri_img, A, grid_aff, aff05, dims05, z, dims)
                mri, usl, ids = C.landmarks(coh, c)
                res = dict(cohort=coh, case=c)
                for name, field in (("modal", u), ("response", r_only), ("both", u + r_both)):
                    img = nib.Nifti1Image(np.moveaxis(field[0].cpu().numpy(), 0, -1).astype(np.float32), grid_aff)
                    err = np.linalg.norm(C.apply(A, usl + C.disp_at(img, usl)) - mri, axis=1)
                    fold, sd = V.regularity(field, fov)
                    mag = field[0].norm(dim=0).cpu().numpy()[fov]
                    res[name] = float(err.mean())
                    res[f"{name}_sdlogj"], res[f"{name}_fold"] = sd, fold
                    res[f"{name}_disp_mean"] = float(mag.mean())
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(res))
                print(f"[steps] {coh} {c}: modal {res['modal']:.2f} | response alone {res['response']:.2f} | both {res['both']:.2f}", flush=True)
                del phi, u, r_both, r_only, f05
                torch.cuda.empty_cache()
            except Exception as e:                                    # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[steps] {coh} {c} FAILED: {e}", flush=True)
    print("STEPS RUN DONE", flush=True)


def do_report():
    from scipy import stats
    per = {(r["cohort"], r["case"]): r for r in csv.DictReader(open(C.FER / "tables/PER_CASE.csv"))}
    rows = []
    for f in sorted(OUT.glob("*/*.json")):
        j = json.loads(f.read_text())
        if (j["cohort"], j["case"]) not in per:                      # case excluded from the analysis
            continue
        p = per[(j["cohort"], j["case"])]
        rows.append(dict(j, affine=float(p["affine"]), fer_stored=float(p["fer"])))
    with open(C.FER / "tables/STEPS.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    L = ["# Steps of the FE refinement on the challenge affine (internal; not in the manuscript)", "",
         "Mean TRE (mm). modal = modal fit alone; response = local FE response alone (no modal fit); both = the refinement.", "",
         "| set | n | affine | modal alone | response alone | both (recomputed) | both (stored run) |", "|---|---|---|---|---|---|---|"]
    groups = [(C.LABEL[c], [r for r in rows if r["cohort"] == c]) for c in C.COHORTS] + [("all", rows)]
    for lab, rr in groups:
        if rr:
            L.append(f"| {lab} | {len(rr)} | " + " | ".join(f"{np.mean([r[k] for r in rr]):.3f}" for k in ("affine", "modal", "response", "both", "fer_stored")) + " |")
    L.append("")
    for a, b in (("modal", "affine"), ("response", "affine"), ("both", "modal"), ("both", "response"), ("both", "affine")):
        x, y = np.array([r[a] for r in rows]), np.array([r[b] for r in rows])
        L.append(f"- {a} vs {b} (all {len(rows)}): {x.mean():.3f} vs {y.mean():.3f}; {a} better in {int((x < y).sum())}; "
                 f"Wilcoxon p = {stats.wilcoxon(x, y, zero_method='zsplit').pvalue:.4f}")
    L.append("")
    L.append("Regularity (mean SDLogJ): " + ", ".join(f"{k} {np.mean([r[k + '_sdlogj'] for r in rows]):.3f}" for k in ("modal", "response", "both"))
             + "; mean |u| over the FoV (mm): " + ", ".join(f"{k} {np.mean([r[k + '_disp_mean'] for r in rows]):.2f}" for k in ("modal", "response", "both")))
    L.append(f"Recomputed vs stored refinement: max |diff| of case means {max(abs(r['both'] - r['fer_stored']) for r in rows):.3f} mm.")
    (C.FER / "STEPS_REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"run": do_run, "report": do_report}[sys.argv[1]]()
