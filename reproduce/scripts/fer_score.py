"""Score every arm of the 2026-10 manuscript version at the landmarks (the only script of the FER family that
combines registration outputs with landmark files).

Arms
  initial        navigation alignment
  affine         NiftyReg challenge baseline (reg_aladin rigid + affine), stored transforms
  f3d_mind       reg_f3d MIND-SSC on that affine                     (fer_f3d.py, per landmark)
  f3d_lncc, f3d_mind_lncc   the LNCC variants of the free-form stage (case means; fer_f3d.py where run here, otherwise
                 the identical, deterministic cells of contrib_factorial[_remind].py)
  fer            affine + finite-element refinement = the proposed method (fer_run.py)
  mcpo           MCPO, official script, defaults, from navigation      (fer_mcpo.py)
  mcpo_pose      rigid pose fitted to MCPO's field (second initialisation)  (fer_mcpo_pose.py)
  mcpo_pose_fer  that pose + the same finite-element refinement            (fer_mcpo_pose.py)
  bestfit        6-DoF rigid fitted to the evaluation landmarks (in-sample bound); bestfit_loo leave-one-out
pred = A (x_us + u(x_us)) for fer; A x_us for affine; error = |pred - x_mr|.

Output: /mnt/c/ReMIND_work/fer/tables/{LANDMARKS.csv,PER_CASE.csv}
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa

TAB = C.FER / "tables"
ARMS = ["initial", "affine", "f3d_mind", "convexadam", "deeds", "syn", "fer", "mcpo", "mcpo_pose", "mcpo_pose_fer", "bestfit", "bestfit_loo"]
# other deformables on the SAME challenge affine (fer_extra_deformables.py, 2026-10-05; fair protocol: whole MR cut to
# the iUS box, field of view by each method's own mask mechanism; deeds has none -> MR zeroed outside the dilated cone)
EXTRA = {"convexadam": C.FER / "extra_deformables_box/convexadam_mask",   # ConvexAdam, use_mask (cone / dilated cone)
         "deeds": C.FER / "extra_deformables/deeds",                     # deedsBCV defaults, MR cropped to the cone
         "syn": C.FER / "extra_deformables_box/syn"}                     # ANTs SyNOnly, Mattes MI, fixed mask = cone


def cell_means():
    """f3d LNCC variants on the challenge affine computed earlier with the identical recipe (deterministic)."""
    out = {}
    p = C.W / "results/v2/contrib/B/CELLS.csv"
    for r in csv.DictReader(open(p)):
        if r["pose"] == "nr" and r["deform"] in ("f3d_mind", "f3d_lncc", "f3d_mind_lncc"):
            out[(r["deform"], r["cohort"], r["case"])] = dict(tre=float(r["tre"]), fold_pct=float(r["fold"]), sdlogj=float(r["sdlogj"]))
    for kind in ("f3d_mind", "f3d_lncc", "f3d_mind_lncc"):
        for f in (C.W / f"remind/B/nr/{kind}").glob("ReMIND-*.json"):
            if "consensus" in f.name:
                continue
            j = json.loads(f.read_text())
            out[(kind, "remind", f.stem)] = dict(tre=j["tre"], fold_pct=j["fold"], sdlogj=j["sdlogj"])
    return out


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    cells = cell_means()
    lm_rows, case_rows = [], []
    for coh in C.COHORTS:
        for c in C.cases(coh):
            d = C.RUN / coh / c
            if not (d / "fer/disp_mm.nii.gz").exists():
                print(f"[score] {coh} {c}: proposed method not run yet")
                continue
            mri, us, ids = C.landmarks(coh, c)
            A = C.inputs(coh, c)[3]
            err = {"initial": np.linalg.norm(us - mri, axis=1),
                   "affine": np.linalg.norm(C.apply(A, us) - mri, axis=1),
                   "fer": np.linalg.norm(C.apply(A, us + C.disp_at(d / "fer/disp_mm.nii.gz", us)) - mri, axis=1)}
            R, t = C.horn(us, mri)
            err["bestfit"] = np.linalg.norm(us @ R.T + t - mri, axis=1)
            loo = []
            for k in range(len(us)):
                m = np.ones(len(us), bool); m[k] = False
                Rk, tk = C.horn(us[m], mri[m])
                loo.append(np.linalg.norm(us[k] @ Rk.T + tk - mri[k]))
            err["bestfit_loo"] = np.array(loo)
            extra = {}
            f = C.FER / f"f3d/f3d_mind/{coh}/{c}.json"
            if f.exists():
                j = json.loads(f.read_text())
                assert j["ids"] == [int(i) for i in ids], (coh, c)
                err["f3d_mind"] = np.array(j["err"])
                extra.update(f3d_mind_fold=j["fold_pct"], f3d_mind_sdlogj=j["sdlogj"], f3d_mind_wall_s=j["wall_s"])
                k = ("f3d_mind", coh, c)
                if k in cells and abs(cells[k]["tre"] - j["tre"]) > 0.005:
                    print(f"[score] WARNING {coh} {c}: f3d MIND rerun {j['tre']:.3f} != stored cell {cells[k]['tre']:.3f}")
            for arm, xd in EXTRA.items():
                f = xd / coh / f"{c}.json"
                if f.exists():
                    j = json.loads(f.read_text())
                    assert j["ids"] == [int(i) for i in ids], (coh, c, arm)
                    err[arm] = np.array(j["err"])
                    extra.update({f"{arm}_fold": j["fold_pct"], f"{arm}_sdlogj": j["sdlogj"], f"{arm}_wall_s": j["wall_s"]})
            f = C.FER / f"mcpo/{coh}/{c}.json"
            if f.exists():
                j = json.loads(f.read_text())
                assert j["ids"] == [int(i) for i in ids], (coh, c)
                err["mcpo"] = np.array(j["err"])
                extra.update(mcpo_fold=j["fold_pct"], mcpo_sdlogj=j["sdlogj"], mcpo_fold_05=j["fold_pct_05"], mcpo_sdlogj_05=j["sdlogj_05"],
                             mcpo_wall_s=j.get("wall_s", np.nan), mcpo_moving=j.get("moving", ""),
                             mcpo_own_box=int("own 0.5 mm box" in j.get("source", "")))   # ReMIND-009: not in the release
            f = C.FER / f"mcpo_t2/{coh}/{c}.json"
            if f.exists():
                extra["mcpo_t2"] = json.loads(f.read_text())["tre"]
            pz, dz = C.FER / f"mcpo_pose/{coh}/{c}/pose.txt", C.FER / f"run_mcpopose/{coh}/{c}"
            if pz.exists() and (dz / "fer/disp_mm.nii.gz").exists() and (dz / "timing.json").exists():
                P = np.loadtxt(pz)
                err["mcpo_pose"] = np.linalg.norm(C.apply(P, us) - mri, axis=1)
                err["mcpo_pose_fer"] = np.linalg.norm(C.apply(P, us + C.disp_at(dz / "fer/disp_mm.nii.gz", us)) - mri, axis=1)
                tz = json.loads((dz / "timing.json").read_text())
                fz = json.loads((C.FER / f"mcpo_pose/{coh}/{c}/fit.json").read_text())
                extra.update(mcpo_pose_fer_fold=tz["fold_pct"], mcpo_pose_fer_sdlogj=tz["sdlogj"], mcpo_pose_fer_deform_s=tz["deform_total_s"],
                             mcpo_pose_fer_basis_s=tz["basis_s"], mcpo_pose_rot_deg=fz.get("rotation_deg", np.nan),
                             mcpo_pose_fit_mm=fz.get("fit_residual_mm", np.nan))
            for kind in ("f3d_lncc", "f3d_mind_lncc"):
                f = C.FER / f"f3d/{kind}/{coh}/{c}.json"
                j = json.loads(f.read_text()) if f.exists() else cells.get((kind, coh, c))
                if j:
                    extra.update({kind: j["tre"], f"{kind}_fold": j["fold_pct"], f"{kind}_sdlogj": j["sdlogj"]})
            tm = json.loads((d / "timing.json").read_text())
            # size of the refinement over the iUS field of view, share of voxels at the 2 mm bound, model size
            import nibabel as nib
            ui = nib.load(str(d / "fer/disp_mm.nii.gz"))
            mag = np.linalg.norm(np.asarray(ui.dataobj, np.float32), axis=-1)
            fov = np.asarray(nib.load(str(C.inputs(coh, c)[4])).dataobj) > 0
            assert fov.shape == mag.shape, (coh, c, fov.shape, mag.shape)
            meta = json.loads((d / "fer/meta.json").read_text())
            with np.load(d / "basis30.npz", allow_pickle=True) as z:
                n_nodes, n_tets = int(z["nodes"].shape[0]), int(z["tets"].shape[0])
            extra.update(fer_disp_mean=float(mag[fov].mean()), fer_disp_p95=float(np.percentile(mag[fov], 95)),
                         fer_disp_max=float(mag[fov].max()), fer_clip_frac=float(meta["residual"]["clipped_frac"]),
                         corr_n=int(meta["residual"]["n"]), corr_inl=int(meta["residual"]["n_inl"]),
                         model_nodes=n_nodes, model_tets=n_tets, fov_ml=float(fov.sum() / 1000.0))
            hard = np.argsort(-err["initial"])[:max(1, int(round(0.3 * len(us))))]
            row = dict(cohort=coh, case=c, n=len(us))
            for a in ARMS:
                if a in err:
                    e = err[a]
                    row[a] = float(e.mean())
                    row[f"{a}_median"] = float(np.median(e))
                    row[f"{a}_max"] = float(e.max())
                    row[f"{a}_tre30"] = float(np.percentile(e, 70))         # the paper's TRE30: 70th percentile per case
                    row[f"{a}_hard30"] = float(e[hard].mean())              # mean over the 30 % hardest landmarks (initial)
            sv = np.linalg.svd(A[:3, :3], compute_uv=False)
            Rm = np.linalg.svd(A[:3, :3])
            rot = float(np.degrees(np.arccos(np.clip((np.trace(Rm[0] @ Rm[2]) - 1) / 2, -1, 1))))
            row.update(fer_fold=tm["fold_pct"], fer_sdlogj=tm["sdlogj"], basis_s=tm["basis_s"], deform_s=tm["deform_total_s"],
                       modal_fit_s=tm.get("modal_fit_s", np.nan), correspondences_s=tm.get("correspondences_s", np.nan),
                       fe_response_s=tm.get("fe_response_s", np.nan), peak_mb=tm["peak_mb"], aff_scale_min=float(sv.min()),
                       aff_scale_max=float(sv.max()), aff_rot_deg=rot, aff_trans_mm=float(np.linalg.norm(A[:3, 3])), **extra)
            case_rows.append(row)
            for j, i in enumerate(ids):
                lm_rows.append(dict(cohort=coh, case=c, lm=int(i), **{a: round(float(err[a][j]), 4) for a in ARMS if a in err}))
    keys = []
    for r in case_rows:
        keys += [k for k in r if k not in keys]
    with open(TAB / "PER_CASE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(case_rows)
    with open(TAB / "LANDMARKS.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cohort", "case", "lm"] + ARMS, restval=""); w.writeheader(); w.writerows(lm_rows)
    for coh in C.COHORTS:
        rr = [r for r in case_rows if r["cohort"] == coh]
        if rr:
            def m(k):
                v = [r[k] for r in rr if k in r and r[k] == r[k]]
                return f"{np.mean(v):.3f} (n {len(v)})" if v else "—"
            print(f"{coh:14s} n {len(rr):2d} | initial {m('initial')} | affine {m('affine')} | f3d MIND {m('f3d_mind')} | "
                  f"FER {m('fer')} | MCPO {m('mcpo')} | MCPO pose {m('mcpo_pose')} | pose+FER {m('mcpo_pose_fer')} | bestfit {m('bestfit')}")
    print(f"wrote {TAB}/PER_CASE.csv ({len(case_rows)} cases), LANDMARKS.csv ({len(lm_rows)} pairs)")


if __name__ == "__main__":
    main()
