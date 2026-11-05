"""Evaluate the FROZEN v2 method (scripts/v2_method.py) on ReMIND cases with the annotation-study
landmarks, next to the arms already produced by annotation_eval.py (initial, NiftyReg L2R,
L2R + f3d, ConvexAdam, deformation-only, pose, shipped ours) and MCPO (official script,
defaults, from the navigation start).  Nothing of the annotation study is modified: the pose
(search + guarded polish), the gate decision and the FEM bases are read from
results/annotation_remind/arms/<case>/ exactly as annotation_eval.py produced them.

Per case:  v2_A0  = same-session control (shipped modal stage from the pilot pose and basis)
           v2     = frozen v2: accepted pose -> R3_sym; abstained -> identity -> I4_fem
           mcpo   = MCPO defaults on the 0.5 mm iUS box from the navigation start
Landmarks: Slicer markups (LPS) in --src: MR-<id>.mrk.json (MR references) and US-<obs>-<id>.mrk.json
per iUS observer; consensus = midpoint of the observers; the pre-registered pair rule is reported.
Usage: python v2_remind_eval.py --cases ReMIND-023 [ReMIND-005 ...] [--obs olga santi] [--src DIR] [--skip-mcpo]
Output: results/v2/remind_eval/<case>/{TRE.md,TRE_sets.csv,TRE_landmarks.csv}, results/v2/remind_eval/SUMMARY.md
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotation_eval as E                                          # noqa
import method_arms as M                                              # noqa
import v2_deform as V                                                # noqa
import v2_method as FM                                               # noqa
from annotation_qc import read_markups, horn, qc_pairs              # noqa
from reg_core import load_to_grid, normalize, resample_to_grid       # noqa
from reg_modal import build_phi                                      # noqa
from rs_prestage import relabel                                      # noqa

ARCH = Path("/mnt/c/ReMIND_work/archive")
OUT = M.R / "v2/remind_eval"
PY = "/home/ncrhurh/anaconda3/envs/monai/bin/python"
MCPO = M.W / "scripts/third_party/MCPO_p4_6543.py"
DEV = "cuda"
ORDER = ["initial", "niftyreg_l2r", "niftyreg_l2r_f3d", "convexadam", "deformation_only", "pose_polished", "ours",
         "v2_A0", "v2", "mcpo", "bestfit_rigid", "bestfit_rigid_loo"]


def paths(case):
    return ARCH / f"data/us/{case}_USpreimri.nii.gz", ARCH / f"data/synthseg_in/preop/{case}_T2.nii.gz"


def disp_at(p, pts):
    u = nib.load(str(p))
    a = np.asarray(u.dataobj, np.float32)
    inv = np.linalg.inv(u.affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    return np.stack([map_coordinates(a[..., k], [v[:, 0], v[:, 1], v[:, 2]], order=1, mode="nearest") for k in range(3)], -1)


def deform(case, cfg, tag, A, basis):
    """The v2 deformation stage from pose A with the given FEM basis (frozen configuration)."""
    fx, mv = paths(case)
    out = OUT / case / tag
    if (out / "disp_mm.nii.gz").exists():
        return out / "disp_mm.nii.gz"
    out.mkdir(parents=True, exist_ok=True)
    mv_r = out / "moving_rig.nii.gz"
    relabel(mv, mv_r, A)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    f_arr, m_arr, grid_aff, dims = load_to_grid(fx, mv_r, 1.0)
    fov = f_arr > 0
    f_n, m_n = normalize(f_arr, fov), normalize(m_arr, m_arr > 0)
    phi, _ = build_phi(basis, grid_aff, dims, 1.0)
    eig = np.load(str(basis))["eigvals"]
    phi = phi.to(DEV)
    _, u = V.register_modal_from(f_n, m_n, fov, phi, eig, None, cfg["modal_iters"], 3e-3, 0.1, tuple(cfg["mind"]), DEV)
    meta = dict(case=case, tag=tag, cfg=json.loads(json.dumps(cfg, default=list)), version=FM.VERSION)
    if cfg["residual"] == "grad":
        u_tot = V.refine_residual_grad(f_n, m_n, fov, u, mind=tuple(cfg["mind"]), dev=DEV)
    else:
        rc = cfg["res"]
        usi, mri_img = nib.load(str(fx)), nib.load(str(mv))
        aff05, dims05 = V.box05(usi)
        f05 = resample_to_grid(usi, aff05, dims05).astype(np.float32)
        f05 = torch.from_numpy(normalize(f05, f05 > 0))[None, None].to(DEV)
        mw05 = V.warp_mr_on_box(mri_img, A, u, grid_aff, aff05, dims05, DEV)
        mw05 = V.normalize_t(mw05, mw05 > 0)
        corr = V.residual_correspondences(f05, mw05, aff05, [tuple(l) for l in rc["levels"]], mind=tuple(rc["mind"]),
                                          coupled=rc["coupled"], weight=rc["weight"], dev=DEV, symmetric=rc["symmetric"],
                                          max_pts=rc["max_pts"])
        z = np.load(str(basis), allow_pickle=True)
        u_nodes, rinfo = V.fem_residual(z, corr, rc["lam_rel"], rc["huber_mm"], trim=rc["trim"])
        r = V.lattice_field(z, u_nodes, grid_aff, dims).to(DEV)
        n = r.norm(dim=1, keepdim=True)
        rinfo["clipped_frac"] = float((n > rc["max_mm"]).float().mean())
        r = r * (rc["max_mm"] / n.clamp(min=rc["max_mm"]))
        meta["residual"] = rinfo
        u_tot = u + r
        del mw05, f05
    torch.cuda.synchronize()
    fold, sdlj = V.regularity(u_tot, fov)
    meta.update(wall_s=time.time() - t0, peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20, fold_pct=fold, sdlogj=sdlj)
    nib.save(nib.Nifti1Image(np.moveaxis(u_tot[0].cpu().numpy(), 0, -1).astype(np.float32), grid_aff), str(out / "disp_mm.nii.gz"))
    (out / "meta.json").write_text(json.dumps(meta))
    mv_r.unlink(missing_ok=True)
    del phi, u, u_tot
    torch.cuda.empty_cache()
    print(f"[v2-remind] {case} {tag}: {meta['wall_s']:.0f} s, {meta['peak_mb']:.0f} MB, fold {fold:.3f} %, SDLogJ {sdlj:.3f}", flush=True)
    return out / "disp_mm.nii.gz"


def mcpo_nav(case):
    fx, mv = paths(case)
    cid = case.replace("-", "")
    ind, outd = Path(f"/tmp/mcporemind/{cid}"), Path(f"/tmp/mcporemindout/{cid}")
    dp = outd / f"disp_{cid}_0000_{cid}_0001.nii.gz"
    keep = OUT / case / "mcpo" / "disp.nii.gz"
    if keep.exists():
        return keep
    ind.mkdir(parents=True, exist_ok=True)
    outd.mkdir(parents=True, exist_ok=True)
    usi, mri_img = nib.load(str(fx)), nib.load(str(mv))
    aff, dims = V.box05(usi)
    f = resample_to_grid(usi, aff, dims).astype(np.float32)
    m = resample_to_grid(mri_img, aff, dims).astype(np.float32)
    pad = [(0, int(s % 2)) for s in f.shape]
    nib.save(nib.Nifti1Image(np.pad(f, pad), aff), str(ind / f"ReMIND2Reg_{cid}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(np.pad(m, pad), aff), str(ind / f"ReMIND2Reg_{cid}_0001.nii.gz"))
    with open(M.W / f"logs/mcpo_remind_{cid}.log", "w") as fh:
        rc = subprocess.run([PY, str(MCPO), "--datasets_path", str(ind) + "/", "--result_path", str(outd) + "/"],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"[mcpo-remind] {case}: rc {rc}", flush=True)
    if dp.exists():
        import shutil
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dp, keep)                     # /tmp and results are on different devices
        shutil.rmtree(ind, ignore_errors=True)
        shutil.rmtree(outd, ignore_errors=True)
        return keep
    return None


def mcpo_pred(dp, us):
    uimg = nib.load(str(dp))
    aff = uimg.affine
    u = np.asarray(uimg.dataobj, np.float32)
    inv = np.linalg.inv(aff)
    v = us @ inv[:3, :3].T + inv[:3, 3]
    du = np.stack([map_coordinates(u[..., k], [v[:, 0], v[:, 1], v[:, 2]], order=1, mode="nearest") for k in range(3)], -1)
    return us + du @ aff[:3, :3].T


def eval_case(case, obs_names, src, skip_mcpo):
    arms = E.ARMS_DIR / case
    info = json.loads((arms / "search_source.json").read_text())
    abstain = bool(info["abstain"])
    A1 = np.loadtxt(arms / "search_polish/aff.txt")
    b_pose, b_id = arms / "ours/basis_rig.npz", arms / "identity/basis_rig.npz"
    ctrl_cfg = dict(FM.ACCEPTED, residual="grad")                     # shipped modal stage = control
    if abstain:
        d_a0 = deform(case, dict(FM.FALLBACK, residual="grad"), "v2_A0_identity", np.eye(4), b_id)
        d_v2 = deform(case, FM.FALLBACK, "v2_I4", np.eye(4), b_id)
        A_v2 = np.eye(4)
    else:
        d_a0 = deform(case, ctrl_cfg, "v2_A0", A1, b_pose)
        d_v2 = deform(case, FM.ACCEPTED, "v2_R3sym", A1, b_pose)
        A_v2 = A1
    dp = None if skip_mcpo else mcpo_nav(case)
    cid = case.split("-")[1]

    def find(cands):
        for c in cands:
            if (src / c).exists():
                return src / c
        raise FileNotFoundError(f"{case}: none of {cands} in {src}")
    # two naming conventions: MR-<id>.mrk.json / US-<obs>-<id>.mrk.json (pilot re-annotation) and
    # ReMIND-<id>-MR.mrk.json / ReMIND-<id>-US-<OBS>.mrk.json (LANDMARKS folder, 2026-09)
    mr, _ = read_markups(find([f"MR-{cid}.mrk.json", f"{case}-MR.mrk.json"]))
    obs = {n: read_markups(find([f"US-{n}-{cid}.mrk.json", f"{case}-US-{n}.mrk.json", f"{case}-US-{n.upper()}.mrk.json",
                                 f"US-{n.lower()}-{cid}.mrk.json"]))[0] for n in obs_names}
    ids = sorted(set(mr).intersection(*[set(v) for v in obs.values()]))
    MR = np.array([mr[i] for i in ids])
    sets = {n: np.array([v[i] for i in ids]) for n, v in obs.items()}
    sets["consensus"] = np.mean([sets[n] for n in obs_names], axis=0)
    keep = np.ones(len(ids), bool)
    if len(obs_names) == 2:
        keep, thr = qc_pairs(np.linalg.norm(sets[obs_names[0]] - sets[obs_names[1]], axis=1))
    rows, lm_rows = [], []
    for sname in ["consensus"] + obs_names:
        variants = [(sname, np.ones(len(ids), bool))] + ([("consensus_prereg_rule", keep)] if sname == "consensus" and not keep.all() else [])
        for vname, m in variants:
            US = sets[sname][m]
            preds, _ = E.arm_predictions(case, US)
            preds["v2_A0"] = (US + disp_at(d_a0, US)) @ A_v2[:3, :3].T + A_v2[:3, 3]
            preds["v2"] = (US + disp_at(d_v2, US)) @ A_v2[:3, :3].T + A_v2[:3, 3]
            if dp is not None:
                preds["mcpo"] = mcpo_pred(dp, US)
            err = {k: np.linalg.norm(preds[k] - MR[m], axis=1) for k in ORDER if k in preds}
            R, t = horn(US, MR[m])
            err["bestfit_rigid"] = np.linalg.norm(US @ R.T + t - MR[m], axis=1)
            loo = []
            for k in range(m.sum()):
                mm = np.ones(m.sum(), bool); mm[k] = False
                Rk, tk = horn(US[mm], MR[m][mm])
                loo.append(np.linalg.norm(US[k] @ Rk.T + tk - MR[m][k]))
            err["bestfit_rigid_loo"] = np.array(loo)
            hard = np.argsort(-err["initial"])[:max(1, int(round(0.3 * m.sum())))]
            for k in ORDER:
                if k in err:
                    e = err[k]
                    rows.append(dict(case=case, set=vname, n=int(m.sum()), arm=k, mean=round(float(e.mean()), 3),
                                     median=round(float(np.median(e)), 3), max=round(float(e.max()), 3), tre30=round(float(e[hard].mean()), 3),
                                     better_than_initial=int((e < err["initial"]).sum()) if k != "initial" else ""))
            if vname == "consensus":
                for j, i in enumerate(ids):
                    lm_rows.append(dict(case=case, lm=i, **{k: round(float(err[k][j]), 2) for k in ORDER if k in err}))
    d = OUT / case
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "TRE_sets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(d / "TRE_landmarks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lm_rows[0])); w.writeheader(); w.writerows(lm_rows)
    L = [f"# {case}: {len(ids)} landmarks; pose rot {info['rot_deg']:.2f} deg, {'ABSTAINED (identity fallback)' if abstain else 'accepted'}; v2 {FM.VERSION}", ""]
    for vname in dict.fromkeys(r["set"] for r in rows):
        n = [r["n"] for r in rows if r["set"] == vname][0]
        L += [f"## {vname} (n={n})", "", "| arm | mean | median | max | TRE30 | better than initial |", "|---|---|---|---|---|---|"]
        L += [f"| {r['arm']} | {r['mean']:.2f} | {r['median']:.2f} | {r['max']:.2f} | {r['tre30']:.2f} | {r['better_than_initial']} |" for r in rows if r["set"] == vname]
        L.append("")
    (d / "TRE.md").write_text("\n".join(L))
    print("\n".join(L[:20]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--obs", nargs="+", default=["olga", "santi"])
    ap.add_argument("--src", default=str(M.W / "annotation_remind/PILOT"))
    ap.add_argument("--skip-mcpo", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    allrows = []
    for case in a.cases:
        try:
            allrows += eval_case(case, a.obs, Path(a.src), a.skip_mcpo)
        except Exception as e:                                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"[v2-remind] {case} FAILED: {e}", flush=True)
    cons = [r for r in allrows if r["set"] == "consensus"]
    cases = sorted({r["case"] for r in cons})
    if len(cases) >= 1:
        L = [f"# v2 {FM.VERSION} on ReMIND: {len(cases)} cases, consensus landmarks (mean of case means, mm)", "",
             "| arm | mean | median of case means | cases better than initial | vs v2: p (Wilcoxon zsplit) | v2 better |", "|---|---|---|---|---|---|"]
        per = {k: np.array([[r["mean"] for r in cons if r["case"] == c and r["arm"] == k][0] for c in cases if any(r["case"] == c and r["arm"] == k for r in cons)]) for k in ORDER}
        v2 = per["v2"]
        for k in ORDER:
            x = per[k]
            if len(x) == 0:
                continue
            p = stats.wilcoxon(v2, x, zero_method="zsplit").pvalue if (k != "v2" and len(x) == len(v2) and len(x) > 1) else float("nan")
            L.append(f"| {k} | {x.mean():.2f} | {np.median(x):.2f} | {int((x < per['initial']).sum()) if k != 'initial' else ''}/{len(x)} | "
                     f"{'' if np.isnan(p) else f'{p:.3f}'} | {'' if k == 'v2' or len(x) != len(v2) else f'{int((v2 < x).sum())}/{len(x)}'} |")
        (OUT / "SUMMARY.md").write_text("\n".join(L))
        print("\n".join(L))


if __name__ == "__main__":
    main()
