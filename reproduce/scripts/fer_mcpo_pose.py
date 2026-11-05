"""Second initialisation of the 2026-10 manuscript: the POSE of MCPO + the same frozen refinement as fer_run.py.

MCPO pose = the rigid transform (US world -> MR world) that best fits MCPO's displacement field over the iUS field of
view (trimmed least squares on every 4th FoV voxel, keep 60 %, 4 passes; mcpo_pose_deform.trimmed_rigid).
Field source (the same MCPO runs as the manuscript's MCPO row, fer_mcpo.py):
  RESECT post / BITE   results/mcpo_pose/<coh>/<case>/pose.txt, fitted earlier from the official script at its defaults
                       on the 0.5 mm iUS box (identical set-up and results to fer_mcpo.py)
  ReMIND               fitted here from the official run on the organisers' ReMIND2Reg files, ceT1 moving (fer_mcpo.FAITH);
                       ReMIND-009 (not in the release) from the run of fer_mcpo.py on our 0.5 mm box (fer/mcpo_fields)
  external             fitted here from brat/out/<case>/mcpo/disp.nii.gz
Refinement: exactly fer_run.py (SynthSeg relabelled by the pose, FE model h 5 mm / 30 modes, NO tumour compartment,
v2_method.ACCEPTED through v2_remind_eval.deform), output layout of fer_run.py under fer/run_mcpopose/.
Usage: python fer_mcpo_pose.py [--cohorts resect_after bite_g4 remind external] [--cases ...]
Output: fer/mcpo_pose/<coh>/<case>/{pose.txt,fit.json}; fer/run_mcpopose/<coh>/<case>/{synthseg_rig.nii.gz,basis30.npz,
        fer/disp_mm.nii.gz,fer/meta.json,timing.json}
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa
import fer_run as FR                                                 # noqa
import fer_mcpo as FM_                                               # noqa  (FAITH, R2R, R2R_ID)
import single_config as SC                                           # noqa
import affine_cascade as AC                                          # noqa
import v2_deform as V                                                # noqa
import v2_method as FM                                               # noqa
import v2_remind_eval as VE                                          # noqa
from mcpo_pose_deform import trimmed_rigid                           # noqa
from reg_core import resample_to_grid                                # noqa

POSE = C.FER / "mcpo_pose"
OUT = C.FER / "run_mcpopose"
COHORTS = ["resect_after", "bite_g4", "remind", "external"]


def fit_pose(ui, fov):
    """Rigid fit of a displacement field (voxel units of ui, as MCPO writes it) over the FoV mask (same grid)."""
    u = np.asarray(ui.dataobj, np.float32)
    ijk = np.argwhere(fov[::4, ::4, ::4]) * 4
    p = ijk @ ui.affine[:3, :3].T + ui.affine[:3, 3]
    q = p + u[ijk[:, 0], ijk[:, 1], ijk[:, 2]] @ ui.affine[:3, :3].T
    Rm, t, inl = trimmed_rigid(p, q, keep=0.6, iters=4)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = Rm, t
    res = float(np.linalg.norm(p[inl] @ Rm.T + t - q[inl], axis=1).mean())
    return A, dict(n_points=int(len(p)), n_inliers=int(inl.sum()), fit_residual_mm=res)


def pose_for(coh, c):
    d = POSE / coh / c
    if (d / "pose.txt").exists():
        return np.loadtxt(d / "pose.txt")
    d.mkdir(parents=True, exist_ok=True)
    if coh in ("resect_after", "bite_g4"):
        src = C.W / "results/mcpo_pose" / coh / c / "pose.txt"
        A = np.loadtxt(src)
        meta = json.loads((C.W / "results/mcpo_pose" / coh / c / "meta.json").read_text())
        info = dict(source=str(src), fit_residual_mm=meta.get("fit_res_mm"), mcpo_run="official script, defaults, 0.5 mm iUS box (results/mcpo_pose)")
    elif coh == "remind" and c not in FM_.R2R_ID:                  # ReMIND-009: the script run here (fer_mcpo.remind_own_case)
        ui = nib.load(str(C.FER / "mcpo_fields/remind" / c / "disp_vox.nii.gz"))
        usi = nib.load(str(C.inputs(coh, c)[0]))
        fov = resample_to_grid(usi, ui.affine, ui.shape[:3]).astype(np.float32) > 0
        A, info = fit_pose(ui, fov)
        info.update(source=str(ui.get_filename()),
                    mcpo_run="official script, defaults, 0.5 mm iUS box of the collection's images (not in the release)")
    elif coh == "remind":
        rid = FM_.R2R_ID[c]
        dp = next((r / c / "ceT1" / f"disp_{rid}_0000_{rid}_0001.nii.gz" for r in FM_.FAITH
                   if (r / c / "ceT1" / f"disp_{rid}_0000_{rid}_0001.nii.gz").exists()), None)
        if dp is None:                                               # ReMIND-065: the release has only a T2
            dp = next((r / c / "T2" / f"disp_{rid}_0000_{rid}_0001.nii.gz" for r in FM_.FAITH
                       if (r / c / "T2" / f"disp_{rid}_0000_{rid}_0001.nii.gz").exists()), None)
        ui = nib.load(str(dp))
        fov = np.asarray(nib.load(str(FM_.R2R / f"ReMIND2Reg_{rid}_0000.nii.gz")).dataobj) > 0
        A, info = fit_pose(ui, fov)
        info.update(source=str(dp), mcpo_run="official script on the organisers' ReMIND2Reg files")
    else:
        ui = nib.load(str(C.W / f"brat/out/{c}/mcpo/disp.nii.gz"))
        usi = nib.load(str(C.W / "brat" / c / "us.nii.gz"))
        fov = resample_to_grid(usi, ui.affine, ui.shape[:3]).astype(np.float32) > 0
        A, info = fit_pose(ui, fov)
        info.update(source=str(ui.get_filename()), mcpo_run="official script, defaults, 256^3 0.5 mm box")
    ang = float(np.degrees(np.arccos(np.clip((np.trace(A[:3, :3]) - 1) / 2, -1, 1))))
    info.update(rotation_deg=ang, translation_mm=float(np.linalg.norm(A[:3, 3])))
    np.savetxt(d / "pose.txt", A)
    (d / "fit.json").write_text(json.dumps(info))
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="*", default=COHORTS)
    ap.add_argument("--cases", nargs="*")
    a = ap.parse_args()
    VE.relabel = AC.relabel
    V.register_modal_from = FR.timed("modal_fit_s", V.register_modal_from)
    V.residual_correspondences = FR.timed("correspondences_s", V.residual_correspondences)
    V.fem_residual = FR.timed("fe_response_s", V.fem_residual)
    for coh in a.cohorts:
        for c in C.cases(coh):
            if a.cases and c not in a.cases:
                continue
            d = OUT / coh / c
            if (d / "fer/disp_mm.nii.gz").exists() and (d / "timing.json").exists():
                continue
            try:
                A = pose_for(coh, c)
                d.mkdir(parents=True, exist_ok=True)
                us, mr, seg, _, _ = C.inputs(coh, c)
                t_all = time.time()
                seg_r, basis = d / "synthseg_rig.nii.gz", d / "basis30.npz"
                AC.relabel(seg, seg_r, A)
                t0 = time.time()
                if not basis.exists():
                    with open(d / "run.log", "a") as fh:
                        rc = subprocess.run([SC.PY, str(SC.S / "biomech_basis.py"), "--synthseg", str(seg_r), "--out", str(basis),
                                             "--h", "5.0", "--n-modes", "30"], stdout=fh, stderr=subprocess.STDOUT).returncode
                    if rc != 0 or not basis.exists():
                        raise RuntimeError("basis failed")
                t_basis = time.time() - t0
                FR.TIMES.clear()
                VE.paths = lambda case, us=us, mr=mr: (us, mr)
                VE.OUT = OUT / coh
                t0 = time.time()
                VE.deform(c, FM.ACCEPTED, "fer", A, basis)
                t_def = time.time() - t0
                meta = json.loads((d / "fer/meta.json").read_text())
                (d / "timing.json").write_text(json.dumps(dict(
                    cohort=coh, case=c, init="mcpo_pose", basis_s=t_basis, deform_total_s=t_def,
                    **{k: float(v) for k, v in FR.TIMES.items()}, total_s=time.time() - t_all, peak_mb=meta["peak_mb"],
                    fold_pct=meta["fold_pct"], sdlogj=meta["sdlogj"], version=FM.VERSION)))
                print(f"[mcpopose] {coh} {c}: basis {t_basis:.0f} s, deformable {t_def:.0f} s, fold {meta['fold_pct']:.3f} %, "
                      f"SDLogJ {meta['sdlogj']:.3f}", flush=True)
            except Exception as e:                                    # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[mcpopose] {coh} {c} FAILED: {e}", flush=True)
    print("MCPO POSE RUN DONE", flush=True)


if __name__ == "__main__":
    main()
