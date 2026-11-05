"""MCPO (Wang et al. 2026; github.com/wjiazheng/MCPO, MIT) as the full-method comparator of the 2026-10 version:
the authors' official script, unmodified, at its default settings, started from the navigation alignment.

  remind    the organisers' ReMIND2Reg files themselves (256^3 at 0.5 mm; fixed = iUS `_0000`, moving = ceT1 `_0001`,
            the script's default; ReMIND-065 has no ceT1 in the release -> T2 `_0002` under the `_0001` name, the
            modality change documented in the repository README).  Fields already computed by
            mcpo_faithful_remind.py are read (early cases from G:, read-only); `--t2` scores the T2 variant where the
            release provides a T2 (supplementary analysis).
            ReMIND-009 is NOT in the release: the script is run here, as for the public sets, on the 0.5 mm box of
            the collection's own iUS, with the collection's ceT1 affinely co-registered to the T2 as the organisers
            did (remind/data/ReMIND-009_ceT1.nii.gz, made by _mcpo_009_prep.py), and with the T2 under `--t2`.
  resect_after / bite_g4 / resect_before
            not part of the challenge release: iUS and MR resampled onto the 0.5 mm box of the iUS extent (the
            resolution of the challenge data), padded to even dimensions as the script requires; script run here.
  external  the field of brat_eval.py (challenge-style 256^3 0.5 mm crop centred on the iUS).

Per case a json with per-landmark errors, folding / SDLogJ at the native 0.5 mm grid and after 2x block averaging
(1 mm, comparable with the other arms), and the wall time when the script is run here.
Usage: python fer_mcpo.py [--cohorts ...] [--t2] [--keep cohort/case ...]
Output: /mnt/c/ReMIND_work/fer/mcpo[_t2]/<cohort>/<case>.json
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa
import v2_deform as V                                                # noqa
from reg_core import resample_to_grid                                # noqa

PY = "/home/ncrhurh/anaconda3/envs/monai/bin/python"
MCPO = C.W / "scripts/third_party/MCPO_p4_6543.py"
R2R = Path("/mnt/g/ECOGRAFIA/ReMIND_downloads/ReMIND2Reg/extracted/ReMIND2Reg/imagesTr")
FAITH = [C.W / "results/v2/remind_eval/mcpo_faithful", Path("/mnt/g/ECOGRAFIA/ReMIND_biomech/results/v2/remind_eval/mcpo_faithful")]
# ReMIND case -> ReMIND2Reg id (world-frame iUS correlation r >= 0.989; mcpo_faithful_remind.py map); 009 not in the release
R2R_ID = {"ReMIND-003": "0101", "ReMIND-006": "0098", "ReMIND-015": "0085", "ReMIND-021": "0080", "ReMIND-023": "0074",
          "ReMIND-024": "0077", "ReMIND-029": "0024", "ReMIND-036": "0063", "ReMIND-038": "0045", "ReMIND-042": "0056",
          "ReMIND-043": "0002", "ReMIND-059": "0014", "ReMIND-062": "0051", "ReMIND-064": "0061", "ReMIND-065": "0012"}


def run_script(ind, outd, log):
    with open(log, "a") as fh:
        return subprocess.run([PY, str(MCPO), "--datasets_path", str(ind) + "/", "--result_path", str(outd) + "/"],
                              stdout=fh, stderr=subprocess.STDOUT).returncode


def measure(ui, fixed_mask, mri, usl):
    """Per-landmark error and regularity of an MCPO field (voxel displacements on the fixed grid)."""
    u = np.asarray(ui.dataobj, np.float32)
    inv = np.linalg.inv(ui.affine)
    v = usl @ inv[:3, :3].T + inv[:3, 3]
    du = np.stack([map_coordinates(u[..., k], v.T, order=1, mode="nearest") for k in range(3)], -1)
    err = np.linalg.norm(usl + du @ ui.affine[:3, :3].T - mri, axis=1)
    inside = int(np.all((v >= 0) & (v <= np.array(u.shape[:3]) - 1), axis=1).sum())
    m = np.asarray(fixed_mask, bool)
    fold05, sd05 = C.regularity(u, m)
    s = [d - d % 2 for d in u.shape[:3]]
    uc = u[:s[0], :s[1], :s[2]].reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2, 3).mean(axis=(1, 3, 5)) / 2.0
    mc = m[:s[0], :s[1], :s[2]].reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2).mean(axis=(1, 3, 5)) > 0.5
    fold1, sd1 = C.regularity(uc, mc)
    return err, dict(fold_pct_05=fold05, sdlogj_05=sd05, fold_pct=fold1, sdlogj=sd1, lm_inside=inside)


def public_case(coh, c, keep, mr_path=None, moving=None):
    """The script on the 0.5 mm box of the iUS extent; `mr_path` overrides the moving image (ReMIND-009: ceT1 / T2)."""
    us, mr, _, _, _ = C.inputs(coh, c)
    if mr_path is not None:
        mr = mr_path
    usi, mri_img = nib.load(str(us)), nib.load(str(mr))
    aff, dims = V.box05(usi)
    fx = resample_to_grid(usi, aff, dims).astype(np.float32)
    mv = resample_to_grid(mri_img, aff, dims).astype(np.float32)
    pad = [(0, int(s % 2)) for s in fx.shape]
    fx, mv = np.pad(fx, pad), np.pad(mv, pad)
    ind, outd = Path("/tmp/fermcpo/in"), Path("/tmp/fermcpo/out")        # no underscore in the path (script splits on it)
    for p in (ind, outd):
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True)
    nib.save(nib.Nifti1Image(fx, aff), str(ind / "ReMIND2Reg_x0_0000.nii.gz"))
    nib.save(nib.Nifti1Image(mv, aff), str(ind / "ReMIND2Reg_x0_0001.nii.gz"))
    (C.FER / "logs").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rc = run_script(ind, outd, C.FER / "logs/mcpo_public.log")
    wall = time.time() - t0
    dp = outd / "disp_x0_0000_x0_0001.nii.gz"
    if not dp.exists():
        raise RuntimeError(f"no output (rc {rc})")
    ui = nib.load(str(dp))
    mri, usl, ids = C.landmarks(coh, c)
    err, reg = measure(ui, fx > 0, mri, usl)
    if keep:
        kd = C.FER / "mcpo_fields" / coh / c
        kd.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dp, kd / "disp_vox.nii.gz")
    shutil.rmtree("/tmp/fermcpo", ignore_errors=True)
    m = dict(ids=[int(i) for i in ids], err=err.tolist(), tre=float(err.mean()), wall_s=wall, grid=list(fx.shape), **reg)
    if moving is not None:
        m.update(moving=moving, source="own 0.5 mm box of the collection's images (not in the ReMIND2Reg release)")
    return m


def remind_own_case(c, t2, keep):
    """ReMIND patient absent from the release (ReMIND-009): the script on our box, ceT1 in the T2 frame by default."""
    ce = C.W / f"remind/data/{c}_ceT1.nii.gz"
    if t2 or not ce.exists():
        return public_case("remind", c, keep, moving="T2")
    return public_case("remind", c, keep, mr_path=ce, moving="ceT1")


def remind_case(c, mod):
    if c not in R2R_ID:
        return None
    rid = R2R_ID[c]
    dp = next((r / c / mod / f"disp_{rid}_0000_{rid}_0001.nii.gz" for r in FAITH
               if (r / c / mod / f"disp_{rid}_0000_{rid}_0001.nii.gz").exists()), None)
    if dp is None:
        return None
    ui = nib.load(str(dp))
    fx = np.asarray(nib.load(str(R2R / f"ReMIND2Reg_{rid}_0000.nii.gz")).dataobj) > 0
    mri, usl, ids = C.landmarks("remind", c)
    err, reg = measure(ui, fx, mri, usl)
    return dict(ids=[int(i) for i in ids], err=err.tolist(), tre=float(err.mean()), r2r=rid, moving=mod, source=str(dp), **reg)


def external_case(c):
    d = C.W / "brat" / c
    ui = nib.load(str(C.W / f"brat/out/{c}/mcpo/disp.nii.gz"))
    usi = nib.load(str(d / "us.nii.gz"))
    fx = resample_to_grid(usi, ui.affine, ui.shape[:3]).astype(np.float32) > 0
    mri, usl, ids = C.landmarks("external", c)
    err, reg = measure(ui, fx, mri, usl)
    return dict(ids=[int(i) for i in ids], err=err.tolist(), tre=float(err.mean()), moving=("T1" if c == "Subject-3" else "FLAIR"), **reg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="*", default=C.COHORTS)
    ap.add_argument("--t2", action="store_true", help="ReMIND only: the T2 variant (supplementary)")
    ap.add_argument("--keep", nargs="*", default=[])
    a = ap.parse_args()
    out = C.FER / ("mcpo_t2" if a.t2 else "mcpo")
    for coh in a.cohorts:
        if a.t2 and coh != "remind":
            continue
        for c in C.cases(coh):
            f = out / coh / f"{c}.json"
            keep = f"{coh}/{c}" in a.keep
            if f.exists() and not keep:
                continue
            try:
                if coh == "remind":
                    if c not in R2R_ID:                               # ReMIND-009: run here (field kept for the pose fit)
                        m = remind_own_case(c, a.t2, keep=not a.t2)
                    elif a.t2:
                        m = remind_case(c, "T2")
                    else:
                        m = remind_case(c, "ceT1") or remind_case(c, "T2")
                elif coh == "external":
                    m = external_case(c)
                else:
                    m = public_case(coh, c, keep)
                if m is None:
                    print(f"[mcpo] {coh} {c}: no field (modality not run)", flush=True)
                    continue
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(dict(cohort=coh, case=c, **m)))
                print(f"[mcpo] {coh} {c}: TRE {m['tre']:.2f}, fold {m['fold_pct']:.3f} %, SDLogJ {m['sdlogj']:.3f} "
                      f"(0.5 mm: {m['sdlogj_05']:.3f}){', moving ' + m['moving'] if 'moving' in m else ''}", flush=True)
            except Exception as e:                                    # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[mcpo] {coh} {c} FAILED: {e}", flush=True)
    print("MCPO DONE", flush=True)


if __name__ == "__main__":
    main()
