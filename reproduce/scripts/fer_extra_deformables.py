"""ADDITIONAL EXPERIMENT (2026-10-05, not in the manuscript for now): other deformable methods on the SAME challenge
affine as the proposed FE refinement, on the SAME inputs, post-resection cohorts only.

Common set-up (identical for every method here)
  grid     the 1 mm working grid of the iUS (reg_core.load_to_grid: axis-aligned box of the iUS extent)
  fixed    iUS resampled on the grid, intensities normalised (1-99 percentile)
  moving   MR relabelled by the challenge affine A (affine_cascade.relabel, exact), resampled on the SAME grid, i.e.
           CUT to the iUS box (the user's request, "recorte RM en base al US": registering the whole MR volume produced
           unwanted deformation of the MR in earlier work).  Two variants: --crop cone (default; the MR is also zeroed
           outside the iUS cone dilated by `DILATE_MM`, output extra_deformables/) and --crop box (the MR keeps its
           content over the whole box, output extra_deformables_box/).
  output   displacement u on the grid, world mm, FE-refinement convention: predicted MR point = A(x + u(x)) for an iUS
           point x; per-landmark error, folding / SDLogJ over the cone (fer_common.regularity), wall time.
Methods (each at its authors' default settings)
  convexadam  convexAdam.convex_adam_MIND.convex_adam_pt (MIND-SSC r1 d2, coupled convex grid 6 / hw 4, Adam 80 it)
  convexadam_mask  the same with its use_mask mode (fixed mask = cone, moving mask = dilated cone); with --crop box
              this is ConvexAdam on the uncropped MR with masks, its intended way of restricting the field of view
  deeds       deedsBCV (Heinrich), defaults -a 1.6 -l 5 -G 8x7x6x5x4 -L 8x7x6x5x4 -Q 5x4x3x2x1; field read by warping
              coordinate images with applyBCVfloat (no assumption on the .dat layout)
  syn         ANTsPy ants.registration type_of_transform='SyNOnly' (Mattes MI, defaults), fixed mask = cone; field read
              by warping coordinate images with the forward transforms (ants.apply_transforms, linear)
  f3d_crop    NiftyReg reg_f3d as in the manuscript (--mindssc 1 -sx 5 -be 0.005, -rmask cone) but with the cropped
              moving image on the grid and an identity initial affine (the affine is already in the moving image)
The manuscript's f3d (uncropped native MR) stays in fer/f3d/f3d_mind for reference.
Usage: python fer_extra_deformables.py --method convexadam|deeds|syn|f3d_crop [--cohorts ...] [--cases ...]
       python fer_extra_deformables.py --method X --selfcheck      (synthetic 3 mm shift of the fixed image: the
                                                                     recovered mean u must be ~(+3, 0, 0) mm)
Env: anaconda3/envs/monai for everything; syn calls _syn_worker.py under anaconda3/envs/mortal (antspyx 0.3.8).
Output: /mnt/c/ReMIND_work/fer/extra_deformables/<method>/<cohort>/<case>.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fer_common as C                                               # noqa
import affine_cascade as AC                                          # noqa
from reg_core import load_to_grid, normalize                         # noqa

OUT = C.FER / "extra_deformables"                                  # --crop cone (default); --crop box -> extra_deformables_box
CROP = "cone"
POST = ["resect_after", "bite_g4", "remind", "external"]
DILATE_MM = 3
DEEDS = Path("/home/ncrhurh/deedsBCV")
NR = Path("/home/ncrhurh/anaconda3/envs/niftyreg/bin")
TMP = Path("/tmp/ferextra")
MORTAL = "/home/ncrhurh/anaconda3/envs/mortal/bin/python"


def prepare(coh, c):
    """fixed, moving (cropped), cone mask, grid affine; all on the 1 mm working grid."""
    us, mr, _, A, _ = C.inputs(coh, c)
    TMP.mkdir(parents=True, exist_ok=True)
    rig = TMP / f"{c}_moving_rig.nii.gz"
    AC.relabel(mr, rig, A)
    f_arr, m_arr, grid_aff, dims = load_to_grid(us, rig, 1.0)
    cone = f_arr > 0
    wide = binary_dilation(cone, iterations=DILATE_MM) if CROP == "cone" else np.ones_like(cone)
    f_n = normalize(f_arr, cone).astype(np.float32) * cone
    m_n = normalize(m_arr, m_arr > 0).astype(np.float32) * wide        # "box": the MR is only cut to the iUS box
    return f_n, m_n, cone, grid_aff, A


def coord_images(grid_aff, shape):
    """Three float images holding the grid (voxel) coordinates i, j, k (exact under trilinear interpolation)."""
    ii, jj, kk = np.meshgrid(*(np.arange(s, dtype=np.float32) for s in shape), indexing="ij")
    return ii, jj, kk


def field_from_warped_coords(wi, wj, wk, wo, grid_aff):
    """u (world mm) from warped coordinate images: for each fixed voxel x, the warped value = moving voxel it samples.
    `wo` = warped image of ones (validity: 0 where the sampling point left the box)."""
    sh = wi.shape
    ii, jj, kk = coord_images(grid_aff, sh)
    ok = wo > 0.5
    u = np.stack([np.where(ok, wi / np.maximum(wo, 1e-6) - ii, 0), np.where(ok, wj / np.maximum(wo, 1e-6) - jj, 0),
                  np.where(ok, wk / np.maximum(wo, 1e-6) - kk, 0)], -1).astype(np.float32)
    return u @ grid_aff[:3, :3].T, ok                                  # voxel -> world mm (axis-aligned grid)


# ----------------------------------------------------------------------------------------------------------- methods
def run_convexadam(f, m, cone, grid_aff, wd):
    import torch
    from convexAdam.convex_adam_MIND import convex_adam_pt
    t0 = time.time()
    d = convex_adam_pt(f, m)                                          # defaults; (X,Y,Z,3) voxel units, moving = x + u
    torch.cuda.empty_cache()
    u = np.asarray(d, np.float32) @ grid_aff[:3, :3].T
    return u, time.time() - t0, {}


def run_deeds(f, m, cone, grid_aff, wd):
    wd.mkdir(parents=True, exist_ok=True)
    aff = grid_aff
    nib.save(nib.Nifti1Image(f, aff), str(wd / "fixed.nii.gz"))
    nib.save(nib.Nifti1Image(m, aff), str(wd / "moving.nii.gz"))
    t0 = time.time()
    with open(wd / "deeds.log", "w") as fh:
        rc = subprocess.run([str(DEEDS / "deedsBCV"), "-F", str(wd / "fixed.nii.gz"), "-M", str(wd / "moving.nii.gz"),
                             "-O", str(wd / "out")], stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"deedsBCV rc {rc}")
    ii, jj, kk = coord_images(aff, f.shape)
    warped = []
    for name, arr in (("ci", ii), ("cj", jj), ("ck", kk), ("one", np.ones_like(ii))):
        nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), str(wd / f"{name}.nii.gz"))
        with open(wd / "deeds.log", "a") as fh:
            rc = subprocess.run([str(DEEDS / "applyBCVfloat"), "-M", str(wd / f"{name}.nii.gz"), "-O", str(wd / "out"),
                                 "-D", str(wd / f"{name}_w.nii.gz")], stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            raise RuntimeError(f"applyBCVfloat rc {rc}")
        warped.append(np.asarray(nib.load(str(wd / f"{name}_w.nii.gz")).dataobj, np.float32))
    u, ok = field_from_warped_coords(*warped, aff)
    return u, wall, dict(valid_pct=float(100 * ok[cone].mean()))


def run_syn(f, m, cone, grid_aff, wd):
    """ANTsPy lives in another env (mortal, no torch): the registration runs in _syn_worker.py through a subprocess."""
    wd.mkdir(parents=True, exist_ok=True)
    aff = grid_aff
    nib.save(nib.Nifti1Image(f, aff), str(wd / "fixed.nii.gz"))
    nib.save(nib.Nifti1Image(m, aff), str(wd / "moving.nii.gz"))
    nib.save(nib.Nifti1Image(cone.astype(np.float32), aff), str(wd / "cone.nii.gz"))
    ii, jj, kk = coord_images(aff, f.shape)
    for name, arr in (("ci", ii), ("cj", jj), ("ck", kk), ("one", np.ones_like(ii))):
        nib.save(nib.Nifti1Image(arr.astype(np.float32), aff), str(wd / f"{name}.nii.gz"))
    with open(wd / "syn.log", "w") as fh:
        rc = subprocess.run([MORTAL, str(Path(__file__).resolve().parent / "_syn_worker.py"), str(wd)],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not (wd / "syn.json").exists():
        raise RuntimeError(f"syn worker rc {rc}")
    info = json.loads((wd / "syn.json").read_text())
    warped = [np.asarray(nib.load(str(wd / f"{n}_w.nii.gz")).dataobj, np.float32) for n in ("ci", "cj", "ck", "one")]
    u, ok = field_from_warped_coords(*warped, aff)
    return u, info["wall_s"], dict(valid_pct=float(100 * ok[cone].mean()), transforms=info["transforms"])


def run_f3d_crop(f, m, cone, grid_aff, wd):
    import niftyreg_prep
    wd.mkdir(parents=True, exist_ok=True)
    aff = grid_aff
    nib.save(nib.Nifti1Image(f, aff), str(wd / "fixed.nii.gz"))
    nib.save(nib.Nifti1Image(m, aff), str(wd / "moving.nii.gz"))
    nib.save(nib.Nifti1Image(cone.astype(np.uint8), aff), str(wd / "cone.nii.gz"))
    t0 = time.time()
    with open(wd / "f3d.log", "w") as fh:
        rc = subprocess.run([str(NR / "reg_f3d"), "-ref", str(wd / "fixed.nii.gz"), "-flo", str(wd / "moving.nii.gz"),
                             "-rmask", str(wd / "cone.nii.gz"), "--mindssc", "1", "-sx", "5", "-be", "0.005",
                             "-cpp", str(wd / "cpp.nii.gz"), "-res", str(wd / "res.nii.gz"), "-voff"],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
        if rc == 0:
            rc = subprocess.run([str(NR / "reg_transform"), "-ref", str(wd / "fixed.nii.gz"), "-def", str(wd / "cpp.nii.gz"),
                                 str(wd / "def.nii.gz")], stdout=fh, stderr=subprocess.STDOUT).returncode
    wall = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"reg_f3d rc {rc}")
    niftyreg_prep.def2disp(wd / "def.nii.gz", wd)                    # -> wd/disp_mm.nii.gz (world mm on the ref grid)
    u = np.asarray(nib.load(str(wd / "disp_mm.nii.gz")).dataobj, np.float32)
    return u, wall, {}


def run_convexadam_mask(f, m, cone, grid_aff, wd):
    """ConvexAdam's own mask mode (use_mask: features outside the masks replaced by the nearest inside ones):
    fixed mask = cone, moving mask = cone dilated by DILATE_MM.  Meant for --crop box (uncropped MR + masks)."""
    import torch
    from convexAdam.convex_adam_MIND import convex_adam_pt
    wd.mkdir(parents=True, exist_ok=True)
    pad = [(0, int(s % 2)) for s in f.shape]                          # its mask mode needs even dimensions (::2 vs pooling)
    fp, mp = np.pad(f, pad), np.pad(m, pad)
    cp, wp = np.pad(cone, pad), np.pad(binary_dilation(cone, iterations=DILATE_MM), pad)
    nib.save(nib.Nifti1Image(cp.astype(np.float32), grid_aff), str(wd / "mask_fixed.nii.gz"))
    nib.save(nib.Nifti1Image(wp.astype(np.float32), grid_aff), str(wd / "mask_moving.nii.gz"))
    t0 = time.time()
    d = convex_adam_pt(fp, mp, use_mask=True, path_fixed_mask=str(wd / "mask_fixed.nii.gz"), path_moving_mask=str(wd / "mask_moving.nii.gz"))
    torch.cuda.empty_cache()
    d = np.asarray(d, np.float32)[:f.shape[0], :f.shape[1], :f.shape[2]]
    u = d @ grid_aff[:3, :3].T
    return u, time.time() - t0, dict(masks="cone / cone dilated", padded_even=True)


RUN = dict(convexadam=run_convexadam, convexadam_mask=run_convexadam_mask, deeds=run_deeds, syn=run_syn, f3d_crop=run_f3d_crop)


def selfcheck(method):
    """Fixed = iUS of a case; moving = the same image shifted by +3 mm along world x (resampled); the recovered u over
    the cone must average ~(+3, 0, 0) mm if the convention (moving = x + u, world axes) is right."""
    coh, c = "remind", "ReMIND-024"
    f, m, cone, grid_aff, A = prepare(coh, c)
    ii, jj, kk = coord_images(grid_aff, f.shape)
    shifted = map_coordinates(f, [ii + 3.0, jj, kk], order=1, mode="constant")   # shifted(x) = f(x + 3) -> f(x) = shifted(x - 3)
    # moving = shifted: fixed f(x) = moving(x - 3) -> u = -3 along i ... we define moving(x + u) = fixed(x) -> u = -3
    u, wall, extra = RUN[method](f, shifted.astype(np.float32), cone, grid_aff, TMP / f"selfcheck_{method}")
    inner = binary_dilation(~cone, iterations=8) == 0                  # away from the cone border
    print(f"[selfcheck {method}] expected u = (-3, 0, 0) mm; recovered mean over the inner cone: "
          f"{u[inner].mean(0).round(2)}, median {np.median(u[inner], 0).round(2)}, {wall:.0f} s {extra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(RUN))
    ap.add_argument("--cohorts", nargs="*", default=POST)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--keep", nargs="*", default=[], help="cohort/case whose field is kept")
    ap.add_argument("--crop", choices=["cone", "box"], default="cone",
                    help="cone: MR zeroed outside the dilated iUS cone (default); box: MR only cut to the iUS box")
    a = ap.parse_args()
    global OUT, CROP, TMP
    CROP = a.crop
    if a.crop == "box":
        OUT, TMP = C.FER / "extra_deformables_box", Path("/tmp/ferextra_box")
    if a.selfcheck:
        selfcheck(a.method)
        return
    for coh in a.cohorts:
        for c in C.cases(coh):
            if a.cases and c not in a.cases:
                continue
            fj = OUT / a.method / coh / f"{c}.json"
            if fj.exists():
                continue
            try:
                f, m, cone, grid_aff, A = prepare(coh, c)
                wd = TMP / a.method / coh / c
                shutil.rmtree(wd, ignore_errors=True)
                u, wall, extra = RUN[a.method](f, m, cone, grid_aff, wd)
                uimg = nib.Nifti1Image(u.astype(np.float32), grid_aff)
                mri, usl, ids = C.landmarks(coh, c)
                err = np.linalg.norm(C.apply(A, usl + C.disp_at(uimg, usl)) - mri, axis=1)
                err_aff = np.linalg.norm(C.apply(A, usl) - mri, axis=1)
                fold, sd = C.regularity(u @ np.linalg.inv(grid_aff[:3, :3]).T, cone)
                mag = np.linalg.norm(u, axis=-1)[cone]
                fj.parent.mkdir(parents=True, exist_ok=True)
                fj.write_text(json.dumps(dict(cohort=coh, case=c, method=a.method, ids=[int(i) for i in ids], err=err.tolist(),
                                              tre=float(err.mean()), affine_tre=float(err_aff.mean()), fold_pct=fold, sdlogj=sd,
                                              disp_mean_mm=float(mag.mean()), disp_max_mm=float(mag.max()), wall_s=wall,
                                              grid=list(f.shape), dilate_mm=DILATE_MM, crop=CROP, **extra)))
                if f"{coh}/{c}" in a.keep:
                    kd = OUT / "fields" / a.method / coh / c
                    kd.mkdir(parents=True, exist_ok=True)
                    nib.save(uimg, str(kd / "disp_mm.nii.gz"))
                shutil.rmtree(wd, ignore_errors=True)
                print(f"[{a.method}] {coh} {c}: affine {err_aff.mean():.2f} -> {err.mean():.2f} mm, fold {fold:.3f} %, "
                      f"SDLogJ {sd:.3f}, |u| {mag.mean():.2f} mm, {wall:.0f} s", flush=True)
            except Exception as e:                                    # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[{a.method}] {coh} {c} FAILED: {e}", flush=True)
    print(f"{a.method.upper()} DONE", flush=True)


if __name__ == "__main__":
    main()
