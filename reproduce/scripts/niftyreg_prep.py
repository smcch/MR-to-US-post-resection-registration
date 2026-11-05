"""Prep/convert utilities for the NiftyReg arms of the RESECT TRE eval.

make-rmask: rebuild reg_core's similarity mask (US FoV minus dilated cavity)
    on the 1 mm working grid and save it next to the baseline output, so
    reg_aladin/reg_f3d see the same similarity support as our arms.
def2disp: convert a NiftyReg deformation field (world-mm positions, 5D NIfTI
    (D,H,W,1,3)) evaluated on the working grid into disp_mm.nii.gz with the
    working-grid affine (u = def - x_world), compatible with resect_tre.py.
    NiftyReg's def/aff map REFERENCE(US) world -> FLOATING(FLAIR) world, i.e.
    predicted_mr = def(x_us) = x_us + u(x_us): same semantics as reg_core.
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reg_core import resample_to_grid


def make_rmask(us_grid_path, cavity_path, out_path, dilate_mm=3.0):
    """US FoV minus the cavity dilated by `dilate_mm` MILLIMETRES.

    The dilation is done with an exact Euclidean distance transform sampled at
    the grid's true voxel sizes, so the radius is physical on any grid.  The
    earlier version took the spacing as np.diag(affine)[0], which is the
    projection of the first column onto +x, not its length: correct on an
    axis-aligned isotropic grid (all RESECT calls, which pass the 1 mm working
    grid -- those masks are unchanged by this fix) and meaningless on an
    oblique one.  The ReMIND arm passes the RAW native US grid, which is
    oblique by 18-52 deg, and 11 of its 14 subjects got a wrong radius:
    0.13 mm where diag[0] was negative, and ReMIND-045 (diag[0] = 4e-6)
    requested 159,550 iterations, dilating the exclusion over the whole
    volume.  See scripts/qa_rmask_spacing.py for the per-grid audit.
    """
    from scipy.ndimage import distance_transform_edt
    g = nib.load(str(us_grid_path))
    zooms = np.linalg.norm(g.affine[:3, :3], axis=0)
    fov = np.asarray(g.dataobj) > 0
    img = nib.load(str(cavity_path))
    lab = (np.asarray(img.dataobj) == 1).astype(np.float32)
    ex = resample_to_grid(img, g.affine, fov.shape, mode="nearest", data=lab) > 0.5
    if ex.any() and dilate_mm > 0:
        # distance in mm from every voxel to the nearest excluded voxel
        dist_mm = distance_transform_edt(~ex, sampling=tuple(float(z) for z in zooms))
        ex = dist_mm <= float(dilate_mm)
    mask = (fov & ~ex).astype(np.uint8)
    nib.save(nib.Nifti1Image(mask, g.affine), str(out_path))
    print(f"rmask {out_path}: {int(mask.sum())} vox "
          f"(fov {int(fov.sum())}, excluded {int((fov & ~mask.astype(bool)).sum())}, "
          f"zooms {zooms.round(3).tolist()}, dilate {dilate_mm} mm)")


def def2disp(def_path, out_dir, wall_s=None):
    d = nib.load(str(def_path))
    arr = np.asarray(d.dataobj, dtype=np.float32)
    arr = arr[:, :, :, 0, :] if arr.ndim == 5 else arr
    aff = d.affine
    ii, jj, kk = np.meshgrid(*[np.arange(s, dtype=np.float64)
                               for s in arr.shape[:3]], indexing="ij")
    ones = np.ones_like(ii)
    world = (np.stack([ii, jj, kk, ones], -1) @ aff.T)[..., :3]
    # NiftyReg deformation fields store positions in LPS-flipped world (the
    # nifti sform is honoured but x,y are negated internally); detect and fix:
    # compare median |def - world| under identity vs negXY interpretation.
    u_ras = arr - world
    arr_neg = arr.copy(); arr_neg[..., 0] *= -1; arr_neg[..., 1] *= -1
    u_neg = arr_neg - world
    m_ras = float(np.median(np.abs(u_ras)))
    m_neg = float(np.median(np.abs(u_neg)))
    u = u_ras if m_ras <= m_neg else u_neg
    conv = "ras" if m_ras <= m_neg else "negxy"
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(u.astype(np.float32), aff), out / "disp_mm.nii.gz")
    log = {"source": str(def_path), "def_convention": conv,
           "median_abs_u_mm": min(m_ras, m_neg)}
    if wall_s is not None:
        log["wall_s"] = float(wall_s)
    (out / "log.json").write_text(json.dumps(log, indent=2))
    print(f"disp {out}: convention={conv} median|u|={min(m_ras, m_neg):.2f} mm")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "make-rmask":
        make_rmask(*sys.argv[2:])
    elif cmd == "def2disp":
        def2disp(sys.argv[2], sys.argv[3],
                 float(sys.argv[4]) if len(sys.argv) > 4 else None)
    else:
        raise SystemExit(f"unknown cmd {cmd}")
