#!/usr/bin/env python
"""Finite-element (FE) refinement of an initial preoperative-MR to intraoperative-ultrasound (iUS) alignment.

This is the proposed method of the paper, with its single frozen configuration (fe_refinement/v2_method.py,
ACCEPTED), run from explicit file paths.  It reproduces, step by step, what reproduce/scripts/fer_run.py did
for every registration of the paper:

  1. the preoperative MR and its SynthSeg label map are placed in the frame defined by the initialization A
     (header update only, no resampling);
  2. a patient-specific linear-elastic FE model is built from the label map (5 mm tetrahedral lattice, two
     compartments, elastic foundation) and its 30 lowest-energy eigenmodes are computed (biomech_basis.py);
  3. the modal coefficients are fitted to the iUS on a 1 mm working grid with the MIND-SSC similarity;
  4. block correspondences between the iUS and the modally warped MR are turned, through the stiffness matrix
     of the model, into a smooth local displacement bounded to 2 mm (local FE response).

Inputs
  --us      iUS sweep (NIfTI), in the navigation frame.  Voxels outside the field of view must be 0.
  --mr      preoperative MR (NIfTI), in the navigation frame.
  --seg     SynthSeg label map of the MR (same grid as --mr).  See README.md for how to obtain it.
  --affine  4x4 matrix (text, NiftyReg layout) mapping iUS world coordinates to MR world coordinates, i.e. the
            initialization A.  Use affine_init.py for the challenge affine, or an identity matrix if the images
            are already aligned by the navigation system.
  --out     output directory.

Outputs (in --out)
  disp_mm.nii.gz    the refinement u on the 1 mm iUS working grid, as a 3-vector field in world millimeters.
                    The complete transform maps an iUS point x (world mm) to  A (x + u(x))  in the MR.
  meta.json         regularity of the field (folding %, SDLogJ), timings, peak GPU memory, configuration.
  basis30.npz       the FE model (nodes, tetrahedra, stiffness) and its modes.
  synthseg_rig.nii.gz  the label map in the frame of A (used to build the model).

Requirements: a CUDA GPU (about 3 GB), PyTorch, NumPy, SciPy, nibabel (see environment.yml).
Typical runtime: 40 s for the refinement plus about 30 s for the FE model on a desktop GPU.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "fe_refinement"))
import v2_deform as V                                                # noqa: E402
import v2_method as FM                                               # noqa: E402
from reg_core import load_to_grid, normalize, resample_to_grid       # noqa: E402
from reg_modal import build_phi                                      # noqa: E402

TIMES = {}


def timed(name, fn):
    def wrap(*a, **k):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = fn(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        TIMES[name] = TIMES.get(name, 0.0) + time.time() - t0
        return out
    return wrap


def relabel(src, dst, A):
    """Write an image whose world map is x -> src(A x), exactly (no resampling): same voxels, sform = A^-1 . src.
    The qform is left undefined because it cannot represent the shear of an affine A."""
    img = nib.load(str(src))
    P = np.linalg.inv(A) @ img.affine
    out = nib.Nifti1Image(np.asarray(img.dataobj), P, img.header)
    out.set_sform(P, code=1)
    out.set_qform(None, code=0)
    nib.save(out, str(dst))


def build_basis(seg_r, basis, log, h=5.0, n_modes=30):
    """Patient-specific FE model and its lowest-energy modes (CPU; fe_refinement/biomech_basis.py)."""
    with open(log, "a") as fh:
        rc = subprocess.run([sys.executable, str(HERE / "fe_refinement" / "biomech_basis.py"), "--synthseg", str(seg_r),
                             "--out", str(basis), "--h", str(h), "--n-modes", str(n_modes)],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not basis.exists():
        raise RuntimeError(f"FE model failed (rc {rc}); see {log}")


def refine(us, mr, A, basis, out, cfg=FM.ACCEPTED, dev="cuda"):
    """The two steps of the refinement from the initialization A (same code path as the paper's runs)."""
    out = Path(out)
    mv_r = out / "moving_rig.nii.gz"
    relabel(mr, mv_r, A)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    f_arr, m_arr, grid_aff, dims = load_to_grid(us, mv_r, 1.0)
    fov = f_arr > 0
    f_n, m_n = normalize(f_arr, fov), normalize(m_arr, m_arr > 0)
    phi, _ = build_phi(basis, grid_aff, dims, 1.0)
    eig = np.load(str(basis))["eigvals"]
    phi = phi.to(dev)
    # step 1: modal fit (30 coefficients, MIND-SSC similarity, eigenvalue-weighted elastic energy)
    _, u = V.register_modal_from(f_n, m_n, fov, phi, eig, None, cfg["modal_iters"], 3e-3, 0.1, tuple(cfg["mind"]), dev)
    meta = dict(cfg=json.loads(json.dumps(cfg, default=list)), version=FM.VERSION)
    # step 2: block correspondences on the 0.5 mm iUS box -> local FE response through the stiffness matrix
    rc = cfg["res"]
    usi, mri_img = nib.load(str(us)), nib.load(str(mr))
    aff05, dims05 = V.box05(usi)
    f05 = resample_to_grid(usi, aff05, dims05).astype(np.float32)
    f05 = torch.from_numpy(normalize(f05, f05 > 0))[None, None].to(dev)
    mw05 = V.warp_mr_on_box(mri_img, A, u, grid_aff, aff05, dims05, dev)
    mw05 = V.normalize_t(mw05, mw05 > 0)
    corr = V.residual_correspondences(f05, mw05, aff05, [tuple(l) for l in rc["levels"]], mind=tuple(rc["mind"]),
                                      coupled=rc["coupled"], weight=rc["weight"], dev=dev, symmetric=rc["symmetric"],
                                      max_pts=rc["max_pts"])
    z = np.load(str(basis), allow_pickle=True)
    u_nodes, rinfo = V.fem_residual(z, corr, rc["lam_rel"], rc["huber_mm"], trim=rc["trim"])
    r = V.lattice_field(z, u_nodes, grid_aff, dims).to(dev)
    n = r.norm(dim=1, keepdim=True)
    rinfo["clipped_frac"] = float((n > rc["max_mm"]).float().mean())
    r = r * (rc["max_mm"] / n.clamp(min=rc["max_mm"]))          # bound of the local response
    meta["residual"] = rinfo
    u_tot = u + r
    del mw05, f05
    if dev == "cuda":
        torch.cuda.synchronize()
    fold, sdlj = V.regularity(u_tot, fov)
    meta.update(wall_s=time.time() - t0, fold_pct=fold, sdlogj=sdlj,
                peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if dev == "cuda" else None)
    nib.save(nib.Nifti1Image(np.moveaxis(u_tot[0].cpu().numpy(), 0, -1).astype(np.float32), grid_aff),
             str(out / "disp_mm.nii.gz"))
    mv_r.unlink(missing_ok=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--us", required=True)
    ap.add_argument("--mr", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--affine", required=True, help="4x4 text matrix, iUS world -> MR world (identity if aligned)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A = np.loadtxt(a.affine)
    assert A.shape == (4, 4), "the affine must be a 4x4 matrix"
    for p in (a.us, a.mr, a.seg):
        if not Path(p).exists():
            raise FileNotFoundError(p)
    V.register_modal_from = timed("modal_fit_s", V.register_modal_from)
    V.residual_correspondences = timed("correspondences_s", V.residual_correspondences)
    V.fem_residual = timed("fe_response_s", V.fem_residual)
    t_all = time.time()
    seg_r, basis = out / "synthseg_rig.nii.gz", out / "basis30.npz"
    relabel(a.seg, seg_r, A)
    t0 = time.time()
    if not basis.exists():
        build_basis(seg_r, basis, out / "basis.log")
    t_basis = time.time() - t0
    meta = refine(a.us, a.mr, A, basis, out, FM.ACCEPTED, a.device)
    meta.update(basis_s=t_basis, total_s=time.time() - t_all, **{k: float(v) for k, v in TIMES.items()},
                affine=A.tolist(), inputs=dict(us=str(a.us), mr=str(a.mr), seg=str(a.seg)))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[fe_refine] FE model {t_basis:.0f} s, refinement {meta['wall_s']:.0f} s "
          f"(modal {TIMES.get('modal_fit_s', 0):.0f}, correspondences {TIMES.get('correspondences_s', 0):.0f}, "
          f"FE response {TIMES.get('fe_response_s', 0):.0f}); folding {meta['fold_pct']:.3f} %, SDLogJ {meta['sdlogj']:.3f}; "
          f"-> {out / 'disp_mm.nii.gz'}")


if __name__ == "__main__":
    main()
