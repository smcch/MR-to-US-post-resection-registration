"""Fig. 5 of the FE-refinement manuscript: what each deformable method does to the tissue.  One row per case; first
column the fused result of the proposed pipeline (anatomical context), then the map of log det J (minus its median
over the field of view, so that the constant factor of an affine does not show) with the deformed grid, for
f3d MIND-SSC, f3d LNCC, MCPO, the FE refinement on the affine and the FE refinement on the MCPO pose, all evaluated
on the 1 mm grid of the iUS (the two refinements share that grid, which is defined by the iUS alone).
Usage: python fig_fields.py cohort:case[:slice] [...]   (slice: author's choice, else automatic)
Needs the kept fields: fer_f3d.py --kinds f3d_mind f3d_lncc --keep ..., fer_mcpo.py --keep ... (public sets).
"""
import argparse
import csv
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

import fig_lib as L
from fig_lib import plt, pe, C

sys.path.insert(0, str(L.W / "scripts"))
import fer_mcpo as MC                                                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cases", nargs="+")
ap.add_argument("--stem", default="fig5_fields")
a = ap.parse_args()
T = {(r["cohort"], r["case"]): r for r in csv.DictReader(open(L.W / "fer/tables/PER_CASE.csv"))}
VMAX = 0.6


def mcpo_on_grid(coh, c, aff, shape):
    """MCPO displacement (mm, world axes) on the working grid, after 2x block averaging of its 0.5 mm field."""
    if coh == "remind":
        rid = MC.R2R_ID[c]
        dp = next(r / c / m / f"disp_{rid}_0000_{rid}_0001.nii.gz" for m in ("ceT1", "T2") for r in MC.FAITH
                  if (r / c / m / f"disp_{rid}_0000_{rid}_0001.nii.gz").exists())
    elif coh == "external":
        dp = L.W / f"brat/out/{c}/mcpo/disp.nii.gz"
    else:
        dp = C.FER / "mcpo_fields" / coh / c / "disp_vox.nii.gz"
    ui = nib.load(str(dp))
    u = np.asarray(ui.dataobj, np.float32)
    s = [d - d % 2 for d in u.shape[:3]]
    uc = u[:s[0], :s[1], :s[2]].reshape(s[0] // 2, 2, s[1] // 2, 2, s[2] // 2, 2, 3).mean(axis=(1, 3, 5))
    umm = uc @ ui.affine[:3, :3].T                                   # voxel displacement -> world mm
    affc = ui.affine @ np.array([[2, 0, 0, 0.5], [0, 2, 0, 0.5], [0, 0, 2, 0.5], [0, 0, 0, 1.0]])
    ii, jj, kk = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    w = np.stack([ii, jj, kk], -1).reshape(-1, 3) @ aff[:3, :3].T + aff[:3, 3]
    inv = np.linalg.inv(affc)
    v = w @ inv[:3, :3].T + inv[:3, 3]
    out = np.stack([map_coordinates(umm[..., k], v.T, order=1, mode="nearest") for k in range(3)], -1)
    return out.reshape(*shape, 3)


def logjac(u):
    """log det (I + grad u) for u (X,Y,Z,3) in voxels (= mm) of a 1 mm grid."""
    J = np.stack([np.stack(np.gradient(u[..., c], axis=(0, 1, 2)), -1) for c in range(3)], -2)   # (..., comp, axis)
    J = J + np.eye(3)
    return np.log(np.clip(np.linalg.det(J), 1e-6, None)), np.linalg.det(J)


def non_affine(u, fov):
    """u minus the least-squares affine fit of x -> x + u(x) over the field of view (voxel = mm coordinates)."""
    ijk = np.argwhere(fov)[::7].astype(float)
    y = ijk + u[fov][::7]
    X = np.c_[ijk, np.ones(len(ijk))]
    B, *_ = np.linalg.lstsq(X, y, rcond=None)                       # (4,3)
    ii, jj, kk = np.meshgrid(*[np.arange(n, dtype=float) for n in u.shape[:3]], indexing="ij")
    P = np.stack([ii, jj, kk, np.ones_like(ii)], -1)
    return (P[..., :3] + u) - P @ B


items = L.parse_items(a.cases)
# author's choice (2026-10-03): the figures show only the baseline, the proposed method and MCPO
METH = [("fer", "Affine + FE refinement (proposed)"), ("mcpo", "MCPO")]
fig, axes = plt.subplots(len(items), 3, figsize=(170 * L.MM, 56 * len(items) * L.MM + 16 * L.MM), squeeze=False)
im = None
for r, (coh, c, kf) in enumerate(items):
    d = L.load_case(coh, c)
    k, lm_v, how = L.choose_slice(d, k_fixed=kf)
    shape = d["us"].shape
    fov = d["us"] > 0.02
    xw = L.world_grid(d["aff"], shape, k)
    cone = fov[:, :, k]
    x0, x1, y0, y1 = L.crop_limits(cone)
    row = T[(coh, c)]
    ax = axes[r, 0]
    L.fuse(ax, L.norm01(L.warped_mr(d, "fer", xw)), d["us"][:, :, k], cone)
    ax.set_ylabel(L.case_label(coh, c), fontsize=9)
    if r == 0:
        ax.set_title("iUS on the registered MR\n(proposed method)", fontsize=8.5)
    fields = {"mcpo": mcpo_on_grid(coh, c, d["aff"], shape), "fer": np.asarray(d["u_fer"].dataobj, np.float32)}
    for cc, (key, title) in enumerate(METH, start=1):
        u = fields[key]
        lj, det = logjac(u)
        lj = lj - np.median(lj[fov])
        ax = axes[r, cc]
        img = np.ma.masked_where(~cone, lj[:, :, k])
        ax.set_facecolor("#d9d9d9")
        im = ax.imshow(img.T, cmap="RdBu_r", vmin=-VMAX, vmax=VMAX, origin="lower", interpolation="nearest")
        # deformed grid: where the regular lattice of the slice goes (in-plane components), after removing from every
        # method the affine that best fits its own field over the field of view (so that only the deformation shows)
        up = non_affine(u, fov)[:, :, k, :2]
        ii = np.arange(shape[0])
        jj = np.arange(shape[1])
        for i in range(x0, x1, 5):
            m = cone[i]
            if m.sum() > 2:
                ax.plot((i + up[i, :, 0])[m], (jj + up[i, :, 1])[m], color="k", lw=0.3, alpha=0.55)
        for j in range(y0, y1, 5):
            m = cone[:, j]
            if m.sum() > 2:
                ax.plot((ii + up[:, j, 0])[m], (j + up[:, j, 1])[m], color="k", lw=0.3, alpha=0.55)
        sd = float(np.std(lj[fov]))
        fold = 100.0 * float((det[fov] <= 0).mean())
        t = ax.text(0.03, 0.04, f"SDLogJ {sd:.2f}" + (f"\nfolding {fold:.1f}%" if fold >= 0.05 else ""), transform=ax.transAxes,
                    ha="left", va="bottom", color="k", fontsize=8)
        t.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
        tre_key = key
        tv = row.get(tre_key, "")
        if tv not in ("", None):
            t2 = ax.text(0.97, 0.04, f"{float(tv):.1f} mm", transform=ax.transAxes, ha="right", va="bottom", color="k", fontsize=8)
            t2.set_path_effects([pe.Stroke(linewidth=2, foreground="white"), pe.Normal()])
        if r == 0:
            ax.set_title(title, fontsize=8.5)
        print(f"[{a.stem}] {L.case_label(coh, c)} {key}: SDLogJ here {sd:.3f} (table {row.get(key + '_sdlogj', '')}), fold {fold:.3f} %, "
              f"slice {k} ({how})", flush=True)
    for ax in axes[r]:
        if (coh, c) in L.ROT180:
            ax.set_xlim(x1, x0); ax.set_ylim(y1, y0)
        else:
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(pad=0.3, rect=(0, 0.10, 1, 1))
cax = fig.add_axes([0.30, 0.075, 0.50, 0.018])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("log det J (relative to its median): blue = local compression, red = local expansion", fontsize=8)
cb.ax.tick_params(labelsize=6.5)
fig.savefig(L.OUT / f"{a.stem}.pdf", dpi=300)
fig.savefig(L.OUT / f"{a.stem}.png", dpi=300)
print(a.stem, "ok")
