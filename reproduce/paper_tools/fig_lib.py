"""Shared rendering helpers of the FE-refinement manuscript's image figures (fusion style of the previous manuscript
version: warped MR in grey, iUS on top in a hot-metal colour map with opacity following echogenicity)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe                                  # noqa: E402
import matplotlib.pyplot as plt                                      # noqa: E402
import nibabel as nib                                                # noqa: E402
import numpy as np                                                   # noqa: E402
from scipy.ndimage import binary_dilation, map_coordinates           # noqa: E402

W = Path("/mnt/c/ReMIND_work")
sys.path.insert(0, str(W / "scripts"))
import fer_common as C                                               # noqa: E402

OUT = W / "paper/MIA/imagenes"
MM = 1 / 25.4
DOT, MAG = "#00E68A", "#FF4FD8"
HALO = [pe.Stroke(linewidth=3.2, foreground="black"), pe.Normal()]
CMAP_US = plt.get_cmap("hot")
G = Path("/mnt/g/ECOGRAFIA/ReMIND_biomech")                          # read-only: cavity / tumour masks for the slice choice
ROT180 = {("resect_after", "Case4"), ("resect_before", "Case4")}     # navigation frame rotated 180 deg about S
NAME = {"resect_after": "RESECT", "bite_g4": "BITE", "resect_before": "RESECT pre", "remind": "ReMIND", "external": "External"}


def case_label(coh, c):
    return f"{NAME[coh]} {c.replace('Case', '').replace('ReMIND-', '').replace('Subject-', 'S')}"   # external: S9 as in Table S1


def world_grid(aff, shape, k):
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    hom = np.stack([ii, jj, np.full_like(ii, k), np.ones_like(ii)], -1)
    return (hom @ aff.T)[..., :3]


_ARR = {}


def arr(img):
    """Decompressed voxel array of a loaded image, read once (nibabel re-reads .nii.gz at every np.asarray)."""
    k = id(img)
    if k not in _ARR:
        _ARR[k] = (img, np.asarray(img.dataobj, np.float32))
    return _ARR[k][1]


def sample_world(img, pts, order=1):
    inv = np.linalg.inv(img.affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    return map_coordinates(arr(img), [v[..., 0], v[..., 1], v[..., 2]], order=order, mode="constant", cval=0.0)


def disp_at(uimg, pts):
    inv = np.linalg.inv(uimg.affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    a = arr(uimg)
    return np.stack([map_coordinates(a[..., c], [v[..., 0], v[..., 1], v[..., 2]], order=1, mode="nearest") for c in range(3)], -1)


def norm01(a, mask=None):
    m = mask if mask is not None else a > 0
    if not m.any():
        return a
    lo, hi = np.percentile(a[m], [2, 99])
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)


def fuse(ax, mr_n, us_sl, cone):
    ax.imshow(mr_n.T, cmap="gray", origin="lower", vmin=0, vmax=1)
    w = np.clip(norm01(us_sl, cone).T, 0, 1)
    rgba = CMAP_US(np.clip(0.15 + 0.85 * w, 0, 1))
    rgba[..., 3] = np.where(cone.T, w ** 0.7 * 0.8, 0.0)
    ax.imshow(rgba, origin="lower", interpolation="bilinear")
    ax.contour(cone.T.astype(float), levels=[0.5], colors="w", linewidths=0.5, alpha=0.6, linestyles=":")


def load_case(coh, c, f3d_field=None):
    """Everything needed to render one case: working grid (iUS at 1 mm), MR, landmarks, affine, fields."""
    us, mr, _, A, wg = C.inputs(coh, c)
    g = nib.load(str(wg))
    mri, usl, _ = C.landmarks(coh, c)
    d = dict(coh=coh, case=c, us=np.asarray(g.dataobj, np.float32), aff=g.affine, mr_img=nib.load(str(mr)), lm_mr=mri, lm_us=usl,
             A=A, u_fer=nib.load(str(C.RUN / coh / c / "fer/disp_mm.nii.gz")))
    f = f3d_field or (C.FER / "f3d/fields/f3d_mind" / coh / c / "disp_mm.nii.gz")
    d["u_f3d"] = nib.load(str(f)) if Path(f).exists() else None
    # second initialisation: MCPO pose + the same refinement (fer_mcpo_pose.py), and MCPO's own field
    pz = C.FER / "mcpo_pose" / coh / c / "pose.txt"
    fz = C.FER / "run_mcpopose" / coh / c / "fer/disp_mm.nii.gz"
    d["P"] = np.loadtxt(pz) if pz.exists() else None
    d["u_mp"] = nib.load(str(fz)) if fz.exists() else None
    d["u_mcpo"] = mcpo_field(coh, c)
    return d


def mcpo_field(coh, c):
    """MCPO's displacement field as written by the official script (voxel units of its own grid), or None."""
    import fer_mcpo as MC
    if coh == "remind" and c not in MC.R2R_ID:                       # ReMIND-009: run on our box (fer_mcpo.remind_own_case)
        dp = C.FER / "mcpo_fields" / coh / c / "disp_vox.nii.gz"
    elif coh == "remind":
        rid = MC.R2R_ID[c]
        dp = next((r / c / m / f"disp_{rid}_0000_{rid}_0001.nii.gz" for m in ("ceT1", "T2") for r in MC.FAITH
                   if (r / c / m / f"disp_{rid}_0000_{rid}_0001.nii.gz").exists()), None)
    elif coh == "external":
        dp = W / f"brat/out/{c}/mcpo/disp.nii.gz"
    else:
        dp = C.FER / "mcpo_fields" / coh / c / "disp_vox.nii.gz"
    return nib.load(str(dp)) if dp is not None and Path(dp).exists() else None


def mcpo_disp_mm(uimg, pts):
    """MCPO displacement (world mm) at world points: voxel displacement interpolated on its grid, rotated to world."""
    return disp_at(uimg, pts) @ uimg.affine[:3, :3].T


def cavity_mask(coh, c):
    """(image, labels or None) used only to choose the slice: resection cavity (RESECT-SEG; ReMIND iUS segmentation,
    label 2) or preoperative tumour (BITE group 2; group-4 folder k = patient k+1)."""
    if coh == "resect_after":
        p = G / f"data/resect/{c}/{c}-US-after-cavity.nii.gz"
        return (p, None) if p.exists() else None
    if coh == "bite_g4":
        k = f"{int(c) + 1:02d}"
        p = G / f"data/bite/group2/{k}/{k}_tumor_tal.nii.gz"
        return (p, None) if p.exists() else None
    if coh == "remind":
        p = W / f"remind/segs/{c}-post-us-segmentation.nii.gz"
        return (p, [2]) if p.exists() else None
    return None


def parse_items(cases):
    """'cohort:case' or 'cohort:case:k' (k = axial slice chosen by the author) -> [(cohort, case, k or None)]."""
    out = []
    for x in cases:
        p = x.split(":")
        out.append((p[0], p[1], int(p[2]) if len(p) > 2 else None))
    return out


def choose_slice(d, min_pairs=3, k_fixed=None):
    """Axial slice of the working grid: among the slices with at least half the largest cavity / tumour area inside
    the cone, the one with most landmark pairs within 6 mm; without a mask, the slice with most iUS coverage among
    those with at least `min_pairs` pairs within 6 mm.  `k_fixed`: the author's choice (2026-10-05: slices that show
    the resection cavity and in which the landmarks of the proposed method clearly improve; see _contact_sheet.py)."""
    inv = np.linalg.inv(d["aff"])
    lm_v = d["lm_us"] @ inv[:3, :3].T + inv[:3, 3]
    if k_fixed is not None:
        return int(k_fixed), lm_v, "author"
    nz = np.unique(np.where(d["us"] > 0.02)[2])
    cav = cavity_mask(d["coh"], d["case"])
    if cav is not None:
        cimg = nib.load(str(cav[0]))
        ca = np.asarray(cimg.dataobj)
        cm = nib.Nifti1Image(((ca > 0) if cav[1] is None else np.isin(ca, cav[1])).astype(np.float32), cimg.affine)
        area = {int(k): float(((sample_world(cm, world_grid(d["aff"], d["us"].shape, k), order=0) > 0.5) & (d["us"][:, :, k] > 0.02)).sum())
                for k in nz}
        amax = max(area.values())
        if amax > 0:
            cand = [k for k in area if area[k] >= 0.5 * amax]
            return int(max(cand, key=lambda k: ((np.abs(lm_v[:, 2] - k) < 6).sum(), area[k]))), lm_v, "cavity"

    def coverage(k):
        sl = d["us"][:, :, k] > 0.02
        n = int(sl.sum())
        if n < 500:
            return 0.0
        ii, jj = np.where(binary_dilation(sl, iterations=14))
        return n / float((ii.max() - ii.min() + 8) * (jj.max() - jj.min() + 8))
    cand = [int(k) for k in nz if (np.abs(lm_v[:, 2] - k) < 6).sum() >= min_pairs] or [int(k) for k in nz]
    return int(max(cand, key=lambda k: ((np.abs(lm_v[:, 2] - k) < 6).sum(), coverage(k)))), lm_v, "landmarks"


def warped_mr(d, arm, xw):
    A = d["A"]
    if arm == "initial":
        return sample_world(d["mr_img"], xw)
    if arm == "affine":
        return sample_world(d["mr_img"], C.apply(A, xw))
    if arm == "f3d":
        return sample_world(d["mr_img"], xw + disp_at(d["u_f3d"], xw))
    if arm == "mcpo":
        return sample_world(d["mr_img"], xw + mcpo_disp_mm(d["u_mcpo"], xw))
    if arm == "mcpo_pose":
        return sample_world(d["mr_img"], C.apply(d["P"], xw))
    if arm == "mcpo_pose_fer":
        return sample_world(d["mr_img"], C.apply(d["P"], xw + disp_at(d["u_mp"], xw)))
    return sample_world(d["mr_img"], C.apply(A, xw + disp_at(d["u_fer"], xw)))


def pullback(d, arm, y):
    """MR landmark y brought to iUS coordinates through the inverse of the arm's transform (fixed-point)."""
    A = d["A"]
    if arm == "initial":
        return y
    Ai = np.linalg.inv(A)
    if arm == "affine":
        return C.apply(Ai, y)
    if arm == "f3d":
        x = C.apply(Ai, y)
        for _ in range(6):
            x = x + (y - (x + disp_at(d["u_f3d"], x[None])[0]))
        return x
    if arm == "mcpo":
        x = y.copy()
        for _ in range(8):
            x = y - mcpo_disp_mm(d["u_mcpo"], x[None])[0]
        return x
    if arm in ("mcpo_pose", "mcpo_pose_fer"):
        z = C.apply(np.linalg.inv(d["P"]), y)
        if arm == "mcpo_pose":
            return z
        x = z.copy()
        for _ in range(6):
            x = z - disp_at(d["u_mp"], x[None])[0]
        return x
    z = C.apply(Ai, y)
    x = z.copy()
    for _ in range(6):
        x = z - disp_at(d["u_fer"], x[None])[0]
    return x


def draw_pairs(ax, d, arm, lm_v, near):
    inv = np.linalg.inv(d["aff"])
    for i in np.where(near)[0]:
        pu = lm_v[i]
        pm = pullback(d, arm, d["lm_mr"][i])
        pm = pm @ inv[:3, :3].T + inv[:3, 3]
        ln, = ax.plot([pu[0], pm[0]], [pu[1], pm[1]], "-", color=MAG, lw=1.6)
        ln.set_path_effects(HALO)
        p1, = ax.plot(pu[0], pu[1], "o", color=DOT, ms=6, mew=0)
        p1.set_path_effects(HALO)
        p2, = ax.plot(pm[0], pm[1], "x", color=MAG, ms=7, mew=2.0)
        p2.set_path_effects(HALO)


def crop_limits(cone, pad=4):
    ii, jj = np.where(binary_dilation(cone, iterations=14))
    return max(ii.min() - pad, 0), min(ii.max() + pad, cone.shape[0]), max(jj.min() - pad, 0), min(jj.max() + pad, cone.shape[1])
