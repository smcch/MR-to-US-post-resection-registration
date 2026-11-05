"""Panels of the method on one real case, shared by fig_pipeline.py (schematic with captures), fig_model.py (from the
SynthSeg segmentation to the FE model) and fig_method.py (the former Fig. 1).  The intermediate fields are recomputed
with the frozen functions (same calls as v2_remind_eval.deform) once and cached in paper/build/method_<coh>_<case>.npz.
"""
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap

import fig_lib as L
from fig_lib import plt, C

# SynthSeg label groups (FreeSurfer numbering), for display only
GROUPS = [("white matter", {2, 41}, "#f2f2f2"), ("cortex", {3, 42}, "#c9a66b"), ("deep gray matter", {10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58, 28, 60}, "#8c6bb1"),
          ("ventricles / CSF", {4, 5, 14, 15, 24, 43, 44}, "#3182bd"), ("cerebellum", {7, 8, 46, 47}, "#74c476"), ("brainstem", {16}, "#e6550d")]
LABELS_CSF = {4, 5, 14, 15, 43, 44, 24}                                 # biomech_basis.LABELS_CSF


def compute_cache(coh, c, cache):
    import torch
    import affine_cascade as AC
    import v2_deform as V
    import v2_method as FM
    from reg_core import load_to_grid, normalize, resample_to_grid
    from reg_modal import build_phi
    us, mr, seg, A, wg = C.inputs(coh, c)
    basis = C.RUN / coh / c / "basis30.npz"
    DEV = "cuda"
    cfg, rc = FM.ACCEPTED, FM.ACCEPTED["res"]
    tmp = Path("/tmp/figmethod")
    tmp.mkdir(exist_ok=True)
    AC.relabel(mr, tmp / "moving_rig.nii.gz", A)
    f_arr, m_arr, grid_aff, dims = load_to_grid(us, tmp / "moving_rig.nii.gz", 1.0)
    fov = f_arr > 0
    f_n, m_n = normalize(f_arr, fov), normalize(m_arr, m_arr > 0)
    phi, _ = build_phi(basis, grid_aff, dims, 1.0)
    eig = np.load(str(basis))["eigvals"]
    phi = phi.to(DEV)
    coef, u = V.register_modal_from(f_n, m_n, fov, phi, eig, None, cfg["modal_iters"], 3e-3, 0.1, tuple(cfg["mind"]), DEV)
    usi, mri_img = nib.load(str(us)), nib.load(str(mr))
    aff05, dims05 = V.box05(usi)
    f05 = resample_to_grid(usi, aff05, dims05).astype(np.float32)
    f05 = torch.from_numpy(normalize(f05, f05 > 0))[None, None].to(DEV)
    mw05 = V.warp_mr_on_box(mri_img, A, u, grid_aff, aff05, dims05, DEV)
    mw05 = V.normalize_t(mw05, mw05 > 0)
    corr = V.residual_correspondences(f05, mw05, aff05, [tuple(l) for l in rc["levels"]], mind=tuple(rc["mind"]), coupled=rc["coupled"],
                                      weight=rc["weight"], dev=DEV, symmetric=rc["symmetric"], max_pts=rc["max_pts"])
    z = np.load(str(basis), allow_pickle=True)
    u_nodes, rinfo = V.fem_residual(z, corr, rc["lam_rel"], rc["huber_mm"], trim=rc["trim"])
    r = V.lattice_field(z, u_nodes, grid_aff, dims).to(DEV)
    n = r.norm(dim=1, keepdim=True)
    r = r * (rc["max_mm"] / n.clamp(min=rc["max_mm"]))
    stored = np.asarray(nib.load(str(C.RUN / coh / c / "fer/disp_mm.nii.gz")).dataobj, np.float32)
    tot = np.moveaxis((u + r)[0].cpu().numpy(), 0, -1)
    print(f"recomputed vs stored field: max |diff| {np.abs(tot - stored).max():.3f} mm, mean {np.abs(tot - stored).mean():.4f} mm")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, u_modal=np.moveaxis(u[0].cpu().numpy(), 0, -1).astype(np.float32),
                        r=np.moveaxis(r[0].cpu().numpy(), 0, -1).astype(np.float32), coef=coef.cpu().numpy(),
                        centres=corr["centres_mm"].cpu().numpy().astype(np.float32), disp=corr["disp_mm"].cpu().numpy().astype(np.float32),
                        weight=corr["weight"].cpu().numpy().astype(np.float32), u_nodes=u_nodes.astype(np.float32), grid_aff=grid_aff)


def prepare(coh, c):
    """Everything the panels need: case, slice, cone, lattice in voxel coordinates, cached fields."""
    cache = L.W / f"paper/build/method_{coh}_{c}.npz"
    if not cache.exists():
        compute_cache(coh, c, cache)
    us, mr, seg, A, wg = C.inputs(coh, c)
    basis = C.RUN / coh / c / "basis30.npz"
    zc = np.load(cache)
    d = L.load_case(coh, c)
    k, lm_v, how = L.choose_slice(d)
    shape = d["us"].shape
    us_sl = d["us"][:, :, k]
    cone = us_sl > 0.02
    x0, x1, y0, y1 = L.crop_limits(cone, pad=6)
    inv = np.linalg.inv(d["aff"])
    zb = np.load(str(basis), allow_pickle=True)
    nodes, E, tets, modes = np.asarray(zb["nodes"]), np.asarray(zb["E"]), np.asarray(zb["tets"]), np.asarray(zb["modes"])
    nodeE = np.zeros(len(nodes)); cnt = np.zeros(len(nodes))
    np.add.at(nodeE, tets.ravel(), np.repeat(E, 4)); np.add.at(cnt, tets.ravel(), 1)
    nodeE /= np.maximum(cnt, 1)
    nv = nodes @ inv[:3, :3].T + inv[:3, 3]
    slab = np.abs(nv[:, 2] - k) < 3.0
    bx0, bx1, by0, by1 = nv[slab, 0].min() - 6, nv[slab, 0].max() + 6, nv[slab, 1].min() - 6, nv[slab, 1].max() + 6
    return dict(coh=coh, c=c, d=d, k=k, how=how, shape=shape, us_sl=us_sl, cone=cone, crop=(x0, x1, y0, y1), inv=inv,
                nodes=nodes, nv=nv, slab=slab, E=E, nodeE=nodeE, tets=tets, modes=modes, surface=np.asarray(zb["surface_nodes"]),
                tet_labels=np.asarray(zb["tet_labels"]), brain=(bx0, bx1, by0, by1), zc=zc, A=A, seg=seg, mr=mr,
                h_mm=float(zb["grid_h_mm"]))


# ----------------------------------------------------------------------------------------------- helpers
def inner_bar(ax, mappable, ticks=None, pos=(0.06, 0.055, 0.42, 0.035), fontsize=5.5):
    cax = ax.inset_axes(list(pos))
    cb = plt.colorbar(mappable, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.tick_params(labelsize=fontsize, colors="w", length=2, pad=1)
    cb.outline.set_edgecolor("w")
    cb.outline.set_linewidth(0.4)
    cb.ax.xaxis.set_ticks_position("top")
    return cb


def frame(ax, P, lattice=False):
    lim = P["brain"] if lattice else P["crop"]
    if (P["coh"], P["c"]) in L.ROT180:
        ax.set_xlim(lim[1], lim[0]); ax.set_ylim(lim[3], lim[2])
    else:
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal", adjustable="datalim")
    ax.set_facecolor("k")


def outline(ax, P):
    ax.contour(P["cone"].T.astype(float), levels=[0.5], colors="w", linewidths=0.7, linestyles=":")


def grid_lines(ax, P, u2, colour="w", step=6):
    shape, cone = P["shape"], P["cone"]
    x0, x1, y0, y1 = P["crop"]
    ii, jj = np.arange(shape[0]), np.arange(shape[1])
    for i in range(int(x0), int(x1), step):
        m = cone[i]
        if m.sum() > 2:
            ax.plot((i + u2[i, :, 0])[m], (jj + u2[i, :, 1])[m], color=colour, lw=0.35, alpha=0.75)
    for j in range(int(y0), int(y1), step):
        m = cone[:, j]
        if m.sum() > 2:
            ax.plot((ii + u2[:, j, 0])[m], (j + u2[:, j, 1])[m], color=colour, lw=0.35, alpha=0.75)


def brain_grid(P):
    """World coordinates of the whole axial section of the brain model (indices beyond the working grid are fine)."""
    bx0, bx1, by0, by1 = [int(round(v)) for v in P["brain"]]
    ii, jj = np.meshgrid(np.arange(bx0, bx1 + 1), np.arange(by0, by1 + 1), indexing="ij")
    hom = np.stack([ii, jj, np.full_like(ii, P["k"]), np.ones_like(ii)], -1)
    return (hom @ P["d"]["aff"].T)[..., :3], (bx0 - 0.5, bx1 + 0.5, by0 - 0.5, by1 + 0.5)


# ----------------------------------------------------------------------------------------------- panels
def panel_mr(ax, P, title=None, fs=7.5):
    """Preoperative MR after the initialisation, whole axial section of the model."""
    xw, ext = brain_grid(P)
    img = L.sample_world(P["d"]["mr_img"], C.apply(P["A"], xw))
    ax.imshow(L.norm01(img, img > 0).T, cmap="gray", origin="lower", vmin=0, vmax=1, extent=ext)
    outline(ax, P)
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P, lattice=True)


def panel_synthseg(ax, P, title=None, fs=7.5, legend=True):
    xw, ext = brain_grid(P)
    seg_img = nib.load(str(P["seg"]))
    lab = L.sample_world(seg_img, C.apply(P["A"], xw), order=0).round().astype(int)
    rgb = np.zeros(lab.shape + (3,))
    from matplotlib.colors import to_rgb
    for name, labels, colr in GROUPS:
        rgb[np.isin(lab, list(labels))] = to_rgb(colr)
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=ext, interpolation="nearest")
    outline(ax, P)
    if legend:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=colr, label=name) for name, _, colr in GROUPS], fontsize=4.8, loc="lower right", frameon=True,
                  facecolor="black", edgecolor="none", labelcolor="white", handlelength=1.0, handleheight=0.8, borderpad=0.4)
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P, lattice=True)


def lattice_edges(P):
    """Unique lattice edges with both nodes in the displayed slab, as (N, 2, 2) in-plane voxel coordinates."""
    t = P["tets"]
    pairs = np.concatenate([t[:, [0, 1]], t[:, [0, 2]], t[:, [0, 3]], t[:, [1, 2]], t[:, [1, 3]], t[:, [2, 3]]])
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    both = P["slab"][pairs[:, 0]] & P["slab"][pairs[:, 1]]
    pairs = pairs[both]
    # keep the axis-aligned edges of the lattice (the diagonals of the tetrahedra would fill the picture)
    dv = np.abs(P["nodes"][pairs[:, 0]] - P["nodes"][pairs[:, 1]])
    axis_aligned = (dv > 1e-3).sum(1) == 1
    pairs = pairs[axis_aligned]
    nv = P["nv"]
    return np.stack([nv[pairs[:, 0], :2], nv[pairs[:, 1], :2]], axis=1)


def panel_compartments(ax, P, title=None, fs=7.5, edges=True):
    """The two compartments of the model on the section (parenchyma / CSF and ventricles), the lattice and the surface nodes."""
    xw, ext = brain_grid(P)
    seg_img = nib.load(str(P["seg"]))
    lab = L.sample_world(seg_img, C.apply(P["A"], xw), order=0).round().astype(int)
    comp = np.zeros(lab.shape)
    comp[lab > 0] = 1.0
    comp[np.isin(lab, list(LABELS_CSF))] = 2.0
    cmap = ListedColormap(["black", "#bdbdbd", "#3182bd"])
    ax.imshow(comp.T, cmap=cmap, origin="lower", extent=ext, interpolation="nearest", vmin=0, vmax=2)
    if edges:
        ax.add_collection(LineCollection(lattice_edges(P), colors="#ffd92f", linewidths=0.35, alpha=0.9))
    sm = np.zeros(len(P["nodes"]), bool); sm[P["surface"]] = True
    s = P["slab"] & sm
    ax.plot(P["nv"][s, 0], P["nv"][s, 1], "o", color="#e6550d", ms=1.6, mew=0, label="surface nodes (elastic foundation)")
    outline(ax, P)
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P, lattice=True)


def panel_stiffness(ax, P, title=None, fs=7.5, bar=True):
    nv, slab = P["nv"], P["slab"]
    ax.set_facecolor("k")
    s = ax.scatter(nv[slab, 0], nv[slab, 1], c=P["nodeE"][slab] / 1000, s=3.0, cmap="viridis", rasterized=True)
    outline(ax, P)
    if bar:
        inner_bar(ax, s, ticks=[0.1, 1, 2, 3], pos=(0.05, 0.9, 0.34, 0.035))
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P, lattice=True)


def panel_mode(ax, P, mi, title=None, fs=7.5):
    nv, slab, nodes, modes = P["nv"], P["slab"], P["nodes"], P["modes"]
    ph = modes[:, :, mi]
    ax.set_facecolor("k")
    ax.scatter(nv[slab, 0], nv[slab, 1], c=np.linalg.norm(ph, axis=1)[slab], s=3.0, cmap="magma", rasterized=True)
    ss = slab & (np.arange(len(nodes)) % 7 == 0)
    ax.quiver(nv[ss, 0], nv[ss, 1], ph[ss, 0], ph[ss, 1], color="white", scale=25, width=0.004, alpha=0.9)
    outline(ax, P)
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P, lattice=True)


def panel_fused(ax, P, arm, title=None, fs=7.5):
    d, k = P["d"], P["k"]
    xw = L.world_grid(d["aff"], P["shape"], k)
    L.fuse(ax, L.norm01(L.warped_mr(d, arm, xw)), P["us_sl"], P["cone"])
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P)


def panel_ius(ax, P, title=None, fs=7.5):
    ax.imshow(np.ma.masked_where(~P["cone"], L.norm01(P["us_sl"], P["cone"])).T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_facecolor("k")
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P)


def panel_modal(ax, P, title=None, fs=7.5, bar=True):
    um, k, cone = P["zc"]["u_modal"], P["k"], P["cone"]
    ax.set_facecolor("k")
    mag = np.ma.masked_where(~cone, np.linalg.norm(um[:, :, k], axis=-1))
    im = ax.imshow(mag.T, cmap="magma", origin="lower", vmin=0)
    grid_lines(ax, P, um[:, :, k, :2])
    if bar:
        inner_bar(ax, im)
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P)


def panel_corr(ax, P, title=None, fs=7.5, bar=True, scale=3.0):
    zc, inv, k, cone = P["zc"], P["inv"], P["k"], P["cone"]
    ax.imshow(np.ma.masked_where(~cone, L.norm01(P["us_sl"], cone)).T, cmap="gray", origin="lower", vmin=0, vmax=1)
    ax.set_facecolor("k")
    cv = zc["centres"] @ inv[:3, :3].T + inv[:3, 3]
    near = np.abs(cv[:, 2] - k) < 0.6
    idx = np.where(near)[0]
    keep = idx[(np.round(cv[idx, 0]) % 3 == 0) & (np.round(cv[idx, 1]) % 3 == 0)]
    q = ax.quiver(cv[keep, 0], cv[keep, 1], zc["disp"][keep, 0] * scale, zc["disp"][keep, 1] * scale, zc["weight"][keep], cmap="viridis",
                  angles="xy", scale_units="xy", scale=1, width=0.005, clim=(0, 1))
    if bar:
        inner_bar(ax, q, ticks=[0, 0.5, 1])
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P)
    return len(keep)


def panel_response(ax, P, title=None, fs=7.5, bar=True):
    rr, k, cone = P["zc"]["r"], P["k"], P["cone"]
    ax.set_facecolor("k")
    mag = np.ma.masked_where(~cone, np.linalg.norm(rr[:, :, k], axis=-1))
    im = ax.imshow(mag.T, cmap="magma", origin="lower", vmin=0, vmax=2)
    grid_lines(ax, P, rr[:, :, k, :2])
    if bar:
        inner_bar(ax, im, ticks=[0, 1, 2])
    if title:
        ax.set_title(title, fontsize=fs)
    frame(ax, P)
