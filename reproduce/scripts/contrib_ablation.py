"""Round 4, phase A (results/v2/contrib/PROTOCOL.md): which ingredient of the v2 deformable stage matters?
Same pose (shipped, accepted poses), same optimiser, same 30 modal degrees of freedom, same symmetric
block correspondences; only the basis and the prior of the residual change.
  basis    fem      the case's FEM modes (shipped pose basis)
           dct      generic DCT basis: 10 lowest-frequency cosines x 3 directions over the mesh bounding box,
                    each scaled to the mean nodal RMS of the FEM modes and given the FEM eigenvalue spectrum
                    (prior identical in strength; only the SHAPE of the basis changes)
           homog    FEM re-solved with one material (all brain labels -> parenchyma)
           found_lo / found_hi   FEM re-solved with the foundation stiffness x0.1 / x10
  residual K        the frozen v2 residual (v2_deform.fem_residual on the basis file's K)
           lap      the same robust solve with the mesh graph Laplacian instead of K
           none     no residual
The frozen v2 modules are only imported.  Resumable.
Usage: python contrib_ablation.py run --arms C0 C1 ... [--cohorts ...] [--cases ...]
       python contrib_ablation.py score | report
Output: results/v2/contrib/A/<arm>/<cohort>/<case>/{disp_mm.nii.gz,meta.json}, A/PER_CASE.csv, A/REPORT.md
"""
import argparse
import copy
import csv
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rg_cohorts                                                    # noqa
import method_arms as M                                              # noqa
import single_config as SC                                           # noqa
import v2_deform as V                                                # noqa
import v2_method as FM                                               # noqa
from reg_core import load_to_grid, normalize, resample_to_grid       # noqa
from reg_modal import build_phi                                      # noqa
from rs_prestage import relabel                                      # noqa

ROOT = M.R / "v2/contrib/A"
BASES = M.R / "v2/contrib/bases"
DEV = "cuda"
ORDER = ["bite_g4", "resect_after", "resect_before"]
ARMS = {
    "C0": dict(basis="fem", residual="K", lam=10.0),
    "C1": dict(basis="dct", residual="K", lam=10.0),
    "C2_l1": dict(basis="fem", residual="lap", lam=1.0),
    "C2_l10": dict(basis="fem", residual="lap", lam=10.0),
    "C2_l100": dict(basis="fem", residual="lap", lam=100.0),
    "C3_l1": dict(basis="dct", residual="lap", lam=1.0),
    "C3_l10": dict(basis="dct", residual="lap", lam=10.0),
    "C3_l100": dict(basis="dct", residual="lap", lam=100.0),
    "C4": dict(basis="homog", residual="K", lam=10.0),
    "C5": dict(basis="found_lo", residual="K", lam=10.0),
    "C6": dict(basis="found_hi", residual="K", lam=10.0),
    "C7": dict(basis="fem", residual="none", lam=0.0),
    "C8": dict(basis="dct", residual="none", lam=0.0),
}


# ------------------------------------------------------------------- bases
def dct_basis(fem_npz, dst):
    if dst.exists():
        return dst
    z = np.load(str(fem_npz), allow_pickle=True)
    d = {k: z[k] for k in z.files}
    nodes = np.asarray(z["nodes"], float)
    modes_fem = np.asarray(z["modes"], np.float32)
    target_rms = float(np.mean([np.sqrt((modes_fem[:, :, k] ** 2).sum(1).mean()) for k in range(modes_fem.shape[2])]))
    lo, hi = nodes.min(0), nodes.max(0)
    L = np.maximum(hi - lo, 1e-6)
    freqs = sorted(itertools.product(range(3), repeat=3), key=lambda k: (sum(x * x for x in k), k))[:10]
    modes = []
    for k in freqs:
        s = np.prod([np.cos(np.pi * k[a] * (nodes[:, a] - lo[a]) / L[a]) for a in range(3)], axis=0)
        for ax in range(3):
            m = np.zeros_like(nodes)
            m[:, ax] = s
            m *= target_rms / max(np.sqrt((m ** 2).sum(1).mean()), 1e-9)
            modes.append(m)
    d["modes"] = np.stack(modes, -1).astype(np.float32)              # (n, 3, 30)
    d["eigvals"] = np.sort(np.asarray(z["eigvals"], float))[:d["modes"].shape[2]]
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(dst), **d)
    return dst


def solved_basis(kind, coh, c):
    """Re-solve the FEM basis from the shipped pose's relabelled SynthSeg with one change."""
    dst = BASES / kind / coh / f"{c}.npz"
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    seg = M.ROOT / coh / c / "pose/synthseg_rig.nii.gz"
    extra = []
    if kind == "homog":
        img = nib.load(str(seg))
        a = np.asarray(img.dataobj)
        h = np.where(a > 0, 2, 0).astype(a.dtype)                    # every brain label -> parenchyma (2)
        segh = dst.parent / f"{c}_synthseg_homog.nii.gz"
        nib.save(nib.Nifti1Image(h, img.affine, img.header), str(segh))
        seg = segh
    elif kind == "found_lo":
        extra = ["--k-found-frac", "0.002"]
    elif kind == "found_hi":
        extra = ["--k-found-frac", "0.2"]
    with open(dst.parent / f"{c}.log", "w") as f:
        rc = subprocess.run([SC.PY, str(SC.S / "biomech_basis.py"), "--synthseg", str(seg), "--out", str(dst), "--h", "5.0",
                             "--n-modes", "30"] + extra, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not dst.exists():
        raise RuntimeError(f"basis {kind} failed rc={rc}")
    return dst


def basis_for(kind, coh, c):
    fem = M.ROOT / coh / c / "pose/basis30.npz"
    if kind == "fem":
        return fem
    if kind == "dct":
        return dct_basis(fem, BASES / "dct" / coh / f"{c}.npz")
    return solved_basis(kind, coh, c)


# --------------------------------------------------------------- residuals
def laplacian_residual(z, corr, lam_rel=10.0, huber_mm=1.0, iters=6):
    """v2_deform.fem_residual with the mesh graph Laplacian (x I3) as the prior instead of the stiffness K;
    same data term, robust weights, normalisation (median diagonal) and lambda convention."""
    tets = np.asarray(z["tets"], int)
    n = len(np.asarray(z["nodes"]))
    e = np.concatenate([tets[:, [i, j]] for i, j in itertools.combinations(range(4), 2)])
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(float)
    Lg = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    L3 = sp.kron(Lg, sp.eye(3)).tocsr()
    Ln = (L3 / float(np.median(L3.diagonal())) + 1e-6 * sp.eye(3 * n)).tocsr()
    pts = corr["centres_mm"].cpu().numpy().astype(float)
    d = corr["disp_mm"].cpu().numpy().astype(float)
    w0 = corr["weight"].cpu().numpy().astype(float)
    H, good = V.lattice_H(z, pts)
    H, d, w0 = H[np.repeat(good, 3)], d[good], w0[good]
    u = np.zeros(3 * n)
    w = w0.copy()
    for _ in range(iters):
        W = sp.diags(np.repeat(w, 3))
        HtW = H.T @ W
        G = (HtW @ H).tocsr()
        gd = G.diagonal()
        cov = gd > 0
        lam = lam_rel * float(gd[cov].mean()) if cov.any() else 1.0
        Amat = (G + lam * Ln).tocsc()
        b = HtW @ d.reshape(-1)
        Minv = sp.diags(1.0 / np.maximum(Amat.diagonal(), 1e-12))
        try:
            u, _ = spla.cg(Amat, b, x0=u, M=Minv, rtol=1e-6, maxiter=2000)
        except TypeError:
            u, _ = spla.cg(Amat, b, x0=u, M=Minv, tol=1e-6, maxiter=2000)
        res = np.linalg.norm((H @ u).reshape(-1, 3) - d, axis=1)
        w = w0 * np.minimum(1.0, huber_mm / np.maximum(res, 1e-6))
    return u.reshape(-1, 3), dict(n=int(len(res)), med_res_mm=float(np.median(res)))


# ------------------------------------------------------------------- run
def run_case(name, spec_arm, coh, c, holdout=None):
    tag = name + (f"_H{holdout}" if holdout else "")
    out = ROOT / tag / coh / c
    if (out / "disp_mm.nii.gz").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(FM.ACCEPTED)
    spec = rg_cohorts.install(coh)
    fx, mv = spec["paths"](c)[:2]
    A = SC.pose_of(coh, c)
    basis = basis_for(spec_arm["basis"], coh, c)
    tmpd = Path("/tmp/contrib_mv")
    tmpd.mkdir(exist_ok=True)
    mv_r = tmpd / f"{tag}_{coh}_{c}_moving_rig.nii.gz"            # local SSD: G: (SMR) is slow for scratch writes
    relabel(mv, mv_r, A)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    f_arr, m_arr, grid_aff, dims = load_to_grid(fx, mv_r, 1.0)
    fov = f_arr > 0
    meta = dict(arm=tag, cohort=coh, case=c, spec=spec_arm)
    hp = None
    if holdout:
        hmask, hp = V.deep_holdout(fov, grid_aff, holdout / 100)
        f_arr = np.where(hmask, 0.0, f_arr).astype(np.float32)
        fov = f_arr > 0
        meta["holdout"] = hp
    f_n, m_n = normalize(f_arr, fov), normalize(m_arr, m_arr > 0)
    phi, _ = build_phi(basis, grid_aff, dims, 1.0)
    eig = np.load(str(basis))["eigvals"]
    phi = phi.to(DEV)
    _, u = V.register_modal_from(f_n, m_n, fov, phi, eig, None, cfg["modal_iters"], 3e-3, 0.1, tuple(cfg["mind"]), DEV)
    if spec_arm["residual"] != "none":
        rc = cfg["res"]
        usi, mri_img = nib.load(str(fx)), nib.load(str(mv))
        aff05, dims05 = V.box05(usi)
        f05 = resample_to_grid(usi, aff05, dims05).astype(np.float32)
        if hp is not None:
            f05 = np.where(V.holdout_flags_grid(f05 > 0, aff05, hp), 0.0, f05).astype(np.float32)
        f05 = torch.from_numpy(normalize(f05, f05 > 0))[None, None].to(DEV)
        mw05 = V.warp_mr_on_box(mri_img, A, u, grid_aff, aff05, dims05, DEV)
        mw05 = V.normalize_t(mw05, mw05 > 0)
        corr = V.residual_correspondences(f05, mw05, aff05, [tuple(l) for l in rc["levels"]], mind=tuple(rc["mind"]),
                                          coupled=rc["coupled"], weight=rc["weight"], dev=DEV, symmetric=rc["symmetric"],
                                          max_pts=rc["max_pts"])
        z = np.load(str(basis), allow_pickle=True)
        if spec_arm["residual"] == "K":
            u_nodes, rinfo = V.fem_residual(z, corr, spec_arm["lam"], rc["huber_mm"], trim=rc["trim"])
        else:
            u_nodes, rinfo = laplacian_residual(z, corr, spec_arm["lam"], rc["huber_mm"])
        r = V.lattice_field(z, u_nodes, grid_aff, dims).to(DEV)
        nrm = r.norm(dim=1, keepdim=True)
        r = r * (rc["max_mm"] / nrm.clamp(min=rc["max_mm"]))
        meta["residual"] = {k: v for k, v in rinfo.items() if not isinstance(v, (list, dict))}
        u = u + r
        del mw05, f05
    torch.cuda.synchronize()
    fold, sdlj = V.regularity(u, fov)
    meta.update(wall_s=time.time() - t0, peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20, fold_pct=fold, sdlogj=sdlj)
    nib.save(nib.Nifti1Image(np.moveaxis(u[0].cpu().numpy(), 0, -1).astype(np.float32), grid_aff), str(out / "disp_mm.nii.gz"))
    (out / "meta.json").write_text(json.dumps(meta))
    mv_r.unlink(missing_ok=True)
    del phi, u
    torch.cuda.empty_cache()
    print(f"[A] {tag} {coh} {c}: {meta['wall_s']:.0f} s, fold {fold:.3f} %, SDLogJ {sdlj:.3f}", flush=True)


def accepted(cohorts, only):
    S = M.scores()
    return [(coh, c) for coh in cohorts for c in sorted(k for (k2, k) in S if k2 == coh)
            if not S[(coh, c)]["abstained"] and (not only or c in only)]


def do_score():
    rows = []
    for d in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for coh in ORDER:
            for cd in sorted((d / coh).glob("*")) if (d / coh).exists() else []:
                if not (cd / "disp_mm.nii.gz").exists():
                    continue
                c = cd.name
                meta = json.loads((cd / "meta.json").read_text())
                tagp, loader = SC.TAG[coh](c)
                mri, us = loader(str(tagp))
                mri, us = np.asarray(mri, float), np.asarray(us, float)
                A = SC.pose_of(coh, c)
                e = np.linalg.norm((us + SC.disp_at(cd / "disp_mm.nii.gz", us)) @ A[:3, :3].T + A[:3, 3] - mri, axis=1)
                e0 = np.linalg.norm(us @ A[:3, :3].T + A[:3, 3] - mri, axis=1)
                row = dict(arm=d.name, cohort=coh, case=c, tre=float(e.mean()), pose_only=float(e0.mean()),
                           fold=meta["fold_pct"], sdlogj=meta["sdlogj"], tre_hidden=np.nan, pose_hidden=np.nan, n_hidden=0)
                if "holdout" in meta:
                    hid = V.holdout_flags(us, meta["holdout"])
                    if hid.any():
                        row.update(tre_hidden=float(e[hid].mean()), pose_hidden=float(e0[hid].mean()), n_hidden=int(hid.sum()))
                rows.append(row)
    ROOT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "PER_CASE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"scored {len(rows)} rows")


def do_report():
    rows = list(csv.DictReader(open(ROOT / "PER_CASE.csv")))
    by = {}
    for r in rows:
        by.setdefault(r["arm"], {})[(r["cohort"], r["case"])] = r
    ref = by.get("C0", {})
    L = ["# Phase A — ingredient ablation of the v2 deformable stage (46 accepted poses, shipped pose)", "",
         "| arm | n | TRE | pose-only | Δ vs C0 | p | better than C0 | Δ BITE / post / pre | fold % max | SDLogJ mean |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in sorted(by, key=lambda a: (a.split("_H")[0], a)):
        X = by[arm]
        ks = sorted(X)
        t = np.array([float(X[k]["tre"]) for k in ks])
        po = np.array([float(X[k]["pose_only"]) for k in ks])
        line = f"| {arm} | {len(ks)} | {t.mean():.3f} | {po.mean():.3f} |"
        common = [k for k in ks if k in ref]
        if arm != "C0" and "_H" not in arm and len(common) > 1:
            x = np.array([float(X[k]["tre"]) for k in common]); y = np.array([float(ref[k]["tre"]) for k in common])
            p = stats.wilcoxon(x, y, zero_method="zsplit").pvalue
            dc = [np.mean([float(X[k]["tre"]) - float(ref[k]["tre"]) for k in common if k[0] == coh] or [np.nan]) for coh in ORDER]
            line += f" {np.mean(x - y):+.3f} | {p:.4f} | {int((x < y).sum())}/{len(common)} | {dc[0]:+.3f} / {dc[1]:+.3f} / {dc[2]:+.3f} |"
        else:
            line += " — | — | — | — |"
        line += f" {max(float(X[k]['fold']) for k in ks):.3f} | {np.mean([float(X[k]['sdlogj']) for k in ks]):.3f} |"
        L.append(line)
    hold = [a for a in by if "_H" in a]
    if hold:
        L += ["", "Hold-out (deep 30 % of the FoV hidden): TRE on HIDDEN landmarks", "", "| arm | cases | pose-only hidden | arm hidden | worse than pose |", "|---|---|---|---|---|"]
        for arm in sorted(hold):
            rr = [r for r in by[arm].values() if int(r["n_hidden"]) > 0]
            th = np.array([float(r["tre_hidden"]) for r in rr]); ph = np.array([float(r["pose_hidden"]) for r in rr])
            L.append(f"| {arm} | {len(rr)} | {ph.mean():.3f} | {th.mean():.3f} | {int((th > ph).sum())}/{len(rr)} |")
    (ROOT / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "score", "report"])
    ap.add_argument("--arms", nargs="*", default=["C0"])
    ap.add_argument("--cohorts", nargs="*", default=ORDER)
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--holdout", type=int, default=0, help="hide the deep N %% of the FoV (0 = off)")
    a = ap.parse_args()
    if a.mode == "score":
        return do_score()
    if a.mode == "report":
        return do_report()
    for name in a.arms:
        for coh, c in accepted(a.cohorts, a.cases):
            try:
                run_case(name, ARMS[name], coh, c, a.holdout or None)
            except Exception as ex:                                  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"[A] {name} {coh} {c} FAILED: {ex}", flush=True)
                torch.cuda.empty_cache()
    print("PHASE A RUN DONE", flush=True)


if __name__ == "__main__":
    main()
