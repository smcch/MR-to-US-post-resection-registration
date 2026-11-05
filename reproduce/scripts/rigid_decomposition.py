"""rigid_decomposition.py -- decompose landmark displacement fields into
rigid + non-rigid components, per case and per dataset.

Question: our modal basis is built on an elastic foundation, which removes the
rigid-body null space.  Any BULK RIGID misalignment (e.g. from ReMIND's
skin-surface ICP patient-to-image registration) is therefore outside the
method's search space by construction.  If ReMIND's initial landmark
displacement is rigid-dominated and RESECT's is not, that explains part of the
TRE gap and motivates an optional rigid pre-stage.

Conventions
-----------
Both datasets use fixed = intra-operative US, moving = pre-op MR, and the
registration field u(x) maps a FIXED (US world RAS mm) point to its predicted
MOVING (MR world) position:  pred = x_us + u(x_us);  TRE = |pred - x_mr|.
Every transform fitted here therefore maps US points -> MR points, the same
direction as u.  (Kabsch is direction-symmetric: rotation angle and residual
RMS of the optimal rigid are identical for the inverse fit; only the raw
translation vector differs, which is why we report centroid-referenced
quantities.)

Per case we fit by least squares on the landmark pairs
  * translation only (3 DoF)   * rigid (6 DoF, Kabsch)   * affine (12 DoF)
and report the residual after each, in-sample and leave-one-out
cross-validated, plus the fraction of the initial error explained.

Because a 6/12-DoF fit soaks up part of any random noise when n is small, a
NULL model is simulated per case (isotropic Gaussian displacement at the same
landmark positions, matched RMS) giving the explained fraction expected by
chance.

Datasets
  RESECT        20 cases with CaseN-MRI-afterUS.tag (TUNE/TEST split flagged)
  ReMIND_clean  ReMIND-BS v1.1 (release/landmarks/ReMIND-BS, 14 subj/157 pairs)
  ReMIND_v10    ReMIND-BS v1.0 archive (155 pairs) -- continuity with
                results/landmark_qc/clean_TRE_table.csv
  ReMIND_unfilt same 14 subjects, all 172 P3 pairs, QA filter OFF -- control
  Cube_P1       oracle-prealigned cube protocol (positive control: its rigid
                component was removed by construction)

Outputs (results/rigid_decomposition/).
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.stats import mannwhitneyu, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resect_tags import load_tag_pairs

WD = Path("/mnt/c/ReMIND_work")
OUT = WD / "results" / "rigid_decomposition"
RESECT_RAW = Path("/mnt/g/ECOGRAFIA/ReMIND_downloads/RESECT/NIFTI")
RESECT_STAGED = WD / "data" / "resect"
NR_RIGID = OUT / "niftyreg_rigid"
# Frozen copy of the ReMIND-BS release + QA diagnostics (the QA pipeline was
# regenerating them while this analysis ran); see landmarks_snapshot/
# MANIFEST.sha256 and rule_application.json for the exact version.
SNAP = OUT / "landmarks_snapshot"

PIDS = ["ReMIND-003", "ReMIND-023", "ReMIND-031", "ReMIND-032", "ReMIND-034",
        "ReMIND-045", "ReMIND-056", "ReMIND-077", "ReMIND-079", "ReMIND-087",
        "ReMIND-091", "ReMIND-096", "ReMIND-107", "ReMIND-109"]
RESECT_TEST = [2, 4, 6, 8, 12, 14, 16, 18, 24, 25]
RESECT_EXTREME = [12, 25]
MRC = ["mr_x_RAS_mm", "mr_y_RAS_mm", "mr_z_RAS_mm"]
USC = ["us_x_RAS_mm", "us_y_RAS_mm", "us_z_RAS_mm"]
RNG = np.random.default_rng(20260810)
NULL_REPS = 400


# ---------------------------------------------------------------- fits ----
def fit_rigid(X, Y):
    """Kabsch: R, t minimising |R X + t - Y| (no scaling)."""
    xm, ym = X.mean(0), Y.mean(0)
    H = (X - xm).T @ (Y - ym)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, ym - R @ xm


def fit_affine(X, Y):
    M, *_ = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y, rcond=None)
    return M[:3].T.copy(), M[3].copy()


def apply_lin(A, b, X):
    return X @ A.T + b


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def rms(v):
    return float(np.sqrt(np.mean(np.sum(np.asarray(v) ** 2, axis=1))))


def meannorm(v):
    return float(np.mean(np.linalg.norm(np.asarray(v), axis=1)))


def loo_residual(X, Y, kind):
    """Leave-one-out cross-validated residual norms (honest, not in-sample)."""
    n = len(X)
    if n - 1 < (3 if kind == "rigid" else 4):
        return None
    res = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        if kind == "rigid":
            R, t = fit_rigid(X[m], Y[m])
            p = R @ X[i] + t
        else:
            A, b = fit_affine(X[m], Y[m])
            p = A @ X[i] + b
        res[i] = np.linalg.norm(p - Y[i])
    return res


def knn_loo_residual(X, D, k=3):
    """Leave-one-out inverse-distance-weighted prediction of each landmark's
    displacement from its k nearest neighbours.  A non-parametric measure of
    how NON-SMOOTH the sampled displacement field is at short range: it bounds
    (annotation noise + genuine fine-scale deformation) from above and is
    computed identically for every dataset.  Returns (residual norms, mean
    nearest-neighbour distance mm)."""
    n = len(X)
    if n < k + 1:
        return None, np.nan
    dm = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(dm, np.inf)
    res = np.empty(n)
    nnd = np.empty(n)
    for i in range(n):
        o = np.argsort(dm[i])[:k]
        w = 1.0 / np.maximum(dm[i][o], 1e-6)
        res[i] = np.linalg.norm((w[:, None] * D[o]).sum(0) / w.sum() - D[i])
        nnd[i] = dm[i][o[0]]
    return res, float(nnd.mean())


def null_explained(X, sigma_rms, reps=NULL_REPS):
    """Explained fraction expected by chance: isotropic noise of the same RMS
    at the same landmark positions, fitted with the same 6/12 DoF."""
    n = len(X)
    s = sigma_rms / np.sqrt(3.0)
    fr, fa = [], []
    for _ in range(reps):
        D = RNG.normal(0.0, s, size=(n, 3))
        Y = X + D
        r0 = rms(D)
        if r0 == 0:
            continue
        R, t = fit_rigid(X, Y)
        fr.append(1.0 - rms(apply_lin(R, t, X) - Y) / r0)
        if n >= 4:
            A, b = fit_affine(X, Y)
            fa.append(1.0 - rms(apply_lin(A, b, X) - Y) / r0)
    f = lambda v: (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)
    return f(fr) + f(fa)


def decompose(case, dataset, X, Y, extra=None):
    """X = US (fixed) points, Y = MR (moving) points, world RAS mm."""
    n = len(X)
    D0 = Y - X
    r0, m0 = rms(D0), meannorm(D0)

    t = D0.mean(axis=0)
    res_t = D0 - t

    R, tr = fit_rigid(X, Y)
    pred_r = apply_lin(R, tr, X)
    res_r = pred_r - Y
    rot_part = (X - X.mean(0)) @ (R - np.eye(3)).T

    if n >= 4:
        A, b = fit_affine(X, Y)
        res_a = apply_lin(A, b, X) - Y
        sv = np.linalg.svd(A, compute_uv=False)
        detA = float(np.linalg.det(A))
    else:
        res_a = np.full_like(D0, np.nan)
        sv, detA = np.full(3, np.nan), np.nan

    loo_r = loo_residual(X, Y, "rigid")
    loo_a = loo_residual(X, Y, "affine")
    knn, nnd = knn_loo_residual(X, D0)
    nr_m, nr_s, na_m, na_s = null_explained(X, r0)
    lr = float(np.sqrt(np.mean(loo_r ** 2))) if loo_r is not None else np.nan
    la = float(np.sqrt(np.mean(loo_a ** 2))) if loo_a is not None else np.nan

    row = {
        "dataset": dataset, "case": case, "n_pairs": n,
        "initial_mean_mm": m0, "initial_rms_mm": r0,
        "trans_only_shift_mm": float(np.linalg.norm(t)),
        "res_trans_mean_mm": meannorm(res_t), "res_trans_rms_mm": rms(res_t),
        "frac_expl_trans": 1.0 - rms(res_t) / r0,
        "rigid_rot_deg": rot_angle_deg(R),
        "rigid_centroid_shift_mm": float(np.linalg.norm(Y.mean(0) - X.mean(0))),
        "rigid_rot_disp_rms_mm": rms(rot_part),
        "rigid_motion_rms_mm": rms(pred_r - X),
        "rigid_t_norm_mm": float(np.linalg.norm(tr)),
        "res_rigid_mean_mm": meannorm(res_r), "res_rigid_rms_mm": rms(res_r),
        "frac_expl_rigid": 1.0 - rms(res_r) / r0,
        "res_rigid_loo_rms_mm": lr,
        "res_rigid_loo_mean_mm": (float(np.mean(loo_r))
                                  if loo_r is not None else np.nan),
        "frac_expl_rigid_loo": 1.0 - lr / r0,
        "res_affine_mean_mm": meannorm(res_a), "res_affine_rms_mm": rms(res_a),
        "frac_expl_affine": 1.0 - rms(res_a) / r0,
        "affine_det": detA, "affine_sv1": float(sv[0]), "affine_sv2": float(sv[1]),
        "affine_sv3": float(sv[2]),
        "affine_mean_scale": float(np.cbrt(abs(detA))) if detA == detA else np.nan,
        "affine_anisotropy": float(sv[0] / sv[2]) if sv[2] == sv[2] else np.nan,
        "affine_dof_ok": bool(n >= 8),     # 12 DoF vs 3n constraints
        "res_affine_loo_rms_mm": la,
        "res_affine_loo_mean_mm": (float(np.mean(loo_a))
                                   if loo_a is not None else np.nan),
        "frac_expl_affine_loo": 1.0 - la / r0,
        "knn_loo_res_mean_mm": (float(np.mean(knn)) if knn is not None else np.nan),
        "knn_loo_res_rms_mm": (float(np.sqrt(np.mean(knn ** 2)))
                               if knn is not None else np.nan),
        "mean_nn_dist_mm": nnd,
        "landmark_spread_rms_mm": rms(X - X.mean(0)),
        "null_frac_expl_rigid_mean": nr_m, "null_frac_expl_rigid_sd": nr_s,
        "null_frac_expl_affine_mean": na_m, "null_frac_expl_affine_sd": na_s,
        "excess_frac_expl_rigid": (1.0 - rms(res_r) / r0) - nr_m,
        "affine_minus_rigid_frac": (rms(res_r) - rms(res_a)) / r0,
    }
    if extra:
        row.update(extra)
    return row


# ------------------------------------------------------------- loaders ----
def load_resect():
    cases = []
    for c in sorted(int(p.name[4:]) for p in RESECT_STAGED.iterdir()
                    if p.is_dir() and p.name.startswith("Case")):
        name = f"Case{c}"
        tag = RESECT_RAW / name / "Landmarks" / f"{name}-MRI-afterUS.tag"
        src = "RESECT_NIFTI_raw"
        if not tag.exists():
            tag, src = RESECT_STAGED / name / f"{name}-MRI-afterUS.tag", "staged"
        if not tag.exists():
            continue
        mri, us = load_tag_pairs(str(tag))
        cases.append((name, us, mri,
                      {"split": "TEST" if c in RESECT_TEST else "TUNE",
                       "extreme": c in RESECT_EXTREME, "source": src}))
    return cases


def _load_remind_dir(root, tag):
    cases = []
    for pid in PIDS:
        t = pd.read_csv(root / pid / f"landmarks_{pid}.csv", comment="#")
        cases.append((pid, t[USC].to_numpy(float), t[MRC].to_numpy(float),
                      {"split": "ALL", "extreme": False, "source": tag}))
    return cases


def load_remind_v11():
    return _load_remind_dir(SNAP / "ReMIND-BS", "ReMIND-BS-1.1_QA-2.0")


def load_remind_v10():
    return _load_remind_dir(SNAP / "v1.0/ReMIND-BS", "ReMIND-BS-1.0_QA-1.0")


def load_remind_unfilt():
    d = pd.read_csv(SNAP / "per_pair_diagnostics_rule.csv")
    cases = []
    for pid in PIDS:
        g = d[d.pid == pid].sort_values("pair_id")
        cases.append((pid, g[USC].to_numpy(float), g[MRC].to_numpy(float),
                      {"split": "ALL", "extreme": False,
                       "source": "P3_unfiltered_172"}))
    return cases


def load_cube():
    cases = []
    for pid in PIDS:
        p = (WD / "data/landmarks_backup/v3_landmarks" / pid /
             f"landmarks_{pid}_clean.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        cases.append((pid,
                      np.array([q["us_RAS_mm"] for q in d["pairs"]], float),
                      np.array([q["mr_RAS_mm"] for q in d["pairs"]], float),
                      {"split": "ALL", "extreme": False, "source": "v3_cube"}))
    return cases


# ------------------------------------------------- registration output ----
def sample_disp(disp_img, pts_world):
    A = np.linalg.inv(disp_img.affine)
    v = (np.c_[pts_world, np.ones(len(pts_world))] @ A.T)[:, :3]
    sh = np.array(disp_img.shape[:3], dtype=float)
    outside = ((v < 0) | (v > sh - 1)).any(axis=1)
    d = np.asarray(disp_img.dataobj, dtype=np.float32)
    u = np.stack([map_coordinates(d[..., c], v.T, order=1, mode="nearest")
                  for c in range(3)], axis=1)
    return u, outside


def method_pred(disp_path, us):
    u, outside = sample_disp(nib.load(str(disp_path)), us)
    return us + u, outside


def affine_from_disp(disp_path, stride=8):
    """Recover the exact 3x4 world affine [A|b] encoded by a rigid/affine
    NiftyReg deformation field converted to disp_mm.nii.gz (x -> x + u(x) is
    affine, so a least-squares fit on the grid is exact).  Returns (A, b,
    max_abs_residual_mm) -- the residual is the validity check."""
    img = nib.load(str(disp_path))
    u = np.asarray(img.dataobj, dtype=np.float64)[::stride, ::stride, ::stride]
    sh = np.asarray(img.dataobj).shape[:3]
    gi = [np.arange(0, sh[k], stride, dtype=np.float64) for k in range(3)]
    ii, jj, kk = np.meshgrid(*gi, indexing="ij")
    X = (np.stack([ii, jj, kk, np.ones_like(ii)], -1) @ img.affine.T)[..., :3]
    X = X.reshape(-1, 3)
    Y = X + u.reshape(-1, 3)
    A, b = fit_affine(X, Y)
    resid = float(np.abs(apply_lin(A, b, X) - Y).max())
    return A, b, resid


# ------------------------------------------------------------- summary ----
SUMM_COLS = ["initial_mean_mm", "initial_rms_mm", "res_trans_rms_mm",
             "res_rigid_mean_mm", "res_rigid_rms_mm", "res_rigid_loo_rms_mm",
             "res_rigid_loo_mean_mm", "res_affine_mean_mm", "res_affine_rms_mm",
             "res_affine_loo_rms_mm", "frac_expl_trans", "frac_expl_rigid",
             "frac_expl_rigid_loo", "frac_expl_affine", "frac_expl_affine_loo",
             "null_frac_expl_rigid_mean", "excess_frac_expl_rigid",
             "affine_minus_rigid_frac", "rigid_rot_deg",
             "rigid_centroid_shift_mm", "rigid_rot_disp_rms_mm",
             "rigid_motion_rms_mm", "affine_mean_scale", "affine_anisotropy",
             "knn_loo_res_mean_mm", "knn_loo_res_rms_mm", "mean_nn_dist_mm",
             "landmark_spread_rms_mm"]


def summarize(df, label):
    r = {"dataset": label, "n_cases": len(df), "n_pairs": int(df.n_pairs.sum())}
    for c in SUMM_COLS:
        v = pd.to_numeric(df[c], errors="coerce").dropna()
        r[f"{c}_mean"] = float(v.mean())
        r[f"{c}_median"] = float(v.median())
        r[f"{c}_sd"] = float(v.std(ddof=1)) if len(v) > 1 else np.nan
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups, rows = {}, []
    for label, loader in [("RESECT", load_resect),
                          ("ReMIND_clean", load_remind_v11),
                          ("ReMIND_v10", load_remind_v10),
                          ("ReMIND_unfilt", load_remind_unfilt),
                          ("Cube_P1", load_cube)]:
        cases = loader()
        groups[label] = cases
        for name, us, mr, extra in cases:
            rows.append(decompose(name, label, us, mr, extra))
        print(f"{label}: {len(cases)} cases, {sum(len(c[1]) for c in cases)} pairs")

    df = pd.DataFrame(rows)
    # pre-op T2 acquisition type (3D SPACE vs 2D thick-slice / FLAIR fallback):
    # a candidate mechanism for a larger residual rigid offset on ReMIND.
    coh = pd.read_csv(WD / "data" / "cohort.csv")[
        ["case", "t2_is_2d", "t2_is_flair"]].rename(columns={"case": "case"})
    df = df.merge(coh, on="case", how="left")
    df["t2_fallback"] = (df.t2_is_2d.fillna(False).astype(bool) |
                         df.t2_is_flair.fillna(False).astype(bool))
    df.to_csv(OUT / "per_case_decomposition.csv", index=False)

    subsets = {
        "RESECT_all20": df[df.dataset == "RESECT"],
        "RESECT_TEST10": df[(df.dataset == "RESECT") & (df.split == "TEST")],
        "RESECT_TEST8_excl_extremes": df[(df.dataset == "RESECT") &
                                         (df.split == "TEST") & (~df.extreme)],
        "RESECT_all18_excl_extremes": df[(df.dataset == "RESECT") & (~df.extreme)],
        "ReMIND_clean": df[df.dataset == "ReMIND_clean"],
        "ReMIND_v10": df[df.dataset == "ReMIND_v10"],
        "ReMIND_unfilt": df[df.dataset == "ReMIND_unfilt"],
        "Cube_P1": df[df.dataset == "Cube_P1"],
    }
    summ = pd.DataFrame([summarize(v, k) for k, v in subsets.items() if len(v)])
    summ.to_csv(OUT / "dataset_summary.csv", index=False)

    # ---------------- Mann-Whitney between datasets ----------------
    METRICS = ["frac_expl_rigid", "frac_expl_rigid_loo", "excess_frac_expl_rigid",
               "frac_expl_trans", "frac_expl_affine", "affine_minus_rigid_frac",
               "rigid_motion_rms_mm", "rigid_centroid_shift_mm", "rigid_rot_deg",
               "rigid_rot_disp_rms_mm", "initial_mean_mm", "initial_rms_mm",
               "res_rigid_mean_mm", "res_rigid_rms_mm", "res_rigid_loo_rms_mm",
               "res_affine_rms_mm", "affine_mean_scale", "affine_anisotropy",
               "knn_loo_res_mean_mm", "mean_nn_dist_mm",
               "landmark_spread_rms_mm"]
    PAIRS = [("ReMIND_clean", "RESECT_all20"),
             ("ReMIND_clean", "RESECT_TEST10"),
             ("ReMIND_clean", "RESECT_all18_excl_extremes"),
             ("ReMIND_clean", "Cube_P1"),
             ("RESECT_all20", "Cube_P1"),
             ("ReMIND_unfilt", "RESECT_all20")]
    mw = []
    for a, b in PAIRS:
        A, B = subsets[a], subsets[b]
        for m in METRICS:
            va = pd.to_numeric(A[m], errors="coerce").dropna().to_numpy()
            vb = pd.to_numeric(B[m], errors="coerce").dropna().to_numpy()
            if len(va) < 3 or len(vb) < 3:
                continue
            st = mannwhitneyu(va, vb, alternative="two-sided")
            mw.append({"group_a": a, "group_b": b, "metric": m,
                       "n_a": len(va), "n_b": len(vb),
                       "median_a": float(np.median(va)),
                       "median_b": float(np.median(vb)),
                       "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                       "U": float(st.statistic), "p_value": float(st.pvalue),
                       "rank_biserial": float(2 * st.statistic /
                                              (len(va) * len(vb)) - 1),
                       "significant_0.05": bool(st.pvalue < 0.05)})
    # within-ReMIND: does the thick-slice / FLAIR pre-op T2 fallback come with
    # a bigger rigid offset?
    rc = subsets["ReMIND_clean"]
    for m in ["initial_mean_mm", "rigid_motion_rms_mm", "rigid_rot_deg",
              "rigid_centroid_shift_mm", "res_rigid_mean_mm",
              "frac_expl_rigid"]:
        va = pd.to_numeric(rc[rc.t2_fallback][m], errors="coerce").dropna()
        vb = pd.to_numeric(rc[~rc.t2_fallback][m], errors="coerce").dropna()
        if len(va) >= 3 and len(vb) >= 3:
            st = mannwhitneyu(va, vb, alternative="two-sided")
            mw.append({"group_a": "ReMIND_T2fallback", "group_b": "ReMIND_T2_3D",
                       "metric": m, "n_a": len(va), "n_b": len(vb),
                       "median_a": float(np.median(va)),
                       "median_b": float(np.median(vb)),
                       "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
                       "U": float(st.statistic), "p_value": float(st.pvalue),
                       "rank_biserial": float(2 * st.statistic /
                                              (len(va) * len(vb)) - 1),
                       "significant_0.05": bool(st.pvalue < 0.05)})
    mwdf = pd.DataFrame(mw)
    mwdf.to_csv(OUT / "mannwhitney.csv", index=False)

    # ------------- method output, oracle rigid, image-driven rigid -------
    ARMS = {"ReMIND_clean": (WD / "results/residual_smooth_test",
                             "modal_residual_s15"),
            "ReMIND_v10": (WD / "results/residual_smooth_test",
                           "modal_residual_s15"),
            "RESECT": (WD / "results/resect_tre", "modal_residual_s15")}
    pair_rows, case_rows = [], []
    for label in ["ReMIND_clean", "ReMIND_v10", "RESECT"]:
        root, arm = ARMS[label]
        for name, us, mr, extra in groups[label]:
            dp = root / name / arm / "disp_mm.nii.gz"
            if not dp.exists():
                continue
            pred, outside = method_pred(dp, us)
            tre = np.linalg.norm(pred - mr, axis=1)
            init = np.linalg.norm(us - mr, axis=1)

            R0, t0 = fit_rigid(us, mr)                 # oracle rigid, initial
            rigid_only = np.linalg.norm(apply_lin(R0, t0, us) - mr, axis=1)
            R1, t1 = fit_rigid(pred, mr)               # oracle rigid on output
            tre_or = np.linalg.norm(apply_lin(R1, t1, pred) - mr, axis=1)
            loo1 = loo_residual(pred, mr, "rigid")
            if len(pred) >= 4:
                A1, b1 = fit_affine(pred, mr)
                tre_oa = np.linalg.norm(apply_lin(A1, b1, pred) - mr, axis=1)
            else:
                tre_oa = np.full(len(pred), np.nan)

            row = {"dataset": label, "case": name, "n_pairs": len(us),
                   "split": extra.get("split", ""),
                   "extreme": extra.get("extreme", False),
                   "n_outside_grid": int(outside.sum()),
                   "initial_mm": float(init.mean()),
                   "method_s15_mm": float(tre.mean()),
                   "oracle_rigid_only_mm": float(rigid_only.mean()),
                   "oracle_rigid_after_method_mm": float(tre_or.mean()),
                   "oracle_rigid_after_method_loo_mm":
                       (float(np.mean(loo1)) if loo1 is not None else np.nan),
                   "oracle_affine_after_method_mm": float(np.mean(tre_oa)),
                   "method_residual_rot_deg": rot_angle_deg(R1),
                   "method_residual_centroid_shift_mm":
                       float(np.linalg.norm(mr.mean(0) - pred.mean(0))),
                   "gain_oracle_rigid_mm": float(tre.mean() - tre_or.mean()),
                   "frac_of_residual_error_rigid":
                       1.0 - float(tre_or.mean()) / float(tre.mean())}

            # image-driven rigid (honest, no landmarks): NiftyReg reg_aladin.
            # ReMIND: computed here (results/rigid_decomposition/niftyreg_rigid).
            # RESECT: the pre-existing, published niftyreg_affine arm (same
            # reg_aladin -rigOnly recipe), reused verbatim.
            nrd = {}
            srcs = {"plain": NR_RIGID / name / "plain" / "disp_mm.nii.gz",
                    "rmask": NR_RIGID / name / "rmask" / "disp_mm.nii.gz"}
            if label == "RESECT":
                srcs["plain"] = (WD / "results/resect_tre" / name /
                                 "niftyreg_affine" / "disp_mm.nii.gz")
            for var, p in srcs.items():
                if p.exists():
                    pr, _ = method_pred(p, us)
                    nrd[var] = np.linalg.norm(pr - mr, axis=1)
                    row[f"img_rigid_{var}_mm"] = float(nrd[var].mean())
                    row[f"img_rigid_{var}_gain_mm"] = float(init.mean() -
                                                            nrd[var].mean())

            # end-to-end image-driven rigid pre-stage + our method, re-run
            # from scratch in the rigid-aligned frame (moving image + SynthSeg
            # resampled through A, basis rebuilt, modal_residual_s15 re-fitted).
            # Final map: x_us -> A (x_us + u_rig(x_us)).
            rid = OUT / "rigidinit" / name / "modal_residual_s15" / "disp_mm.nii.gz"
            aff_p = NR_RIGID / name / "plain" / "disp_mm.nii.gz"
            if rid.exists() and aff_p.exists():
                A2, b2, rr = affine_from_disp(aff_p)
                p2, _ = method_pred(rid, us)
                t2 = np.linalg.norm(apply_lin(A2, b2, p2) - mr, axis=1)
                nrd["rigidinit"] = t2
                row["rigidinit_modal_mm"] = float(t2.mean())
                row["rigidinit_affine_fit_resid_mm"] = rr

            # ORACLE rigid pre-stage + our method, same pipeline but with the
            # rigid fitted to the evaluation landmarks (upper bound, not
            # deployable).  Final map: x_us -> A_oracle (x_us + u(x_us)).
            oid = OUT / "oracleinit" / name / "modal_residual_s15" / "disp_mm.nii.gz"
            oaf = OUT / "oracleinit" / name / "oracle_aff.txt"
            if oid.exists() and oaf.exists():
                Ao = np.loadtxt(oaf)
                p3, _ = method_pred(oid, us)
                t3 = np.linalg.norm(apply_lin(Ao[:3, :3], Ao[:3, 3], p3) - mr,
                                    axis=1)
                nrd["oracleinit"] = t3
                row["oracleinit_modal_mm"] = float(t3.mean())
            case_rows.append(row)
            for i in range(len(us)):
                pr = {"dataset": label, "case": name, "idx": i,
                      "initial_mm": float(init[i]),
                      "method_s15_mm": float(tre[i]),
                      "oracle_rigid_only_mm": float(rigid_only[i]),
                      "oracle_rigid_after_method_mm": float(tre_or[i])}
                for var, v in nrd.items():
                    pr[f"img_rigid_{var}_mm"] = float(v[i])
                pair_rows.append(pr)
    cdf = pd.DataFrame(case_rows)
    cdf.to_csv(OUT / "oracle_rigid_remind.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(OUT / "per_pair_oracle.csv", index=False)

    # ---------------- paired tests on the oracle / image-driven ----------
    GRP = {"ReMIND_clean": cdf[cdf.dataset == "ReMIND_clean"],
           "ReMIND_v10": cdf[cdf.dataset == "ReMIND_v10"],
           "RESECT_TEST10": cdf[(cdf.dataset == "RESECT") & (cdf.split == "TEST")],
           "RESECT_TEST8": cdf[(cdf.dataset == "RESECT") & (cdf.split == "TEST")
                               & (~cdf.extreme)]}
    wil = []
    for label, sel in GRP.items():
        if len(sel) < 5:
            continue
        base_cols = ["initial_mm", "method_s15_mm"]
        for ref in base_cols:
            for col in ["oracle_rigid_only_mm", "oracle_rigid_after_method_mm",
                        "oracle_affine_after_method_mm", "img_rigid_plain_mm",
                        "img_rigid_rmask_mm", "rigidinit_modal_mm",
                        "oracleinit_modal_mm"]:
                if col not in sel.columns or col == ref:
                    continue
                a = pd.to_numeric(sel[ref], errors="coerce").to_numpy(float)
                b = pd.to_numeric(sel[col], errors="coerce").to_numpy(float)
                ok = ~(np.isnan(a) | np.isnan(b))
                if ok.sum() < 5:
                    continue
                try:
                    p = float(wilcoxon(a[ok], b[ok]).pvalue)
                except ValueError:
                    p = np.nan
                wil.append({"group": label, "reference": ref, "arm": col,
                            "n": int(ok.sum()), "mean_reference": float(a[ok].mean()),
                            "mean_arm": float(b[ok].mean()),
                            "delta_mm": float((a[ok] - b[ok]).mean()),
                            "n_cases_arm_better": int((b[ok] < a[ok]).sum()),
                            "p_value": p})
    pd.DataFrame(wil).to_csv(OUT / "oracle_wilcoxon.csv", index=False)

    # ---------------- ceiling table ----------------
    ceil = []
    for key, sub in subsets.items():
        if not len(sub):
            continue
        ds = sub.dataset.iloc[0]
        sel = cdf[(cdf.dataset == ds) & cdf.case.isin(sub.case)]
        got = len(sel) == len(sub)
        r = {"group": key, "n_cases": len(sub),
             "n_pairs": int(sub.n_pairs.sum()),
             "initial_mean_mm": float(sub.initial_mean_mm.mean()),
             "ceiling_post_rigid_mm": float(sub.res_rigid_mean_mm.mean()),
             "ceiling_post_rigid_loo_mm":
                 float(pd.to_numeric(sub.res_rigid_loo_mean_mm,
                                     errors="coerce").mean()),
             "ceiling_post_affine_mm": float(sub.res_affine_mean_mm.mean()),
             "knn_loo_res_mean_mm":
                 float(pd.to_numeric(sub.knn_loo_res_mean_mm,
                                     errors="coerce").mean()),
             "method_s15_mm": float(sel.method_s15_mm.mean()) if got else np.nan,
             "method_minus_ceiling_mm":
                 float(sel.method_s15_mm.mean() - sub.res_rigid_mean_mm.mean())
                 if got else np.nan,
             "oracle_rigid_after_method_mm":
                 float(sel.oracle_rigid_after_method_mm.mean()) if got else np.nan,
             "frac_expl_rigid_mean": float(sub.frac_expl_rigid.mean()),
             "frac_expl_rigid_loo_mean":
                 float(pd.to_numeric(sub.frac_expl_rigid_loo,
                                     errors="coerce").mean())}
        ceil.append(r)
    ceildf = pd.DataFrame(ceil)
    ceildf.to_csv(OUT / "ceiling_analysis.csv", index=False)

    # ------------- does the rigid component predict our failures? --------
    cor = []
    for label in ["ReMIND_clean", "RESECT"]:
        d1 = df[df.dataset == ("ReMIND_clean" if label == "ReMIND_clean"
                               else "RESECT")].set_index("case")
        d2 = cdf[cdf.dataset == label].set_index("case")
        j = d1.join(d2[["method_s15_mm", "gain_oracle_rigid_mm",
                        "frac_of_residual_error_rigid"]], how="inner")
        for x in ["frac_expl_rigid", "rigid_motion_rms_mm",
                  "rigid_centroid_shift_mm", "rigid_rot_deg", "initial_mean_mm"]:
            for y in ["method_s15_mm", "gain_oracle_rigid_mm",
                      "frac_of_residual_error_rigid"]:
                rho, p = spearmanr(j[x], j[y])
                cor.append({"dataset": label, "n": len(j), "x": x, "y": y,
                            "spearman_rho": float(rho), "p_value": float(p)})
    pd.DataFrame(cor).to_csv(OUT / "correlations.csv", index=False)

    # ---------------- json digest ----------------
    dig = {"generated_utc": pd.Timestamp.utcnow().isoformat(),
           "null_reps": NULL_REPS,
           "landmark_sets": {
               "ReMIND_clean": "landmarks_snapshot/ReMIND-BS -- frozen copy of "
                               "release/landmarks/ReMIND-BS as of 2026-08-10 "
                               "21:25:08Z: set ReMIND-BS v1.1, QA rule "
                               "ReMIND-BS-QA-2.0 (A1/A2/A3, C3 retired by "
                               "AMENDMENT.md), 14 subjects / 170 pairs, "
                               "2 swap corrections",
               "ReMIND_v10": "landmarks_snapshot/v1.0/ReMIND-BS (155 pairs, "
                             "QA-1.0; the set behind "
                             "results/landmark_qc/clean_TRE_table.csv)",
               "RESECT": "D:/ReMIND_downloads/RESECT/NIFTI/CaseN/Landmarks/"
                         "CaseN-MRI-afterUS.tag via scripts/resect_tags.py",
               "Cube_P1": "data/landmarks_backup/v3_landmarks/<pid>/"
                          "landmarks_<pid>_clean.json (oracle-prealigned)"},
           "datasets": {k: {"n_cases": int(len(v)),
                            "n_pairs": int(v.n_pairs.sum()),
                            "initial_mean_mm": float(v.initial_mean_mm.mean()),
                            "frac_expl_rigid_mean": float(v.frac_expl_rigid.mean()),
                            "frac_expl_rigid_loo_mean": float(
                                pd.to_numeric(v.frac_expl_rigid_loo,
                                              errors="coerce").mean()),
                            "null_frac_expl_rigid_mean": float(
                                v.null_frac_expl_rigid_mean.mean()),
                            "rigid_motion_rms_mm_mean": float(
                                v.rigid_motion_rms_mm.mean()),
                            "rigid_rot_deg_mean": float(v.rigid_rot_deg.mean()),
                            "res_rigid_mean_mm_mean": float(
                                v.res_rigid_mean_mm.mean()),
                            "res_affine_mean_mm_mean": float(
                                v.res_affine_mean_mm.mean())}
                        for k, v in subsets.items() if len(v)}}
    (OUT / "decomposition.json").write_text(json.dumps(dig, indent=1))

    pd.set_option("display.width", 240)
    fmt = lambda x: f"{x:.3f}"
    print("\n== per-case decomposition ==")
    print(df[["dataset", "case", "n_pairs", "initial_mean_mm", "rigid_rot_deg",
              "rigid_centroid_shift_mm", "rigid_motion_rms_mm",
              "res_rigid_mean_mm", "frac_expl_rigid", "frac_expl_rigid_loo",
              "null_frac_expl_rigid_mean", "frac_expl_affine"]]
          .to_string(index=False, float_format=fmt))
    print("\n== dataset summary ==")
    print(summ[["dataset", "n_cases", "n_pairs", "initial_mean_mm_mean",
                "rigid_motion_rms_mm_mean", "rigid_rot_deg_mean",
                "rigid_centroid_shift_mm_mean", "res_rigid_mean_mm_mean",
                "frac_expl_rigid_mean", "frac_expl_rigid_loo_mean",
                "null_frac_expl_rigid_mean_mean", "frac_expl_affine_mean",
                "affine_minus_rigid_frac_mean"]]
          .to_string(index=False, float_format=fmt))
    print("\n== Mann-Whitney (key metrics) ==")
    print(mwdf[mwdf.metric.isin(["frac_expl_rigid", "frac_expl_rigid_loo",
                                 "excess_frac_expl_rigid", "rigid_motion_rms_mm",
                                 "rigid_centroid_shift_mm", "rigid_rot_deg",
                                 "res_rigid_mean_mm"])]
          .to_string(index=False, float_format=fmt))
    print("\n== ceiling ==")
    print(ceildf.to_string(index=False, float_format=fmt))
    print("\n== oracle / image-driven rigid per case ==")
    print(cdf.to_string(index=False, float_format=fmt))
    print("\n== paired tests ==")
    print(pd.DataFrame(wil).to_string(index=False, float_format=fmt))
    print("\n== correlations ==")
    print(pd.DataFrame(cor).to_string(index=False, float_format=fmt))
    print("\noutputs ->", str(OUT).replace("/mnt/c/", "C:/"))


if __name__ == "__main__":
    main()
