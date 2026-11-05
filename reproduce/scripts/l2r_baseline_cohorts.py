"""Official Learn2Reg 2024 / ReMIND2Reg NiftyReg baseline on the paper's three
landmark cohorts, compared case by case with our shipped (gated) method.

Baseline = the challenge's script (github.com/ReubenDo/ReMIND2Reg,
niftyreg/run_niftyreg.py), verbatim flags:
    reg_aladin -ref US -flo MR -rmask US_mask [-fmask MR_mask] -noSym -nac -ln 2 -lp 10
rigid then affine (no -rigOnly).  -fmask is only used by the symmetric
version, so with -noSym it has no effect and is omitted.  As in the challenge
data, the iUS is resampled to a 0.5 mm isotropic box covering its field of
view, and the reference mask is the iUS field of view (iUS > 0).  MR inputs
are the same volumes every arm of the paper uses (RESECT FLAIR, BITE T1+Gd).
Prediction for an iUS landmark: A x_us (NiftyReg convention Affine*Ref=Flo,
validated in the paper).

Ours = the shipped gated method, single configuration (method_arms.py:
accepted pose: A(x_us + u_pose) with the polished pose; abstained:
x_us + u_identity, the deformation-only arm); the summary prints the cohort
means of single_config.py's SCORES.csv next to ours as a check.
TRE30 uses the same definition as tre30.py (70th percentile per case, then
mean over cases).  Resumable; outputs in results/l2r_niftyreg_baseline/.

Usage: python l2r_baseline_cohorts.py
"""
import csv
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import stats
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rg_cohorts                                                    # noqa
import method_arms as M                                              # noqa
from reg_core import resample_to_grid                                # noqa
from resect_tags import load_tag_pairs as rt                         # noqa
from bite_tags import load_tag_pairs as bt                           # noqa

W = Path("/mnt/c/ReMIND_work")
R = W / "results"
OUT = R / "l2r_niftyreg_baseline"
NR = Path("/home/ncrhurh/anaconda3/envs/niftyreg/bin")
SAT = M.SAT
LABEL = {"resect_before": "RESECT pre-resection", "resect_after": "RESECT post-resection",
         "bite_g4": "BITE post-resection"}
TAG = {"resect_before": lambda c: (W / f"data/resect_before/{c}/{c}-MRI-beforeUS.tag", rt),
       "resect_after": lambda c: (W / f"data/resect/{c}/{c}-MRI-afterUS.tag", rt),
       "bite_g4": lambda c: (W / f"data/bite/group4/{c}/Tags.tag", bt)}
COH = {k: dict(tag=TAG[k],
               aff=lambda c, k=k: M.pose(k, c),
               umod=lambda c, k=k: M.u_pose(k, c),
               ufall=lambda c, k=k: M.u_identity(k, c))
       for k in ("resect_before", "resect_after", "bite_g4")}


def disp_at(upath, pts):
    u = nib.load(str(upath))
    a = np.asarray(u.dataobj, np.float32)
    inv = np.linalg.inv(u.affine)
    v = pts @ inv[:3, :3].T + inv[:3, 3]
    return np.stack([map_coordinates(a[..., c], [v[:, 0], v[:, 1], v[:, 2]], order=1, mode="nearest")
                     for c in range(3)], -1)


def table_col(path, col):
    p = R / path
    if not p.exists():
        return {}
    return {r["case"]: float(r[col]) for r in csv.DictReader(open(p)) if r.get(col) not in (None, "", "nan")}


def horn(a, b):
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rm, cb - Rm @ ca


def half_mm_box(img, crop_fov=False, margin=2.0):
    """0.5 mm isotropic box.  Default: the image extent (RESECT/BITE iUS volumes
    are already bounded by their field of view).  crop_fov=True: bounding box of
    the nonzero iUS voxels + margin, for volumes stored on a larger grid."""
    if crop_fov:
        msk = np.asarray(img.dataobj) > 0
        idx = [np.where(msk.any(axis=tuple(x for x in range(3) if x != ax)))[0] for ax in range(3)]
        lo_i = [float(i.min()) for i in idx]
        hi_i = [float(i.max() + 1) for i in idx]
        c = np.array([[i, j, k] for i in (lo_i[0], hi_i[0]) for j in (lo_i[1], hi_i[1])
                      for k in (lo_i[2], hi_i[2])])
    else:
        sh = np.array(img.shape[:3], float)
        c = np.array([[i * sh[0], j * sh[1], k * sh[2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    w = c @ img.affine[:3, :3].T + img.affine[:3, 3]
    lo, hi = w.min(0), w.max(0)
    if crop_fov:
        lo, hi = lo - margin, hi + margin
    dims = np.maximum(np.ceil((hi - lo) / 0.5).astype(int), 8)
    aff = np.diag([0.5, 0.5, 0.5, 1.0])
    aff[:3, 3] = lo
    arr = np.asarray(img.dataobj, np.float32)
    us = resample_to_grid(img, aff, dims).astype(np.float32)
    m = resample_to_grid(img, aff, dims, mode="nearest", data=(arr > 0).astype(np.float32)) > 0.5
    return us, m.astype(np.uint8), aff


def run_baseline(usp, mrp, d, crop_fov=False):
    """Returns the affine; resumable."""
    d.mkdir(parents=True, exist_ok=True)
    affp = d / "aff.txt"
    if affp.exists():
        return np.loadtxt(affp)
    us05, m05, aff05 = half_mm_box(nib.load(str(usp)), crop_fov=crop_fov)
    nib.save(nib.Nifti1Image(us05, aff05), str(d / "us05.nii.gz"))
    nib.save(nib.Nifti1Image(m05, aff05), str(d / "us05_mask.nii.gz"))
    cmd = [NR / "reg_aladin", "-ref", d / "us05.nii.gz", "-flo", mrp, "-rmask", d / "us05_mask.nii.gz",
           "-noSym", "-aff", d / "aff_tmp.txt", "-res", d / "res.nii.gz", "-nac", "-ln", "2", "-lp", "10"]
    for k in range(8):
        # reg_aladin segfaults (rc -11) sporadically with the default 8 OpenMP threads, and on every run for external
        # Subject-3 with its 512x512x160 3D T1 (2026-11-05); with 1 or 2 threads it runs and returns the identical
        # matrix, so the last four attempts use 2 threads.
        run = cmd + (["-omp", "2"] if k >= 4 else [])
        with open(d / "reg_aladin.log", "a") as f:
            f.write("\n$ " + " ".join(map(str, run)) + "\n")
            rc = subprocess.run([str(x) for x in run], stdout=f, stderr=subprocess.STDOUT).returncode
        if rc == 0 and (d / "aff_tmp.txt").exists():
            break
        print(f"    reg_aladin rc={rc}, retry {k + 1}", flush=True)
    else:
        raise RuntimeError("reg_aladin failed 8x")
    (d / "aff_tmp.txt").rename(affp)
    for p in ("res.nii.gz", "us05.nii.gz"):
        (d / p).unlink(missing_ok=True)
    return np.loadtxt(affp)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for coh, P in COH.items():
        spec = rg_cohorts.install(coh)
        for r in csv.DictReader(open(R / f"rigid_gate/search_tight/{coh}/PER_CASE.csv")):
            c, rot = r["case"], float(r["rot_deg"])
            try:
                tagp, loader = P["tag"](c)
                if not tagp.exists():
                    print(f"[l2r] {coh} {c}: no tag, skipped", flush=True)
                    continue
                mri, us = loader(str(tagp))
                mri, us = np.asarray(mri, float), np.asarray(us, float)
                usp, mrp = spec["paths"](c)[:2]
                A = run_baseline(usp, mrp, OUT / coh / c)
                e_init = np.linalg.norm(us - mri, axis=1)
                e_l2r = np.linalg.norm(us @ A[:3, :3].T + A[:3, 3] - mri, axis=1)
                if rot < SAT:
                    Ap = np.loadtxt(P["aff"](c))
                    y = us + disp_at(P["umod"](c), us)
                    pred = y @ Ap[:3, :3].T + Ap[:3, 3]
                else:
                    pred = us + disp_at(P["ufall"](c), us)
                e_ours = np.linalg.norm(pred - mri, axis=1)
                Rb, tb = horn(us, mri)
                e_bf = np.linalg.norm(us @ Rb.T + tb - mri, axis=1)
                sv = np.linalg.svd(A[:3, :3], compute_uv=False)
                row = dict(cohort=coh, case=c, n=len(us), abstained=rot >= SAT,
                           initial=e_init.mean(), l2r_niftyreg=e_l2r.mean(), ours=e_ours.mean(),
                           bestfit_rigid=e_bf.mean(),
                           initial_tre30=np.percentile(e_init, 70), l2r_tre30=np.percentile(e_l2r, 70),
                           ours_tre30=np.percentile(e_ours, 70),
                           l2r_scale_min=sv.min(), l2r_scale_max=sv.max())
                rows.append(row)
                print(f"[l2r] {coh} {c}: initial {row['initial']:.2f} | L2R NiftyReg {row['l2r_niftyreg']:.2f} | "
                      f"ours {row['ours']:.2f} | scales {sv.round(3).tolist()}", flush=True)
            except Exception as ex:                                  # noqa: BLE001
                print(f"[l2r] {coh} {c} FAILED: {ex}", flush=True)

    with open(OUT / "PER_CASE.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    sc = M.scores()
    PAPER_MEAN = {k: float(np.mean([r["final"] for (kk, _), r in sc.items() if kk == k])) for k in COH}
    lines = ["cohort | n | initial | L2R NiftyReg baseline | ours | best-fit rigid | "
             "TRE30 init / L2R / ours | ours better | p (Wilcoxon) | L2R worse than initial | ours worse than initial"]
    for coh in list(COH) + ["ALL"]:
        rr = [r for r in rows if coh == "ALL" or r["cohort"] == coh]
        if not rr:
            continue
        g = lambda k: np.array([r[k] for r in rr], float)                          # noqa: E731
        a, b = g("ours"), g("l2r_niftyreg")
        p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue
        chk = f" (SCORES {PAPER_MEAN[coh]:.2f})" if coh in PAPER_MEAN else ""
        lines.append(
            f"{LABEL.get(coh, 'ALL'):22s} | {len(rr)} | {g('initial').mean():.2f} ± {g('initial').std(ddof=1):.2f} | "
            f"{b.mean():.2f} ± {b.std(ddof=1):.2f} (median {np.median(b):.2f}) | "
            f"{a.mean():.2f} ± {a.std(ddof=1):.2f} (median {np.median(a):.2f}){chk} | "
            f"{g('bestfit_rigid').mean():.2f} | "
            f"{g('initial_tre30').mean():.2f} / {g('l2r_tre30').mean():.2f} / {g('ours_tre30').mean():.2f} | "
            f"{int((a < b).sum())}/{len(rr)} | {p:.4f} | "
            f"{int((b > g('initial')).sum())} (max +{max(0, (b - g('initial')).max()):.2f}) | "
            f"{int((a > g('initial')).sum())} (max +{max(0, (a - g('initial')).max()):.2f})")
    sc = [r for r in rows if r["l2r_scale_max"] > 1.10 or r["l2r_scale_min"] < 0.90]
    lines.append(f"L2R affine with a scale outside 0.90-1.10: {len(sc)} cases "
                 f"{[(r['cohort'], r['case'], round(r['l2r_scale_min'], 2), round(r['l2r_scale_max'], 2)) for r in sc]}")
    (OUT / "SUMMARY.txt").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print("L2R DONE")


if __name__ == "__main__":
    main()
