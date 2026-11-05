"""DeedsSSC (Heinrich, CuRIOUS 2018) reproduced with the public deedsBCV tool
(github.com/mattiaspaul/deedsBCV, MIT) on the paper's three landmark cohorts,
following the recipe of the CuRIOUS 2018 report (Xiao et al. 2020, Sec. III-B):
scans resampled to isotropic 0.5 mm, linear pre-alignment (linearBCV: SSC
correspondences + least-trimmed-squares affine), nonlinear deeds with the
default parameters, and a final 6-parameter rigid fit to the nonlinearly
warped landmarks ("constrained to follow a rigid 6-parameter transform for
improved robustness").  Landmarks are transported by warping a sphere-label
image of the MR landmarks with deeds' own -S option (nearest neighbour), so no
displacement-field convention is assumed.  The 0.5 mm box is the iUS extent
(no padding, as in the challenge: the MR is resliced into the iUS space; a
first attempt with a 25 mm zero halo around the iUS made linearBCV's
least-trimmed-squares fit lock onto the halo and fail).  MR landmarks whose
sphere falls outside the box are not transported directly; they are covered
by the final rigid fit, which is estimated on the transported ones.
Comparison only (review question: which published methods are open?); the
manuscript is not touched.
Outputs: /tmp/deeds_ssc/<cohort>/<case>/ (scratch), results/deeds_ssc/PER_CASE.csv, SUMMARY.txt.
Usage: python deeds_ssc_cohorts.py        (run + summary)
       python deeds_ssc_cohorts.py score  (summary only)
"""
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import nibabel as nib
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rg_cohorts                                                    # noqa
import method_arms as M                                              # noqa
import l2r_baseline_cohorts as L                                     # noqa
from reg_core import resample_to_grid                                # noqa

DEEDS = Path.home() / "deedsBCV"
RIGID = os.environ.get("DEEDS_RIGID", "0") == "1"       # linearBCV -R 1 (rigid pre-alignment) variant
TAGV = "_rigid" if RIGID else ""
TMP = Path("/tmp/deeds_ssc" + TAGV)
OUT = M.R / ("deeds_ssc" + TAGV)
PAD, SP, RAD = 0.0, 0.5, 1.5            # mm padding, voxel size, sphere radius
ORDER = [c for c in ["resect_after", "bite_g4", "resect_before"]
         if c in os.environ.get("DEEDS_COHORTS", "resect_after,bite_g4,resect_before").split(",")]


def padded_box(img):
    sh = np.array(img.shape[:3], float)
    c = np.array([[i * sh[0], j * sh[1], k * sh[2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    w = c @ img.affine[:3, :3].T + img.affine[:3, 3]
    lo, hi = w.min(0) - PAD, w.max(0) + PAD
    dims = np.ceil((hi - lo) / SP).astype(int)
    aff = np.diag([SP, SP, SP, 1.0])
    aff[:3, 3] = lo
    return aff, dims


def spheres(mri, aff, dims):
    lab = np.zeros(tuple(int(x) for x in dims), np.int16)
    inv = np.linalg.inv(aff)
    r = int(round(RAD / SP))
    ii, jj, kk = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    ball = (ii ** 2 + jj ** 2 + kk ** 2) <= r * r
    offs = list(zip(ii[ball], jj[ball], kk[ball]))
    for n, p in enumerate(mri):
        v = np.round(p @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        for di, dj, dk in offs:
            a, b, c = v[0] + di, v[1] + dj, v[2] + dk
            if 0 <= a < dims[0] and 0 <= b < dims[1] and 0 <= c < dims[2]:
                lab[a, b, c] = n + 1
    return lab


def run(cmd, log):
    with open(log, "a") as f:
        f.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        rc = subprocess.run([str(x) for x in cmd], stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise RuntimeError(f"{Path(str(cmd[0])).name} rc={rc}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    csvp = OUT / "PER_CASE.csv"
    done = set()
    if csvp.exists():
        done = {(r["cohort"], r["case"]) for r in csv.DictReader(open(csvp))}
    else:
        csvp.write_text("cohort,case,n,n_found,initial,deeds_raw,deeds_rigidfit,dims,wall_s\n")
    for coh in ORDER:
        spec = rg_cohorts.install(coh)
        for r in csv.DictReader(open(M.R / f"rigid_gate/search_tight/{coh}/PER_CASE.csv")):
            c = r["case"]
            if (coh, c) in done:
                continue
            t0 = time.time()
            d = TMP / coh / c
            d.mkdir(parents=True, exist_ok=True)
            log = d / "deeds.log"
            try:
                tagp, loader = L.TAG[coh](c)
                mri, us = loader(str(tagp))
                mri, us = np.asarray(mri, float), np.asarray(us, float)
                usp, mrp = spec["paths"](c)[:2]
                usi, mri_img = nib.load(str(usp)), nib.load(str(mrp))
                aff, dims = padded_box(usi)
                fx = resample_to_grid(usi, aff, dims).astype(np.float32)
                mv = resample_to_grid(mri_img, aff, dims).astype(np.float32)
                nib.save(nib.Nifti1Image(fx, aff), str(d / "us05.nii.gz"))
                nib.save(nib.Nifti1Image(mv, aff), str(d / "mr05.nii.gz"))
                nib.save(nib.Nifti1Image(spheres(mri, aff, dims), aff), str(d / "spheres.nii.gz"))
                run([DEEDS / "linearBCV", "-F", d / "us05.nii.gz", "-M", d / "mr05.nii.gz", "-O", d / "lin"]
                    + (["-R", "1"] if RIGID else []), log)
                run([DEEDS / "deedsBCV", "-F", d / "us05.nii.gz", "-M", d / "mr05.nii.gz",
                     "-A", d / "lin_matrix.txt", "-S", d / "spheres.nii.gz", "-O", d / "def"], log)
                seg = np.asarray(nib.load(str(d / "def_deformed_seg.nii.gz")).dataobj)
                warped, found = np.full_like(mri, np.nan), []
                for n in range(len(mri)):
                    idx = np.argwhere(seg == n + 1)
                    if len(idx):
                        cen = idx.mean(0)
                        warped[n] = cen @ aff[:3, :3].T + aff[:3, 3]
                        found.append(n)
                found = np.array(found, int)
                e_init = np.linalg.norm(us - mri, axis=1).mean()
                raw = np.linalg.norm(warped[found] - us[found], axis=1).mean() if len(found) else np.nan
                if len(found) >= 3:
                    Rm, t = L.horn(mri[found], warped[found])
                    fit = np.linalg.norm(mri @ Rm.T + t - us, axis=1).mean()
                else:
                    fit = np.nan
                wall = time.time() - t0
                dims_s = "x".join(str(int(x)) for x in dims)
                with open(csvp, "a") as f:
                    f.write(f"{coh},{c},{len(mri)},{len(found)},{e_init:.4f},{raw:.4f},{fit:.4f},{dims_s},{wall:.0f}\n")
                print(f"[deeds] {coh} {c}: initial {e_init:.2f} | deeds raw {raw:.2f} | +rigid fit {fit:.2f} "
                      f"| labels {len(found)}/{len(mri)} | {wall:.0f}s", flush=True)
                for p in ("us05.nii.gz", "mr05.nii.gz", "def_deformed.nii.gz", "def_displacements.dat"):
                    (d / p).unlink(missing_ok=True)
            except Exception as ex:                                  # noqa: BLE001
                print(f"[deeds] {coh} {c} FAILED: {ex}", flush=True)
    summary()


def summary():
    rows = list(csv.DictReader(open(OUT / "PER_CASE.csv")))
    S = M.scores()
    base = {(r["cohort"], r["case"]): float(r["l2r_niftyreg"])
            for r in csv.DictReader(open(M.R / "l2r_niftyreg_baseline/PER_CASE.csv"))}

    def wil(a, b):
        ok = ~(np.isnan(a) | np.isnan(b))
        return float(stats.wilcoxon(a[ok], b[ok], zero_method="zsplit").pvalue)

    Ls = []
    for coh in ORDER:
        rr = [r for r in rows if r["cohort"] == coh]
        if not rr:
            continue
        g = lambda k: np.array([float(r[k]) for r in rr])                        # noqa: E731
        ini, raw, fit = g("initial"), g("deeds_raw"), g("deeds_rigidfit")
        ours = np.array([S[(coh, r["case"])]["final"] for r in rr])
        l2r = np.array([base[(coh, r["case"])] for r in rr])
        Ls.append(f"\n=== {L.LABEL[coh]}  n={len(rr)}  (labels found {int(g('n_found').sum())}/{int(g('n').sum())}) ===")
        Ls.append(f"  initial          {ini.mean():.2f} +- {ini.std(ddof=1):.2f}")
        Ls.append(f"  DeedsSSC raw     {np.nanmean(raw):.2f} +- {np.nanstd(raw, ddof=1):.2f}, median {np.nanmedian(raw):.2f}, "
                  f"max {np.nanmax(raw):.2f}; worse than initial {int(np.nansum(raw > ini))}")
        Ls.append(f"  DeedsSSC +rigid  {np.nanmean(fit):.2f} +- {np.nanstd(fit, ddof=1):.2f}, median {np.nanmedian(fit):.2f}, "
                  f"max {np.nanmax(fit):.2f}; worse than initial {int(np.nansum(fit > ini))}")
        Ls.append(f"  NiftyReg baseline {l2r.mean():.2f} | ours {ours.mean():.2f}")
        Ls.append(f"  ours vs DeedsSSC+rigid: p={wil(ours, fit):.4f}, ours better {int(np.nansum(ours < fit))}/{len(rr)}; "
                  f"ours vs raw: p={wil(ours, raw):.4f}, ours better {int(np.nansum(ours < raw))}/{len(rr)}; "
                  f"DeedsSSC+rigid vs NiftyReg: p={wil(fit, l2r):.4f}, deeds better {int(np.nansum(fit < l2r))}/{len(rr)}")
        Ls.append(f"  mean wall {g('wall_s').mean():.0f} s/case")
    (OUT / "SUMMARY.txt").write_text("\n".join(Ls))
    print("\n".join(Ls))
    print("DEEDS DONE")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "score":
        summary()
    else:
        main()
